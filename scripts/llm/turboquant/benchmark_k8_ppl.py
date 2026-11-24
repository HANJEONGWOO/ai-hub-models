# SPDX-License-Identifier: BSD-3-Clause
"""Score the four frozen ICEIC2027 bundles once on the legacy WikiText windows.

No build, deployment, codec changes, generation timing, or automatic retries.
Reuse the repeated-performance experiment's host/device hash verification.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
from pathlib import Path

from benchmark_asymmetric_once import ARTIFACTS, digest, execute, read, save_new
from benchmark_k8_repeat import (
    ASSETS,
    GROUPS,
    MASKS,
    SERIAL,
    adb,
    config,
    now,
    snapshot,
    verify_freeze,
)

SOURCE = ARTIFACTS / "iceic2027_k8_repeat_20261027"
WINDOWS = tuple(f"wikitext_w{i}.bin" for i in range(4))


def score_row(data: dict, group: str, window: str, token_hash: str) -> dict:
    sessions = data.get("sessions", [])
    checks = {
        "teacher_forced": data["mode"] == "score",
        "context": data["context_length"] == data["prompt_tokens"] == 1024
        and data["context_buckets"] == [1024],
        "single_pass": len(sessions) == 1
        and sessions[0]["session"] == 0
        and sessions[0]["cached_tokens_at_end"] == 1024,
        "targets": data["scored_tokens"] == 1023,
        "config": data["config_hash"] == config(group).config_hash()
        and data["quantize_current_kv"],
        "tokens": data["assets"]["tokens_file"] == window
        and data["assets"]["tokens_sha256"] == token_hash,
        "memory": data["kv_store_bytes"] == 30277632,
        "unprofiled": not data.get("op_profiles") and not data.get("diagnostic_only"),
        "finite": math.isfinite(data["nll_sum"])
        and data["nll_sum"] >= 0
        and math.isfinite(data["ppl"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Scoring contract failed: {group}/{window}: {checks}")
    mean_nll = data["nll_sum"] / data["scored_tokens"]
    # Existing runner JSON prints nine significant digits.
    if not math.isclose(math.exp(mean_nll), data["ppl"], rel_tol=1e-7):
        raise ValueError("Runner PPL disagrees with NLL/token count")
    return {
        "group": group,
        "window": window,
        "nll_sum": data["nll_sum"],
        "scored_tokens": data["scored_tokens"],
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "kv_store_bytes": data["kv_store_bytes"],
        "config_hash": data["config_hash"],
        "tokens_sha256": token_hash,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    expected = {(g, w) for g in GROUPS for w in WINDOWS}
    if len(rows) != 16 or {(r["group"], r["window"]) for r in rows} != expected:
        raise ValueError("Require exactly four unique windows for each configuration")
    result = []
    for group in GROUPS:
        selected = [r for r in rows if r["group"] == group]
        tokens = sum(r["scored_tokens"] for r in selected)
        nll = math.fsum(r["nll_sum"] for r in selected)
        if tokens != 4092 or not math.isfinite(nll) or nll < 0:
            raise ValueError("Invalid aggregate NLL or target count")
        result.append(
            {
                "group": group,
                "scored_tokens": tokens,
                "nll_sum": nll,
                "mean_nll": nll / tokens,
                "ppl": math.exp(nll / tokens),
                "kv_store_bytes": 30277632,
            }
        )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root: Path) -> None:
    protocol = read(root / "protocol.json")
    if not read(root / "measurement_complete.json")["hash_verification_passed"]:
        raise ValueError("Final frozen-artifact verification is required")
    rows = [
        score_row(
            read(root / "reports" / f"{group}_w{i}.json"),
            group,
            window,
            protocol["token_hashes"][window],
        )
        for group in GROUPS
        for i, window in enumerate(WINDOWS)
    ]
    summary = aggregate(rows)
    indexed = {row["group"]: row for row in summary}
    contrasts = {}
    for layout in MASKS:
        structured = indexed[f"{layout}_structured"]
        lm = indexed[f"{layout}_lm"]
        contrasts[layout] = {
            "structured_minus_lm_mean_nll": structured["mean_nll"] - lm["mean_nll"],
            "structured_minus_lm_ppl": structured["ppl"] - lm["ppl"],
            "structured_ppl_increase_percent": 100
            * (structured["ppl"] / lm["ppl"] - 1),
        }
    save_new(root / "summary.json", {"groups": summary, "contrasts": contrasts})
    write_csv(root / "window_nll.csv", rows)
    write_csv(root / "summary.csv", summary)
    for row in summary:
        print(row, flush=True)


def run(root: Path, source: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    os.environ["ANDROID_SERIAL"] = SERIAL
    active = adb("pidof qnn-llm-runner")
    if active["returncode"] not in (0, 1) or active["stdout"].strip():
        raise RuntimeError("Device unavailable or another runner is active")
    assets = read(ASSETS / "assets.json")
    if assets["wikitext_windows"] != list(WINDOWS):
        raise ValueError("Expected the unchanged legacy WikiText four-window set")
    for name, expected in assets["sha256"].items():
        if digest(ASSETS / name) != expected:
            raise ValueError("Changed input asset: " + name)
    if any((ASSETS / w).stat().st_size != 1024 * 4 for w in WINDOWS):
        raise ValueError("Expected 1024 int32 token IDs per window")
    save_new(
        root / "protocol.json",
        {
            "created_utc": now(),
            "source": str(source),
            "source_freeze_sha256": digest(source / "measurement_freeze.json"),
            "source_protocol_sha256": digest(source / "protocol.json"),
            "model": "Qwen3-1.7B W4A16",
            "device_serial": SERIAL,
            "sdk": "2.48.0.260626",
            "groups": GROUPS,
            "masks": MASKS,
            "assets": str(ASSETS),
            "token_hashes": {w: assets["sha256"][w] for w in WINDOWS},
            "config_hashes": {g: config(g).config_hash() for g in GROUPS},
            "script_sha256": digest(Path(__file__)),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "protocol": "Existing HTP runner AR128 teacher-forced scoring; one fresh process/session per window, 1024 tokens / 1023 targets; 4 windows x 4 configs = 16 passes. No warmup or repeat, no logit dump or profiling, no generation-speed measurement.",
            "aggregation": "exp(sum(window_nll_sum) / sum(scored_tokens)), 4092 targets/config; not mean of window PPLs",
            "scope": "Legacy WikiText reference only, not a new independent heldout or generalization claim. Existing encoder numerical-gate limitations remain unchanged.",
        },
    )
    print("Verifying frozen host/device artifacts before scoring", flush=True)
    verify_freeze(source)
    save_new(
        root / "preflight.json",
        {"utc": now(), "hash_verification_passed": True, "device": snapshot(full=True)},
    )
    for group in GROUPS:
        native = read(source / group / "convert_report.json")["native_decoder"][
            "libraries"
        ]["hexagon-v81"]["sha256"]
        for i, window in enumerate(WINDOWS):
            label = f"{group}_w{i}"
            report = root / "reports" / f"{label}.json"
            tag = root.name + "_" + label
            remote = adb(
                f"test -e /data/local/tmp/qaihm_turboquant/llm/reports/{tag}.json && echo exists"
            )
            if remote["returncode"] not in (0, 1) or remote["stdout"].strip():
                raise FileExistsError(
                    "Device report exists or device unavailable: " + tag
                )
            start = now()
            execute(
                "run_device_llm.py",
                [
                    "run",
                    "--name",
                    source.name + "_" + group,
                    "--assets",
                    str(ASSETS),
                    "--mode",
                    "score",
                    "--tokens",
                    window,
                    "--sessions",
                    "1",
                    "--report",
                    str(report),
                    "--remote-report-tag",
                    tag,
                ],
                report.with_suffix(".stdout.log"),
            )
            data = read(report)
            row = score_row(data, group, window, assets["sha256"][window])
            if data["native_decoder"]["sha256"] != native:
                raise ValueError("Native runtime differs from frozen binary")
            save_new(
                report.with_suffix(".checked.json"),
                {
                    "start_utc": start,
                    "end_utc": now(),
                    "row": row,
                    "report_sha256": digest(report),
                },
            )
            print(
                f"{label}: NLL={row['mean_nll']:.9f} PPL={row['ppl']:.7f}", flush=True
            )
    print("Verifying frozen host/device artifacts after scoring", flush=True)
    verify_freeze(source)
    save_new(
        root / "measurement_complete.json",
        {
            "utc": now(),
            "passes": 16,
            "hash_verification_passed": True,
            "device": snapshot(full=True),
        },
    )
    analyze(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "analyze"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "run":
        run(args.root, args.source)
    else:
        analyze(args.root)


if __name__ == "__main__":
    main()
