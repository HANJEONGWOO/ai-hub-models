# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""One-run CL1024 comparison of legacy int16, FP16 attention and current TQ.

Use explicit existing bundles; never rebuild or modify historical artifacts.
Attempts are exclusive-create, including failed runs. All groups use the same
runner, assets and fixed C1024 graph selection. Functional runs are not timings.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from model_identity import sha256_file
from summarize_native_results import performance, read

PROFILES = {
    "int16": "baseline_int16_kv",
    "fp16": "baseline_fp16_kv_fp16_attn",
    "turboquant": "k4_v4_scaled",
}
SCRIPTS = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["push", "functional", "performance", "quality", "summarize"]
    )
    for group in PROFILES:
        parser.add_argument(f"--{group}-bundle", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--name", default="fp16_attention_20260929")
    args = parser.parse_args()
    root = args.reports
    root.mkdir(parents=True, exist_ok=True)
    assets = read(args.assets / "assets.json")
    if (
        assets["context_length"] != 1024
        or assets["prompt_tokens"] != 35
        or len(assets["wikitext_windows"]) != 4
    ):
        raise ValueError("Expected unchanged 1.7B CL1024 assets")
    for name, digest in assets["sha256"].items():
        if sha256_file(args.assets / name) != digest:
            raise ValueError(f"Asset identity changed: {name}")
    bundles = {g: getattr(args, g + "_bundle").resolve() for g in PROFILES}
    metadata = {g: read(p / "convert_report.json") for g, p in bundles.items()}
    for group, data in metadata.items():
        if (
            data["profile"] != PROFILES[group]
            or data["context_length"] != 1024
            or set(data["parts"]) != {f"part{i}_of_4" for i in range(1, 5)}
        ):
            raise ValueError(f"Unexpected bundle: {group}")
    tq = metadata["turboquant"]
    if (
        not tq.get("quantize_current_kv")
        or not tq.get("native_decoder")
        or tq["config"]["rotation"] != "dense_qr"
        or tq["config"].get("qjl")
    ):
        raise ValueError("Expected Dense+Native K4/V4, QJL-off, quantized current KV")
    identity = {
        "runner_sha256": sha256_file(args.runner),
        "fp16_graph_audit_sha256": sha256_file(root / "fp16_graph_audit.json"),
        "assets_sha256": assets["sha256"],
        "graph_contexts": [1024],
        "runs_per_group_condition": 1,
        "groups": {
            g: {
                "bundle": str(p),
                "conversion_sha256": sha256_file(p / "convert_report.json"),
                "bins_sha256": {
                    f"part{i}_of_4.bin": sha256_file(p / f"part{i}_of_4.bin")
                    for i in range(1, 5)
                },
            }
            for g, p in bundles.items()
        },
    }
    experiment = root / "experiment.json"
    if experiment.exists():
        if read(experiment) != identity:
            raise ValueError("Frozen experiment inputs changed")
    elif args.stage == "push":
        with experiment.open("x") as stream:
            json.dump(identity, stream, indent=2)
    else:
        raise ValueError("Push/freeze experiment before measuring")

    def execute(options: list[str], log: Path) -> None:
        print("START", log.stem, flush=True)
        with log.open("x") as stream:
            subprocess.run(
                [sys.executable, str(SCRIPTS / "run_device_llm.py"), *options],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print("DONE", log.stem, flush=True)

    def run(group: str, label: str, options: list[str]) -> dict:
        path = root / f"{group}_{label}.json"
        if path.exists() or path.with_suffix(".log").exists():
            raise FileExistsError(f"Refusing repeated measurement: {path}")
        execute(
            [
                "run",
                "--name",
                args.name + "_" + group,
                "--assets",
                str(args.assets),
                "--context-buckets",
                "1024",
                "--report",
                str(path),
                "--remote-report-tag",
                args.name + "_" + path.stem,
                *options,
            ],
            path.with_suffix(".stdout.log"),
        )
        data = read(path)
        if data["config_hash"] != metadata[group]["config_hash"] or data[
            "context_buckets"
        ] != [1024]:
            raise ValueError("Wrong runtime configuration")
        if (
            data["assets"]["tokens_sha256"]
            != assets["sha256"][data["assets"]["tokens_file"]]
        ):
            raise ValueError("Wrong measurement input")
        expected_bytes = int((28.875 if group == "turboquant" else 112) * 2**20)
        if data["kv_store_bytes"] != expected_bytes:
            raise ValueError("Unexpected 1.7B CL1024 host cache allocation")
        if group == "fp16" and (
            len(data["kv_streams"]) != 56
            or any(s["dtype"] != "float16" for s in data["kv_streams"])
        ):
            raise ValueError("Device cache is not 56 FP16 K/V streams")
        return data

    if args.stage == "push":
        audit = read(root / "fp16_graph_audit.json")
        if (
            not audit["passed"]
            or len(audit["graphs"]) != 6
            or audit["config_hash"] != metadata["fp16"]["config_hash"]
            or Path(audit["bundle"]).resolve() != bundles["fp16"]
        ):
            raise ValueError("Complete FP16 compiled graph audit must pass first")
        for ar in (1, 128):
            layers = [
                layer
                for name, graph in audit["graphs"].items()
                if name.startswith(f"{'token' if ar == 1 else 'prompt'}_ar{ar}_cl1024_")
                for layer in graph["layers"]
            ]
            if sorted(layers) != list(range(28)):
                raise ValueError("FP16 audit is missing/duplicating model layers")
        for group, bundle in bundles.items():
            # Push writes runtime metadata; keep historical bundles immutable.
            staging = root / "staging" / group
            staging.mkdir(parents=True, exist_ok=True)
            for i in range(1, 5):
                target = staging / f"part{i}_of_4.bin"
                if not target.exists():
                    target.symlink_to(bundle / target.name)
            conversion = staging / "convert_report.json"
            if not conversion.exists():
                conversion.symlink_to(bundle / conversion.name)
            execute(
                [
                    "push",
                    "--bundle-dir",
                    str(staging),
                    "--name",
                    args.name + "_" + group,
                    "--runner",
                    str(args.runner),
                ],
                root / f"{group}_push.stdout.log",
            )
    elif args.stage == "functional":
        checks = {}
        for group in PROFILES:
            reset = run(
                group,
                "reset",
                ["--mode", "generate", "--n-gen", "8", "--sessions", "2"],
            )
            sessions = reset["sessions"]
            passed = (
                len(sessions) == 2
                and sessions[0]["generated"] == sessions[1]["generated"]
                and all(
                    len(s["generated"]) == 8 and s["cached_tokens_at_end"] == 42
                    for s in sessions
                )
            )
            if not passed:
                raise ValueError(f"Reset/cache functional check failed: {group}")
            eos = run(
                group, "eos", ["--mode", "generate", "--n-gen", "128", "--stop-on-eos"]
            )
            checks[group] = {
                "reset": passed,
                "eos": eos["stop_reason"] == "eos",
                "stop_reason": eos["stop_reason"],
            }
        (root / "functional.json").write_text(json.dumps(checks, indent=2) + "\n")
    elif args.stage == "performance":
        checks = read(root / "functional.json")
        if not all(c["reset"] for c in checks.values()):
            raise ValueError("Reset check must pass")
        for condition, tokens, prompt in (
            ("short", "prompt_ids.bin", 35),
            ("long", "boundary_prompt_cl1024.bin", 897),
        ):
            for group in PROFILES:
                run(
                    group,
                    f"{condition}_once",
                    [
                        "--mode",
                        "generate",
                        "--tokens",
                        tokens,
                        "--n-gen",
                        "128",
                        "--sessions",
                        "1",
                    ],
                )
                print(
                    json.dumps(
                        performance(root / f"{group}_{condition}_once.json", prompt)
                    ),
                    flush=True,
                )
    elif args.stage == "quality":
        for window in range(4):
            for group in PROFILES:
                data = run(
                    group,
                    f"score_w{window}",
                    ["--mode", "score", "--tokens", f"wikitext_w{window}.bin"],
                )
                if data["scored_tokens"] != 1023 or not math.isfinite(data["nll_sum"]):
                    raise ValueError("Invalid PPL scoring result")
    else:
        summary = {
            "experiment": str(experiment.resolve()),
            "runs_per_group_condition": 1,
            "functional": read(root / "functional.json"),
            "groups": {},
        }
        devices = set()
        for group in PROFILES:
            windows = [read(root / f"{group}_score_w{i}.json") for i in range(4)]
            tokens = sum(w["scored_tokens"] for w in windows)
            nll = sum(w["nll_sum"] for w in windows)
            summary["groups"][group] = {
                "diagnostic_only": not summary["functional"][group]["eos"],
                "short": performance(root / f"{group}_short_once.json", 35),
                "long": performance(root / f"{group}_long_once.json", 897),
                "quality": {
                    "ppl": math.exp(nll / tokens),
                    "nll_sum": nll,
                    "scored_tokens": tokens,
                    "windows": [
                        {k: w[k] for k in ("ppl", "nll_sum", "scored_tokens")}
                        for w in windows
                    ],
                },
            }
            for path in [root / f"{group}_{c}_once.json" for c in ("short", "long")] + [
                root / f"{group}_score_w{i}.json" for i in range(4)
            ]:
                devices.add(json.dumps(read(path)["device"], sort_keys=True))
        if len(devices) != 1:
            raise ValueError("Device identity changed during comparison")
        summary["device"] = json.loads(devices.pop())
        summary["notes"] = (
            "Single runs; no variance estimate. TTFT excludes model loading. Fixed C1024. FP16 is untiled/unrotated; TQ retains tiled rotated attention and Native decoder. W4A16 and calibrated non-KV boundaries retained; not an all-FP16 model. Failed EOS groups are diagnostic."
        )
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
