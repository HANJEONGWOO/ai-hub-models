# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Measure quantized-current K4/V4 once; reuse historical int16 and raw-current runs.

Run push, functional, performance, quality, then summarize. Attempt logs are
exclusive-create: a failed attempt is retained, never silently overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from summarize_native_results import invariants, performance, read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["push", "functional", "performance", "quality", "summarize"]
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--previous-reports", type=Path, required=True)
    parser.add_argument("--name", default="current_kv_native_20260922")
    args = parser.parse_args()
    root = args.reports
    root.mkdir(parents=True, exist_ok=True)
    bundle = read(args.bundle / "convert_report.json")
    if (
        bundle["profile"] != "k4_v4_scaled"
        or not bundle.get("quantize_current_kv")
        or bundle["config"]["rotation"] != "dense_qr"
        or not bundle.get("native_decoder")
        or bundle["context_buckets"] != [128, 256, 512, 1024]
    ):
        raise ValueError(
            "Expected Dense+Native K4/V4, quantized current KV, C128/256/512/1024."
        )
    for part in range(1, 5):
        if not (args.bundle / f"part{part}_of_4.bin").is_file():
            raise FileNotFoundError(f"Missing compiled part {part}.")
    runner_hash = hashlib.sha256(args.runner.read_bytes()).hexdigest()
    previous_experiment = read(args.previous_reports / "experiment.json")
    if runner_hash != previous_experiment["runner_sha256"]:
        raise ValueError("Use the same runner binary as the historical measurement.")

    def execute(command: list[str], log: Path) -> None:
        print("START", log.stem, flush=True)
        with log.open("x") as output:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("run_device_llm.py")),
                    *command,
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print("DONE", log.stem, flush=True)

    def run(filename: str, options: list[str]) -> None:
        path = root / filename
        if path.exists() or path.with_suffix(".log").exists():
            raise FileExistsError(f"Refusing a repeated measurement: {path}")
        execute(
            [
                "run",
                "--name",
                args.name,
                "--assets",
                str(args.assets),
                "--report",
                str(path),
                "--remote-report-tag",
                args.name + "_" + path.stem,
                *options,
            ],
            path.with_suffix(".stdout.log"),
        )
        data = read(path)
        if data["config_hash"] != bundle["config_hash"] or not data.get(
            "quantize_current_kv"
        ):
            raise ValueError(
                f"Wrong current-token policy/configuration on device: {path}"
            )

    if args.stage == "push":
        audit = read(root / "boundary_current_native.json")
        if not audit["passed"] or not all(
            g.get("quantize_current_kv") for g in audit["graphs"].values()
        ):
            raise ValueError(
                "Compiled current-KV graph audit must pass before staging."
            )
        with (root / "experiment.json").open("x") as output:
            json.dump(
                {
                    "runner_sha256": runner_hash,
                    "bundle": str(args.bundle.resolve()),
                    "config_hash": bundle["config_hash"],
                    "quantize_current_kv": True,
                    "previous_reports": str(args.previous_reports.resolve()),
                    "performance_sessions_per_configuration_condition": 1,
                    "bins_sha256": {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in args.bundle.glob("part*_of_4.bin")
                    },
                },
                output,
                indent=2,
            )
        execute(
            [
                "push",
                "--bundle-dir",
                str(args.bundle),
                "--runner",
                str(args.runner),
                "--name",
                args.name,
            ],
            root / "push.stdout.log",
        )
    elif args.stage == "functional":
        for label, options in (
            ("reset", ["--n-gen", "8", "--sessions", "2"]),
            ("eos", ["--n-gen", "32", "--stop-on-eos"]),
            ("switches", ["--n-gen", "600"]),
        ):
            run(
                f"generation_current_native_{label}.json",
                ["--mode", "generate", *options],
            )
    elif args.stage == "performance":
        for condition, tokens, prompt in (
            ("short", "prompt_ids.bin", 35),
            ("long", "boundary_prompt_cl1024.bin", 897),
        ):
            filename = f"perf_current_native_{condition}_once.json"
            run(
                filename,
                [
                    "--mode",
                    "generate",
                    "--n-gen",
                    "128",
                    "--sessions",
                    "1",
                    "--tokens",
                    tokens,
                ],
            )
            print(json.dumps(performance(root / filename, prompt)), flush=True)
    elif args.stage == "quality":
        for window in range(4):
            run(
                f"score_current_native_w{window}.json",
                ["--mode", "score", "--tokens", f"wikitext_w{window}.bin"],
            )
    else:
        groups = {
            "baseline_int16": args.previous_reports,
            "dense_native": args.previous_reports,
            "current_native": root,
        }
        summary = {
            "performance_sessions_per_configuration_condition": 1,
            "historical_groups": ["baseline_int16", "dense_native"],
            "measurement_notes": "Historical controls are reused, not remeasured. Same device, runner and token files; different dates. Loading excluded from TTFT, profiling off; no temperature control, warmup exclusion or variance estimate. PPL is exp(total NLL / 4092), not mean window PPL. Baseline fixed C1024; both TurboQuant groups use C128/256/512/1024.",
            "short": {},
            "long": {},
            "quality": {},
            "experiment": read(root / "experiment.json"),
            "graph_audit": read(root / "boundary_current_native.json"),
        }
        device = None
        digests = {}
        native_package = None
        config_hashes = {}
        for group, directory in groups.items():
            for condition, prompt in (("short", 35), ("long", 897)):
                path = directory / f"perf_{group}_{condition}_once.json"
                data = read(path)
                if device is not None and data["device"] != device:
                    raise ValueError(f"Device mismatch: {path}")
                device = data["device"]
                digest = data["assets"]["tokens_sha256"]
                if condition in digests and digests[condition] != digest:
                    raise ValueError(f"Input mismatch: {path}")
                digests[condition] = digest
                if bool(data.get("quantize_current_kv")) != (group == "current_native"):
                    raise ValueError(f"Wrong current KV policy: {path}")
                if (
                    group != "baseline_int16"
                    and data["config_hash"] != bundle["config_hash"]
                ):
                    raise ValueError(f"Storage configuration mismatch: {path}")
                if (
                    group in config_hashes
                    and config_hashes[group] != data["config_hash"]
                ):
                    raise ValueError(f"Configuration changed within group: {path}")
                config_hashes[group] = data["config_hash"]
                if group != "baseline_int16":
                    if (
                        native_package is not None
                        and native_package != data["native_decoder"]
                    ):
                        raise ValueError(f"Native package mismatch: {path}")
                    native_package = data["native_decoder"]
                summary[condition][group] = {
                    **performance(path, prompt),
                    "reused_prior_measurement": group != "current_native",
                }
            windows = []
            for window in range(4):
                path = directory / f"score_{group}_w{window}.json"
                data = read(path)
                key = f"w{window}"
                digest = data["assets"]["tokens_sha256"]
                if (
                    data["mode"] != "score"
                    or data["scored_tokens"] != 1023
                    or data["device"] != device
                    or data["config_hash"] != config_hashes[group]
                    or bool(data.get("quantize_current_kv"))
                    != (group == "current_native")
                    or (key in digests and digests[key] != digest)
                ):
                    raise ValueError(f"Quality condition mismatch: {path}")
                digests[key] = digest
                windows.append(
                    {"nll_sum": data["nll_sum"], "report": str(path.resolve())}
                )
            nll = sum(w["nll_sum"] for w in windows)
            summary["quality"][group] = {
                "ppl": math.exp(nll / 4092),
                "nll_sum": nll,
                "scored_tokens": 4092,
                "windows": windows,
                "reused_prior_measurement": group != "current_native",
            }
        summary["invariants"] = invariants(root, "current_native")
        summary["input_sha256"] = digests
        summary["device"] = device
        summary["config_hashes"] = config_hashes
        summary["native_decoder"] = native_package
        for condition in ("short", "long"):
            old, new = (
                summary[condition][g] for g in ("dense_native", "current_native")
            )
            summary[condition + "_change_percent"] = {
                k: (new[k] / old[k] - 1) * 100
                for k in (
                    "ttft_ms",
                    "prefill_tok_per_s",
                    "decode_tok_per_s",
                    "host_kv_MiB",
                    "end_VmRSS_MiB",
                )
            }
        with (root / "comparison_current_kv.json").open("x") as output:
            json.dump(summary, output, indent=2)
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "short_change_percent",
                        "long_change_percent",
                        "quality",
                        "invariants",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
