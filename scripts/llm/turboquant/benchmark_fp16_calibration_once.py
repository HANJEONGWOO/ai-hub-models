# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Measure calibrated FP16 once; reuse identity-checked historical FP16 results."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from model_identity import sha256_file
from summarize_native_results import performance, read

SCRIPTS = Path(__file__).resolve().parent


def write_once(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("push", "functional", "performance", "quality", "summarize")
    )
    for name in ("bundle", "baseline-reports", "runner", "assets", "reports"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--name", default="fp16_calibration_20260929")
    args = parser.parse_args()
    root = args.reports.resolve()
    root.mkdir(parents=True, exist_ok=True)
    bundle = args.bundle.resolve()
    conversion = read(bundle / "convert_report.json")
    historical = read(args.baseline_reports / "experiment.json")
    assets = read(args.assets / "assets.json")
    if conversion["profile"] != "baseline_fp16_kv_fp16_attn_calibrated" or conversion[
        "context_buckets"
    ] != [1024]:
        raise ValueError("Expected calibrated FP16, fixed CL1024")
    if assets["sha256"] != historical["assets_sha256"] or historical[
        "graph_contexts"
    ] != [1024]:
        raise ValueError("Historical/current input or graph context mismatch")
    if sha256_file(args.runner) != historical["runner_sha256"]:
        raise ValueError("Runner differs from historical measurement")
    for name, digest in assets["sha256"].items():
        if sha256_file(args.assets / name) != digest:
            raise ValueError("Input asset modified")
    old = historical["groups"]["fp16"]
    old_bundle = Path(old["bundle"])
    if sha256_file(old_bundle / "convert_report.json") != old["conversion_sha256"]:
        raise ValueError("Historical conversion metadata changed")
    for name, digest in old["bins_sha256"].items():
        if sha256_file(old_bundle / name) != digest:
            raise ValueError("Historical binary changed")
    baseline = read(args.baseline_reports / "summary.json")
    for label in ("short_once", "long_once", *(f"score_w{i}" for i in range(4))):
        data = read(args.baseline_reports / f"fp16_{label}.json")
        if data["context_buckets"] != [1024] or data["device"] != baseline["device"]:
            raise ValueError("Historical report configuration/device mismatch")
    audit = read(root / "fp16_graph_audit.json")
    if (
        not audit["passed"]
        or not audit.get("shared_interfaces")
        or audit["shared_interfaces"]["violations"]
        or Path(audit["bundle"]).resolve() != bundle
        or audit["config_hash"] != conversion["config_hash"]
    ):
        raise ValueError("Calibrated bundle audit must pass first")
    for prefix in ("prompt_ar128_", "token_ar1_"):
        layers = [
            l
            for n, g in audit["graphs"].items()
            if n.startswith(prefix)
            for l in g["layers"]
        ]
        if sorted(layers) != list(range(28)):
            raise ValueError("Missing/duplicate audited layers")
    identity = {
        "bundle": str(bundle),
        "conversion_sha256": sha256_file(bundle / "convert_report.json"),
        "bins_sha256": {
            f"part{i}_of_4.bin": sha256_file(bundle / f"part{i}_of_4.bin")
            for i in range(1, 5)
        },
        "runner_sha256": sha256_file(args.runner),
        "assets_sha256": assets["sha256"],
        "audit_sha256": sha256_file(root / "fp16_graph_audit.json"),
        "baseline_summary_sha256": sha256_file(args.baseline_reports / "summary.json"),
        "baseline_reports": str(args.baseline_reports.resolve()),
        "baseline_report_sha256": {
            p.name: sha256_file(p) for p in args.baseline_reports.glob("fp16_*.json")
        },
        "runs_per_condition": 1,
        "graph_contexts": [1024],
    }
    frozen = root / "experiment.json"
    if frozen.exists():
        if read(frozen) != identity:
            raise ValueError("Experiment inputs changed")
    elif args.stage == "push":
        write_once(frozen, identity)
    else:
        raise ValueError("Push/freeze experiment first")

    def execute(options: list[str], log: Path) -> None:
        with log.open("x") as stream:
            subprocess.run(
                [sys.executable, str(SCRIPTS / "run_device_llm.py"), *options],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def run(label: str, options: list[str]) -> dict:
        path = root / f"calibrated_{label}.json"
        if path.exists():
            raise FileExistsError("Refusing repeated measurement")
        print("START", label, flush=True)
        execute(
            [
                "run",
                "--name",
                args.name,
                "--assets",
                str(args.assets),
                "--context-buckets",
                "1024",
                "--report",
                str(path),
                "--remote-report-tag",
                args.name + "_" + label,
                *options,
            ],
            path.with_suffix(".stdout.log"),
        )
        data = read(path)
        if (
            data.get("activation_calibration_sha256")
            != conversion["activation_calibration_sha256"]
        ):
            raise ValueError("Device has a different calibration artifact")
        if data["config_hash"] != conversion["config_hash"] or data[
            "context_buckets"
        ] != [1024]:
            raise ValueError("Wrong device graph configuration")
        if data["device"] != baseline["device"]:
            raise ValueError("Device differs from historical baseline")
        if (
            data["assets"]["tokens_sha256"]
            != assets["sha256"][data["assets"]["tokens_file"]]
        ):
            raise ValueError("Wrong benchmark tokens")
        if (
            data["kv_store_bytes"] != 112 * 2**20
            or len(data["kv_streams"]) != 56
            or any(s["dtype"] != "float16" for s in data["kv_streams"])
        ):
            raise ValueError("Wrong device KV storage")
        print("DONE", label, flush=True)
        return data

    if args.stage == "push":
        execute(
            [
                "push",
                "--bundle-dir",
                str(bundle),
                "--name",
                args.name,
                "--runner",
                str(args.runner),
            ],
            root / "push.stdout.log",
        )
    elif args.stage == "functional":
        reset = run("reset", ["--mode", "generate", "--n-gen", "8", "--sessions", "2"])
        sessions = reset["sessions"]
        passed = (
            len(sessions) == 2
            and sessions[0]["generated"] == sessions[1]["generated"]
            and all(
                len(s["generated"]) == 8 and s["cached_tokens_at_end"] == 42
                for s in sessions
            )
        )
        eos = run("eos", ["--mode", "generate", "--n-gen", "128", "--stop-on-eos"])
        write_once(
            root / "functional.json",
            {
                "reset": passed,
                "eos": eos["stop_reason"] == "eos",
                "stop_reason": eos["stop_reason"],
            },
        )
        if not passed:
            raise ValueError("Reset/cache functional check failed")
    elif args.stage == "performance":
        if not read(root / "functional.json")["reset"]:
            raise ValueError("Reset gate failed")
        for label, tokens, prompt in (
            ("short", "prompt_ids.bin", 35),
            ("long", "boundary_prompt_cl1024.bin", 897),
        ):
            run(
                label + "_once",
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
                json.dumps(performance(root / f"calibrated_{label}_once.json", prompt)),
                flush=True,
            )
    elif args.stage == "quality":
        for window in range(4):
            data = run(
                f"score_w{window}",
                ["--mode", "score", "--tokens", f"wikitext_w{window}.bin"],
            )
            if data["scored_tokens"] != 1023 or not math.isfinite(data["nll_sum"]):
                raise ValueError("Invalid PPL result")
    else:
        checks = read(root / "functional.json")
        windows = [read(root / f"calibrated_score_w{i}.json") for i in range(4)]
        nll = sum(w["nll_sum"] for w in windows)
        tokens = sum(w["scored_tokens"] for w in windows)
        current = {
            "diagnostic_only": not checks["eos"],
            **{
                c: performance(root / f"calibrated_{c}_once.json", n)
                for c, n in (("short", 35), ("long", 897))
            },
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
        summary = {
            "experiment": str(frozen),
            "device": baseline["device"],
            "functional": checks,
            "groups": {
                "original_fp16": baseline["groups"]["fp16"],
                "calibrated_fp16": current,
            },
            "original_reused": True,
            "runs_per_condition": 1,
            "notes": "Historical FP16 reused with identical runner/assets/device and fixed C1024; timestamps differ. Single runs, no variance estimate. PPL is four held-out windows only. FP16 inputs do not specify hardware accumulation precision.",
        }
        write_once(root / "summary.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
