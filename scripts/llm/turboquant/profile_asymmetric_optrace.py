#!/usr/bin/env python3
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Freeze/build/measure current K4/V4, K5/V3 and K6/V2 once with HTP optrace.

Reuses immutable quantized DLCs and the validated optimized runtime package.
Never regenerates ONNX/calibration or runs a new uninstrumented benchmark.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from profile_stages_once import SCRIPTS, digest, read, save_new

GROUPS = ("k4_v4", "k5_v3", "k6_v2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("freeze", "build", "measure", "render", "summarize")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--optimized", type=Path)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "freeze":
        if not all((args.original, args.optimized, args.assets)):
            parser.error("freeze requires --original, --optimized and --assets")
        root.mkdir(parents=True, exist_ok=False)
        (root / "reference").mkdir()
        groups = {}
        for group in GROUPS:
            bundle = args.original.resolve() / group
            report = (
                (args.original if group == "k4_v4" else args.optimized).resolve()
                / "reports"
                / f"{group}_perf_long.json"
            )
            groups[group] = {
                "bundle": str(bundle),
                "conversion_sha256": digest(bundle / "convert_report.json"),
                "reference_report": str(report),
                "reference_report_sha256": digest(report),
                "bins_sha256": {
                    p.name: digest(p) for p in sorted(bundle.glob("*.bin"))
                },
            }
            if group != "k4_v4":
                package = args.optimized.resolve() / "native"
                groups[group].update(
                    native_runtime_package=str(package),
                    native_runtime_manifest_sha256=digest(package / "manifest.json"),
                )
        save_new(
            root / "reference/experiment.json",
            {
                "groups": groups,
                "assets_sha256": read(args.assets / "assets.json")["sha256"],
                "assets": str(args.assets.resolve()),
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "protocol": {
                    "model": "Qwen3-1.7B W4A16",
                    "context_length": 1024,
                    "prompt_tokens": 897,
                    "generated_tokens": 128,
                    "sessions_per_group": 1,
                    "order": list(GROUPS),
                    "profile_prefill": "all eight AR128 chunks (first chunk has one valid token, then seven full chunks)",
                    "profile_decode_steps": [0, 63, 126],
                    "diagnostic_only": True,
                    "limitations": [
                        "No repeated measurements or new PPL run",
                        "Greedy generated tokens can differ across configurations",
                        "Busy cycles are parallel work, not stage wall latency",
                        "Existing strict encoder scale validation failures remain open",
                    ],
                },
            },
        )
        return
    experiment = read(root / "reference/experiment.json")

    def run(script: str, *parameters: str, python: str = sys.executable) -> None:
        subprocess.run([python, str(SCRIPTS / script), *parameters], check=True)

    if args.action == "build":
        subprocess.run(
            [
                "bash",
                str(SCRIPTS / "qnn_runner/build_android.sh"),
                str(root / "runner"),
            ],
            check=True,
        )

        def build(group: str) -> None:
            run(
                "build_optrace_contexts.py",
                "--source",
                experiment["groups"][group]["bundle"],
                "--out",
                str(root / group),
                "--parts",
                "1",
                "2",
                "3",
                "4",
            )

        # Two context builders fit the host memory budget; device calls stay serial.
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(build, GROUPS))
    elif args.action == "measure":
        run(
            "profile_optrace_once.py",
            "measure",
            "--root",
            str(root),
            "--reference-reports",
            str(root / "reference"),
            "--assets",
            experiment["assets"],
            "--groups",
            *GROUPS,
        )
    elif args.action == "render":
        for group in GROUPS:
            run(
                "render_optrace.py",
                "--logs",
                str(root / "traces" / group),
                "--contexts",
                str(root / group),
                "--out",
                str(root / "rendered" / group),
                "--jobs",
                "4",
                python="/home/hjw/qnn-venv/bin/python",
            )
    else:
        run("summarize_optrace.py", "--root", str(root), "--groups", *GROUPS)


if __name__ == "__main__":
    main()
