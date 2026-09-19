# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Stage three existing bundles and run one performance session per condition.

Refuses to overwrite reports, including partially completed attempts. Functional
and WikiText quality runs are separate from the six performance sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from summarize_native_results import performance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["push", "functional", "performance", "quality", "all"]
    )
    parser.add_argument("--baseline-bundle", type=Path, required=True)
    parser.add_argument("--dense-bundle", type=Path, required=True)
    parser.add_argument("--qjl-bundle", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--device-prefix", default="qjl_compare_20260919")
    args = parser.parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("run_device_llm.py")
    groups = {
        "baseline_int16": args.baseline_bundle,
        "dense_native": args.dense_bundle,
        "qjl_native": args.qjl_bundle,
    }
    hashes = {}
    expected_profiles = {
        "baseline_int16": "baseline_int16_kv",
        "dense_native": "k4_v4_scaled",
        "qjl_native": "k3qjl_v4_scaled",
    }
    for group, bundle in groups.items():
        metadata = json.loads((bundle / "convert_report.json").read_text())
        if metadata["profile"] != expected_profiles[group]:
            raise ValueError(f"Wrong profile for {group}: {metadata['profile']}")
        if any(not (bundle / f"part{part}_of_4.bin").is_file() for part in range(1, 5)):
            raise FileNotFoundError(f"Incomplete four-part bundle: {bundle}")
        hashes[group] = metadata["config_hash"]

    def execute(command: list[str], log: Path) -> None:
        print("START", log.stem, flush=True)
        with log.open("x") as output:
            subprocess.run(
                [sys.executable, str(script), *command],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print("DONE", log.stem, flush=True)

    if args.stage in ("push", "all"):
        metadata = {
            "runner_sha256": hashlib.sha256(args.runner.read_bytes()).hexdigest(),
            "config_hashes": hashes,
            "groups": {k: str(v) for k, v in groups.items()},
        }
        with (args.reports / "experiment.json").open("x") as output:
            output.write(json.dumps(metadata, indent=2) + "\n")
        for group, bundle in groups.items():
            execute(
                [
                    "push",
                    "--bundle-dir",
                    str(bundle),
                    "--runner",
                    str(args.runner),
                    "--name",
                    args.device_prefix + "_" + group,
                ],
                args.reports / f"push_{group}.stdout.log",
            )

    def run(group: str, filename: str, extra: list[str]) -> None:
        report = args.reports / filename
        if report.exists() or report.with_suffix(".log").exists():
            raise FileExistsError(
                f"Refusing to repeat/overwrite an attempted run: {report}"
            )
        execute(
            [
                "run",
                "--name",
                args.device_prefix + "_" + group,
                "--assets",
                str(args.assets),
                "--report",
                str(report),
                "--remote-report-tag",
                args.device_prefix + "_" + report.stem,
                *extra,
            ],
            report.with_suffix(".stdout.log"),
        )
        data = json.loads(report.read_text())
        if data["config_hash"] != hashes[group]:
            raise ValueError(f"Wrong on-device configuration: {report}")

    if args.stage in ("functional", "all"):
        for label, options in (
            ("reset", ["--n-gen", "8", "--sessions", "2"]),
            ("eos", ["--n-gen", "32", "--stop-on-eos"]),
            ("switches", ["--n-gen", "600"]),
        ):
            run(
                "qjl_native",
                f"generation_qjl_native_{label}.json",
                ["--mode", "generate", *options],
            )
    if args.stage in ("performance", "all"):
        for condition, prompt, token_file in (
            ("short", 35, "prompt_ids.bin"),
            ("long", 897, "boundary_prompt_cl1024.bin"),
        ):
            for group in groups:
                filename = f"perf_{group}_{condition}_once.json"
                run(
                    group,
                    filename,
                    [
                        "--mode",
                        "generate",
                        "--n-gen",
                        "128",
                        "--sessions",
                        "1",
                        "--tokens",
                        token_file,
                    ],
                )
                print(
                    json.dumps(performance(args.reports / filename, prompt)), flush=True
                )
    if args.stage in ("quality", "all"):
        for group in groups:
            for window in range(4):
                run(
                    group,
                    f"score_{group}_w{window}.json",
                    ["--mode", "score", "--tokens", f"wikitext_w{window}.bin"],
                )


if __name__ == "__main__":
    main()
