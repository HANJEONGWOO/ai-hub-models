#!/usr/bin/env python3
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Profile immutable FP16/TurboQuant bundles once each, without a model rebuild.

Reuse an existing benchmark experiment and device bundles. Only a uniquely named
diagnostic runner is uploaded. Never overwrite the benchmark runner or reports.
The resulting latency/throughput is diagnostic, not a benchmark replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

DEVICE_ROOT = "/data/local/tmp/qaihm_turboquant/llm"
SCRIPTS = Path(__file__).resolve().parent


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_new(path: Path, data: dict) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-reports", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adb", type=Path, default=Path("/mnt/c/adb/adb.exe"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    experiment = read(args.reference_reports / "experiment.json")
    assets = read(args.assets / "assets.json")
    if assets["context_length"] != 1024:
        raise ValueError("This diagnostic protocol requires CL1024")
    if (args.assets / "boundary_prompt_cl1024.bin").stat().st_size != 897 * 4:
        raise ValueError("Expected the frozen 897-token int32 long prompt")
    if assets["sha256"] != experiment["assets_sha256"]:
        raise ValueError("Asset manifest differs from the frozen reference")
    for name, expected in assets["sha256"].items():
        if digest(args.assets / name) != expected:
            raise ValueError(f"Asset changed: {name}")

    def adb(*command: str) -> str:
        return subprocess.run(
            [str(args.adb), *command], check=True, capture_output=True, text=True
        ).stdout.strip()

    def temperature() -> str:
        result = re.search(
            r"^\s*temperature:\s*(\d+)", adb("shell", "dumpsys battery"), re.MULTILINE
        )
        return result.group(1) if result else "unavailable"

    groups = {}
    for group in ("fp16", "turboquant"):
        source = experiment["groups"][group]
        old_report = read(args.reference_reports / f"{group}_long_once.json")
        name = old_report["bundle_name"]
        bundle = Path(source["bundle"])
        if digest(bundle / "convert_report.json") != source["conversion_sha256"]:
            raise ValueError(f"Changed conversion metadata: {group}")
        for filename, expected in source["bins_sha256"].items():
            if digest(bundle / filename) != expected:
                raise ValueError(f"Changed local context binary: {group}/{filename}")
            remote = f"{DEVICE_ROOT}/bundles/{name}/{filename}"
            if adb("shell", "sha256sum", remote).split()[0] != expected:
                raise ValueError(f"Changed device context binary: {remote}")
        groups[group] = {**source, "device_bundle": name}

    runner_hash = digest(args.runner)
    runner_name = "qnn-llm-runner-stages-" + runner_hash[:16]
    identity = {
        "diagnostic_only": True,
        "reference_experiment": str(args.reference_reports / "experiment.json"),
        "runner_sha256": runner_hash,
        "runner_name": runner_name,
        "groups": groups,
        "assets_sha256": assets["sha256"],
        "context_buckets": [1024],
        "prompt_tokens": 897,
        "generated_tokens": 128,
        "profile_prefill": "all_chunks",
        "profile_decode_steps": [0, 63, 126],
        "sessions_per_group": 1,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device_fingerprint": adb("shell", "getprop", "ro.build.fingerprint"),
        "device_soc": adb("shell", "getprop", "ro.soc.model"),
        "notes": "Detailed profiling enabled throughout; do not use tok/s as baseline results.",
    }
    identity_path = args.out / "experiment.json"
    if identity_path.exists():
        if read(identity_path) != identity or list(args.out.glob("*.attempt.json")):
            raise ValueError(
                "Existing experiment differs or inference was already attempted"
            )
    else:
        save_new(identity_path, identity)
    runner_win = subprocess.check_output(
        ["wslpath", "-w", str(args.runner.resolve())], text=True
    ).strip()
    remote_runner = f"{DEVICE_ROOT}/bin/{runner_name}"
    adb("push", runner_win, remote_runner)
    adb("shell", "chmod", "+x", remote_runner)
    if adb("shell", "sha256sum", remote_runner).split()[0] != runner_hash:
        raise ValueError("Diagnostic runner upload hash mismatch")
    for group, source in groups.items():
        report = args.out / f"{group}_profile_once.json"
        attempt = {
            "group": group,
            "temperature_before_tenths_c": temperature(),
            "status": "started",
        }
        save_new(report.with_suffix(".attempt.json"), attempt)
        if report.exists() or report.with_suffix(".log").exists():
            raise FileExistsError(report)
        print(f"START {group}: all prefill chunks and decode 0,63,126", flush=True)
        with report.with_suffix(".stdout.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "run_device_llm.py"),
                    "--adb",
                    str(args.adb),
                    "run",
                    "--name",
                    source["device_bundle"],
                    "--assets",
                    str(args.assets),
                    "--mode",
                    "generate",
                    "--tokens",
                    "boundary_prompt_cl1024.bin",
                    "--n-gen",
                    "128",
                    "--context-buckets",
                    "1024",
                    "--runner-name",
                    runner_name,
                    "--profile-prefill-all",
                    "--profile-decode-steps",
                    "0",
                    "63",
                    "126",
                    "--report",
                    str(report),
                    "--remote-report-tag",
                    args.out.parent.name + "_" + group,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        data = read(report)
        if data["context_buckets"] != [1024] or len(data["op_profiles"]) != 11:
            raise ValueError("Missing profile steps or wrong graph context")
        if len(data["generated"]) != 128:
            raise ValueError("Diagnostic generation did not complete")
        data["diagnostic_only"] = True
        data["profiling_protocol"] = identity["notes"]
        data["temperature_before_tenths_c"] = attempt["temperature_before_tenths_c"]
        data["temperature_after_tenths_c"] = temperature()
        # This is the new diagnostic report, never a historical artifact.
        report.write_text(json.dumps(data, indent=2) + "\n")
        save_new(
            report.with_suffix(".complete.json"),
            {
                "status": "complete",
                "report_sha256": digest(report),
                "temperature_after_tenths_c": data["temperature_after_tenths_c"],
            },
        )
        print(f"DONE {group}: {len(data['op_profiles'])} profiled steps", flush=True)


if __name__ == "__main__":
    main()
