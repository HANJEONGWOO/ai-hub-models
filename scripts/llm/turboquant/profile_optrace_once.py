#!/usr/bin/env python3
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Stage separate diagnostic contexts and collect one CL1024 optrace session.

The pilot is a synthetic single-graph serialization check, not a measurement.
The measure action uses the frozen 897+128-token protocol, once per group.
Never overwrite production contexts, runners, or earlier measurements.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from profile_stages_once import DEVICE_ROOT, SCRIPTS, digest, read, save_new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pilot", "measure"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-reports", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--pilot-label", default="pilot")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["fp16", "turboquant"],
    )
    parser.add_argument("--adb", type=Path, default=Path("/mnt/c/adb/adb.exe"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.root.name):
        raise ValueError("Root basename must be safe for a device directory")
    reference = read(args.reference_reports / "experiment.json")
    for group in args.groups:
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]+", group)
            or group not in reference["groups"]
        ):
            raise ValueError("Unknown or unsafe group " + group)

    def adb(*command: str) -> str:
        return subprocess.check_output([str(args.adb), *command], text=True).strip()

    def win(path: Path) -> str:
        return subprocess.check_output(
            ["wslpath", "-w", str(path.resolve())], text=True
        ).strip()

    def push(path: Path, remote: str) -> str:
        adb("push", win(path), remote)
        sha = digest(path)
        if adb("shell", "sha256sum", remote).split()[0] != sha:
            raise ValueError("Upload hash mismatch: " + remote)
        return sha

    def native(source: dict, remote: str) -> dict:
        package = source.get("native_decoder")
        if not package:
            return {}
        lib = package["libraries"]["hexagon-v81"]
        local = Path(lib["path"])
        if digest(local) != lib["sha256"]:
            raise ValueError("Native package changed")
        push(local, remote + "/" + local.name)
        return {
            "package": package["package"],
            "interface": package["interface"],
            "library": local.name,
            "sha256": lib["sha256"],
        }

    def source_for(group: str) -> tuple[Path, dict]:
        record = reference["groups"][group]
        bundle = Path(record["bundle"])
        if digest(bundle / "convert_report.json") != record["conversion_sha256"]:
            raise ValueError("Changed source conversion")
        source = read(bundle / "convert_report.json")
        if override := record.get("native_runtime_package"):
            from run_device_llm import native_runtime_package

            manifest = Path(override) / "manifest.json"
            if digest(manifest) != record["native_runtime_manifest_sha256"]:
                raise ValueError("Runtime package manifest changed")
            package, provenance = native_runtime_package(
                source.get("native_decoder"),
                Path(override),
                Path(source["native_decoder"]["qairt_sdk"]),
            )
            source = {**source, **provenance, "native_decoder": package}
        return bundle, source

    def checked_context(group: str, part: int) -> tuple[Path, dict]:
        path = args.root / group / f"part{part}_of_4"
        meta = read(path / "complete.json")
        binary = path / f"part{part}_of_4.bin"
        if digest(binary) != meta["context_sha256"]:
            raise ValueError("Changed optrace context")
        for filename, expected in meta["identity"]["source_dlc_sha256"].items():
            if digest(Path(filename)) != expected:
                raise ValueError("Changed original DLC")
        return binary, meta

    if args.action == "pilot":
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.pilot_label):
            raise ValueError("Unsafe pilot label")
        output = args.root / args.pilot_label
        output.mkdir(exist_ok=False)
        _, source = source_for("turboquant")
        binary, meta = checked_context("turboquant", 2)
        remote = f"{DEVICE_ROOT}/diagnostics/{args.root.name}_{args.pilot_label}"
        adb("shell", f"test ! -e {remote} && mkdir -p {remote}/trace")
        push(binary, remote + "/part2_of_4.bin")
        package = native(source, remote)
        runner = args.root / "runner/qnn-optrace-smoke"
        runner_sha = push(runner, remote + "/smoke")
        adb("shell", "chmod", "+x", remote + "/smoke")
        graph = "token_ar1_cl1024_2_of_4"
        save_new(
            output / "attempt.json",
            {
                "synthetic_inputs": True,
                "not_a_benchmark": True,
                "graph": graph,
                "context": meta,
                "runner_sha256": runner_sha,
            },
        )
        command = f"cd {DEVICE_ROOT}/bin && export LD_LIBRARY_PATH={DEVICE_ROOT}/bin:/vendor/lib64 && export ADSP_LIBRARY_PATH='{remote}:{DEVICE_ROOT}/bin:/vendor/dsp/cdsp:/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp' && {remote}/smoke libQnnHtp.so libQnnSystem.so {remote}/part2_of_4.bin {graph} {remote}/trace {remote}/{package['library']} {package['interface']}"
        with (output / "smoke.log").open("x") as stream:
            subprocess.run(
                [str(args.adb), "shell", command],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        adb("pull", remote + "/trace", win(output / "trace"))
        save_new(
            output / "complete.json",
            {"logs": {p.name: digest(p) for p in (output / "trace").glob("*.log")}},
        )
        print("PILOT complete", flush=True)
        return

    assets = read(args.assets / "assets.json")
    if (
        assets["sha256"] != reference["assets_sha256"]
        or assets["context_length"] != 1024
    ):
        raise ValueError("Frozen CL1024 assets differ")
    for name, expected in assets["sha256"].items():
        if digest(args.assets / name) != expected:
            raise ValueError("Changed asset " + name)
    if (args.assets / "boundary_prompt_cl1024.bin").stat().st_size != 897 * 4:
        raise ValueError("Expected 897 prompt tokens")
    output = args.root / "reports"
    output.mkdir(exist_ok=True)
    runner = args.root / "runner/qnn-llm-runner"
    runner_name = "qnn-llm-runner-optrace-" + digest(runner)[:16]
    push(runner, f"{DEVICE_ROOT}/bin/{runner_name}")
    adb("shell", "chmod", "+x", f"{DEVICE_ROOT}/bin/{runner_name}")
    for group in args.groups:
        report = output / f"{group}_profile_once.json"
        if report.with_suffix(".complete.json").exists():
            if (
                digest(report)
                != read(report.with_suffix(".complete.json"))["report_sha256"]
            ):
                raise ValueError("Completed report changed")
            print("REUSE measured", group, flush=True)
            continue
        if report.with_suffix(".attempt.json").exists():
            raise ValueError(
                "An inference was already attempted; inspect it, do not repeat"
            )
        bundle, source = source_for(group)
        record = reference["groups"][group]
        old_path = Path(
            record.get(
                "reference_report", args.reference_reports / f"{group}_long_once.json"
            )
        )
        if (expected := record.get("reference_report_sha256")) and digest(
            old_path
        ) != expected:
            raise ValueError("Unprofiled reference changed")
        old = read(old_path)
        if old.get("config_hash") != source.get("config_hash"):
            raise ValueError("Reference configuration differs")
        if (
            source.get("native_decoder")
            and old.get("native_decoder", {}).get("sha256")
            != source["native_decoder"]["libraries"]["hexagon-v81"]["sha256"]
        ):
            raise ValueError("Reference and profiled runtime kernels differ")
        name = args.root.name + "_" + group
        remote = f"{DEVICE_ROOT}/bundles/{name}"
        adb("shell", f"test ! -e {remote} && mkdir -p {remote}")
        contexts = {}
        for part in range(1, 5):
            binary, contexts[str(part)] = checked_context(group, part)
            push(binary, remote + "/" + binary.name)
        runtime = {
            k: source[k]
            for k in (
                "context_length",
                "context_buckets",
                "config_hash",
                "config",
                "quantize_current_kv",
                "model",
                "split_manifest_sha256",
                "num_parts",
            )
            if k in source
        }
        if package := native(source, remote):
            runtime["native_decoder"] = package
        for key in ("compiled_native_decoder", "native_runtime_override"):
            if key in source:
                runtime[key] = source[key]
        runtime_path = output / f"{group}_runtime_manifest.json"
        save_new(runtime_path, runtime)
        push(runtime_path, remote + "/runtime_manifest.json")
        temperature = re.search(
            r"^\s*temperature:\s*(\d+)", adb("shell", "dumpsys battery"), re.MULTILINE
        )
        identity = {
            "diagnostic_only": True,
            "reference_report": str(old_path),
            "reference_report_sha256": digest(old_path),
            "source_bundle": str(bundle),
            "contexts": contexts,
            "device_bundle": name,
            "runner_sha256": digest(runner),
            "assets_sha256": assets["sha256"],
            "context_buckets": [1024],
            "prompt_tokens": 897,
            "generated_tokens": 128,
            "sessions": 1,
            "decode_steps": [0, 63, 126],
            "temperature_before_tenths_c": temperature.group(1)
            if temperature
            else None,
        }
        save_new(report.with_suffix(".attempt.json"), identity)
        print("MEASURE", group, "one 897+128-token session", flush=True)
        trace = args.root / "traces" / group
        command = [
            sys.executable,
            str(SCRIPTS / "run_device_llm.py"),
            "--adb",
            str(args.adb),
            "run",
            "--name",
            name,
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
            "--optrace-out",
            str(trace),
            "--report",
            str(report),
            "--remote-report-tag",
            name,
        ]
        with report.with_suffix(".stdout.log").open("x") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        data = read(report)
        logs = sorted(trace.glob("*.log"))
        if (
            len(data["op_profiles"]) != 11
            or len(logs) != 44
            or len(data["generated"]) != 128
        ):
            raise ValueError("Incomplete profiled session")
        temperature = re.search(
            r"^\s*temperature:\s*(\d+)", adb("shell", "dumpsys battery"), re.MULTILINE
        )
        save_new(
            report.with_suffix(".complete.json"),
            {
                "report_sha256": digest(report),
                "generated_ids_match_reference": data["generated"] == old["generated"],
                "temperature_after_tenths_c": temperature.group(1)
                if temperature
                else None,
                "logs_sha256": {p.name: digest(p) for p in logs},
            },
        )
        print(
            "DONE",
            group,
            "44 optrace logs; generated IDs match:",
            data["generated"] == old["generated"],
            flush=True,
        )


if __name__ == "__main__":
    main()
