#!/usr/bin/env python3
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Decode immutable QNN optrace logs using their exact build-time schematics."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from profile_stages_once import digest, read, save_new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pattern", default="*.log")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--sdk", type=Path, default=Path("/home/hjw/qairt/2.48.0.260626")
    )
    args = parser.parse_args()
    lib = args.sdk / "lib/x86_64-linux-clang"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(lib) + ":" + env.get("LD_LIBRARY_PATH", "")
    logs = sorted(args.logs.glob(args.pattern))
    if not logs:
        raise ValueError("No trace logs")
    args.out.mkdir(parents=True, exist_ok=True)
    config = args.out / "viewer_config.json"
    features = {
        "features": {
            "htp_json": False,
            "qhas_json": False,
            "qhas_schema": False,
            "traceback": False,
            "memory_info": False,
            "runtrace": True,
            "enable_input_output_flow_events": False,
            "enable_sequencer_flow_events": False,
        }
    }
    if config.exists():
        if read(config) != features:
            raise ValueError("Viewer config changed")
    else:
        save_new(config, features)

    def render(log: Path) -> None:
        match = re.fullmatch(
            r"execute_\d+_((?:prompt|token)_ar\d+_cl1024_(\d+)_of_4)", log.stem
        )
        if not match:
            raise ValueError("Unexpected trace name " + log.name)
        graph, part = match.groups()
        schematic = args.contexts / f"part{part}_of_4" / f"{graph}_schematic.bin"
        expected = read(schematic.parent / "complete.json")["schematics"][
            schematic.name
        ]
        if digest(schematic) != expected:
            raise ValueError("Changed schematic")
        out = args.out / log.stem
        identity = {"log_sha256": digest(log), "schematic_sha256": expected}
        if (out / "complete.json").exists():
            if read(out / "complete.json") != identity:
                raise ValueError("Rendered trace provenance differs")
            return
        out.mkdir(exist_ok=False)
        command = [
            str(args.sdk / "bin/x86_64-linux-clang/qnn-profile-viewer"),
            "--reader",
            str(lib / "libQnnHtpOptraceProfilingReader.so"),
            "--input_log",
            str(log),
            "--schematic",
            str(schematic),
            "--output",
            str(out / "trace.json.gz"),
            "--config",
            str(config),
        ]
        save_new(out / "command.json", {"argv": command})
        with (out / "viewer.log").open("x") as stream:
            subprocess.run(
                command,
                cwd=out,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        save_new(out / "complete.json", identity)
        print("DECODED", log.name, flush=True)

    if not 1 <= args.jobs <= 4:
        raise ValueError("Use 1 to 4 viewer workers")
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        list(executor.map(render, logs))


if __name__ == "__main__":
    main()
