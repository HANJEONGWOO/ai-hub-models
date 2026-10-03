#!/usr/bin/env python3
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Build diagnostic optrace contexts from unchanged historical quantized DLCs."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from convert_parts import DSP_ARCH, SOC_MODEL
from profile_stages_once import digest, read, save_new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parts", type=int, nargs="+", required=True)
    parser.add_argument(
        "--sdk", type=Path, default=Path("/home/hjw/qairt/2.48.0.260626")
    )
    args = parser.parse_args()
    source = read(args.source / "convert_report.json")
    lib = args.sdk / "lib/x86_64-linux-clang"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(lib) + ":" + env.get("LD_LIBRARY_PATH", "")
    args.out.mkdir(parents=True, exist_ok=True)
    for part in args.parts:
        if part not in (1, 2, 3, 4):
            raise ValueError("Expected one of the four historical parts")
        key = f"part{part}_of_4"
        names = list(source["parts"][key]["graphs"])
        target = args.out / key
        target.mkdir(exist_ok=True)
        dlcs = [args.source / f"{name}.dlc" for name in names]
        hashes = {str(p): digest(p) for p in dlcs}
        identity = {
            "source_conversion_sha256": digest(args.source / "convert_report.json"),
            "source_dlc_sha256": hashes,
            "graphs": names,
        }
        if (target / "complete.json").exists():
            old = read(target / "complete.json")
            if (
                old["identity"] != identity
                or digest(target / (key + ".bin")) != old["context_sha256"]
            ):
                raise ValueError("Existing context provenance changed")
            print("REUSE", key, flush=True)
            continue
        save_new(target / "attempt.json", identity)
        htp = target / "htp.json"
        save_new(
            htp,
            {
                "graphs": [{"graph_names": names, "O": 3, "vtcm_mb": 0}],
                "devices": [
                    {
                        "soc_model": SOC_MODEL,
                        "dsp_arch": DSP_ARCH,
                        "pd_session": "unsigned",
                    }
                ],
                "context": {"weight_sharing_enabled": len(names) > 1},
            },
        )
        backend = target / "backend.json"
        save_new(
            backend,
            {
                "backend_extensions": {
                    "shared_library_path": str(lib / "libQnnHtpNetRunExtensions.so"),
                    "config_file_path": str(htp),
                }
            },
        )
        command = [
            str(args.sdk / "bin/x86_64-linux-clang/qnn-context-binary-generator"),
            "--backend",
            str(lib / "libQnnHtp.so"),
            "--model",
            str(lib / "libQnnModelDlc.so"),
            "--dlc_path",
            ",".join(map(str, dlcs)),
            "--binary_file",
            key,
            "--config_file",
            str(backend),
            "--output_dir",
            str(target),
            "--profiling_level",
            "detailed",
            "--profiling_option",
            "optrace",
        ]
        if native := source.get("native_decoder"):
            package = native["libraries"]["x86_64-linux-clang"]
            if digest(Path(package["path"])) != package["sha256"]:
                raise ValueError("Native host package changed")
            command.extend(
                ["--op_packages", package["path"] + ":" + native["interface"]]
            )
        save_new(target / "command.json", {"argv": command, "cwd": str(target)})
        print("BUILD", key, names, flush=True)
        start = time.monotonic()
        with (target / "build.log").open("x") as log:
            subprocess.run(
                command,
                cwd=target,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if hashes != {str(p): digest(p) for p in dlcs}:
            raise ValueError("Source DLC was modified")
        schematics = list(target.glob("*schematic*"))
        if not schematics:
            raise ValueError("No optrace schematic emitted")
        save_new(
            target / "complete.json",
            {
                "identity": identity,
                "context_sha256": digest(target / (key + ".bin")),
                "elapsed_s": time.monotonic() - start,
                "schematics": {p.name: digest(p) for p in schematics},
            },
        )
        print("DONE", key, round(time.monotonic() - start, 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
