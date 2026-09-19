# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Build the QHPI prepare library and the V81 execution library without SDK copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdk", type=Path, default=Path("~/qairt/2.48.0.260626").expanduser()
    )
    parser.add_argument(
        "--hexagon-tools",
        type=Path,
        default=Path(
            "~/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools"
        ).expanduser(),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--test-hvx",
        action="store_true",
        help="Also test the exact vector kernel with Hexagon libnative on the host",
    )
    args = parser.parse_args()
    args.out = args.out.expanduser().resolve()
    args.sdk = args.sdk.expanduser().resolve()
    args.hexagon_tools = args.hexagon_tools.expanduser().resolve()
    source = Path(__file__).with_name("decoder.cpp")
    common = [
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-shared",
        "-fno-exceptions",
        "-fno-rtti",
        "-I",
        str(args.sdk / "include/QNN"),
        str(source),
    ]
    targets = {
        "x86_64-linux-clang": ["clang++", *common],
        "hexagon-v81": [
            str(args.hexagon_tools / "bin/hexagon-clang++"),
            "-mv81",
            "-mhvx",
            "-mhvx-length=128B",
            *common,
        ],
    }
    manifest = {
        "package": "TurboQuantNative",
        "interface": "TurboQuantInterfaceProvider",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_files": {
            name: hashlib.sha256(source.with_name(name).read_bytes()).hexdigest()
            for name in ("decoder.cpp", "hvx_decode.h", "Decode4.xml")
        },
        "qairt_sdk": str(args.sdk),
        "hexagon_tools": str(args.hexagon_tools),
        "libraries": {},
    }
    for target, command in targets.items():
        output = args.out / target / "libTurboQuantNative.so"
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([*command, "-o", str(output)], check=True)
        manifest["libraries"][target] = {
            "path": str(output.resolve()),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
        print(f"Built {output}", flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.test_hvx:
        libnative = args.hexagon_tools / "libnative"
        executable = args.out / "test_hvx"
        subprocess.run(
            [
                "clang++",
                "-std=c++17",
                "-O2",
                "-D__HVXDBL__",
                "-I",
                str(libnative / "include"),
                str(source.with_name("test_hvx.cpp")),
                "-L",
                str(libnative / "lib"),
                f"-Wl,-rpath,{libnative / 'lib'}",
                "-lnative",
                "-lpthread",
                "-o",
                str(executable),
            ],
            check=True,
        )
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
