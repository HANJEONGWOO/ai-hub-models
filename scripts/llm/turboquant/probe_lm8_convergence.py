# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Same-input encoder/decoder diagnostics, separate from unprofiled performance."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_asymmetric_once import ARTIFACTS, execute, read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("encoder", "decoder"),
        default=["encoder", "decoder"],
    )
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = read(root / "protocol.json")
    for group, profile in (
        ("lm8_100", "k8_v4_scaled"),
        ("lm8_converged", "lm8_converged_v4_scaled"),
        ("structured", "structured_k8_v4_scaled"),
    ):
        if "encoder" not in args.components:
            continue
        work = root / ("encoder_" + group)
        if args.stage == "build" and work.exists():
            raise FileExistsError(work)
        options = [
            "--work-dir",
            str(work),
            "--snapshot",
            str(ARTIFACTS / "qwen3_1_7b_kv_snapshot.npz"),
            "--profile",
            profile,
            "--range-scale",
            "0.5",
            "--tokens",
            "1",
            "128",
            "--head-major",
            "--operations",
            "encode",
            "--device-dir",
            f"/data/local/tmp/qaihm_turboquant/{root.name}_{group}_encoder",
        ]
        execute(
            "htp_codec_validation.py",
            [args.stage, *options],
            root / "logs" / f"encoder_{group}_{args.stage}.log",
        )
        if args.stage == "run":
            execute(
                "htp_codec_validation.py",
                ["compare", *options],
                root / "logs" / f"encoder_{group}_compare.log",
            )
    # Extra isolated Structured LUT control separates decoder from quantizer.
    # It is not a fourth full-model performance/PPL configuration.
    for group, lut, arithmetic in (
        ("lm8_100", "mse4", False),
        ("lm8_converged", "lm8_converged", False),
        ("structured", "structured_k8", True),
        ("structured_lut_control", "structured_k8", False),
    ):
        if "decoder" not in args.components:
            continue
        work = root / ("decoder_" + group)
        if args.stage == "build" and work.exists():
            raise FileExistsError(work)
        options = [
            "--work-dir",
            str(work),
            "--package",
            protocol["native"],
            "--bits",
            "8",
            "--lut",
            lut,
            "--tokens",
            "1",
            "3",
            "256",
            "1023",
            "--scale-edges",
            "--strict-bit-exact",
            "--optrace",
            "--device-dir",
            f"/data/local/tmp/qaihm_turboquant/{root.name}_{group}_decoder",
        ]
        if arithmetic:
            options += ["--structured-arithmetic"]
        execute(
            "validate_native_decoder.py",
            [args.stage, *options],
            root / "logs" / f"decoder_{group}_{args.stage}.log",
        )
        if args.stage == "run":
            execute(
                "validate_native_decoder.py",
                ["compare", *options],
                root / "logs" / f"decoder_{group}_compare.log",
            )


if __name__ == "__main__":
    main()
