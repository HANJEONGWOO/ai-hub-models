# SPDX-License-Identifier: BSD-3-Clause
"""Independent V2/V4/V8 HTP codec diagnostics, not LLM performance repeats."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_asymmetric_once import ARTIFACTS, digest, execute, read, save_new
from benchmark_boundary_v_once import validate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "summarize"))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    validate(root)
    if args.action == "run":
        snapshot = ARTIFACTS / "qwen3_1_7b_kv_snapshot.npz"
        save_new(
            root / "probe_protocol.json",
            {
                "snapshot": str(snapshot),
                "snapshot_sha256": digest(snapshot),
                "encoder_profiles": ["k4_v2_scaled", "k4_v4_scaled", "k4_v8_scaled"],
                "encoder_tokens": [1, 128],
                "encoder_cases": ["layer0", "layer27", "range"],
                "synthetic_range_scale": 0.5,
                "real_snapshots_unscaled": True,
                "decoder_bits": [2, 4, 8],
                "decoder_tokens": [1, 3, 256],
                "tolerances": "unchanged HTP_FP16; failures reported independently",
            },
        )
        for bits in (2, 4, 8):
            name = f"encoder_v{bits}"
            execute(
                "htp_codec_validation.py",
                [
                    "all",
                    "--work-dir",
                    str(root / name),
                    "--snapshot",
                    str(snapshot),
                    "--profile",
                    f"k4_v{bits}_scaled",
                    "--tokens",
                    "1",
                    "128",
                    "--operations",
                    "encode",
                    "--head-major",
                    "--range-scale",
                    "0.5",
                    "--device-dir",
                    f"/data/local/tmp/qaihm_turboquant/{root.name}_{name}",
                ],
                root / "logs" / f"{name}.log",
            )
            name = f"decoder_b{bits}"
            execute(
                "validate_native_decoder.py",
                [
                    "all",
                    "--work-dir",
                    str(root / name),
                    "--package",
                    str(root / "native"),
                    "--bits",
                    str(bits),
                    "--tokens",
                    "1",
                    "3",
                    "256",
                    "--device-dir",
                    f"/data/local/tmp/qaihm_turboquant/{root.name}_{name}",
                ],
                root / "logs" / f"{name}.log",
            )
    else:
        sources, encoders, decoders = {}, {}, {}
        for bits in (2, 4, 8):
            path = root / f"encoder_v{bits}" / "p2_report.json"
            sources[str(path)] = digest(path)
            encoders[str(bits)] = read(path)
            path = root / f"decoder_b{bits}" / "correctness.json"
            sources[str(path)] = digest(path)
            decoders[str(bits)] = read(path)
        save_new(
            root / "codec_validation.json",
            {
                "source_sha256": sources,
                "encoders": encoders,
                "native_decoders": decoders,
                "limits": "Detailed encoder SDK op cycles are diagnostic, not physical optrace or end-to-end latency; AR128 scale gate is not relaxed",
            },
        )


if __name__ == "__main__":
    main()
