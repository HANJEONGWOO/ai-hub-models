# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Compare already collected HTP scores; no device execution or timing rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from htp_codec_validation import read_native


def errors(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float64)
    delta = actual.astype(np.float64) - reference
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(reference)),
    }


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x.astype(np.float64) - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve previous analysis")
    work = args.work_dir
    manifest = json.loads((work / "manifest.json").read_text())
    sequences = sorted(
        {g["seq"] for g in manifest["graphs"].values() if g["operation"] == "qk"}
    )
    report = {
        "scope": "Post-processing of the same single-execute HTP diagnostic outputs",
        "note": "Exclude tokens 0..3 in every head: the diagnostic injects all nibble codes into the first four K rows of head 0. Attention here is softmax(device QK) @ shared synthetic V on CPU, not a second device attention measurement. Queries already include 1/sqrt(d).",
        "sequences": {},
    }
    for seq in sequences:
        scores, queries = {}, []
        for group in ("lm", "structured_lut", "bitplane"):
            name = f"{group}_qk_ar{seq}"
            tokens = manifest["graphs"][name]["tokens"]
            if tokens <= 4:
                raise ValueError("Need more than four diagnostic tokens")
            scores[group] = read_native(
                work / f"out_{name}/Result_0",
                "score",
                "float16",
                (8, 2, seq, tokens),
            )[..., 4:]
            queries.append((work / f"{name}_query.raw").read_bytes())
        if not queries[0] == queries[1] == queries[2]:
            raise ValueError("QK groups do not share queries")
        for kind in ("packed", "scale"):
            if (work / f"structured_lut_qk_ar{seq}_{kind}.raw").read_bytes() != (
                work / f"bitplane_qk_ar{seq}_{kind}.raw"
            ).read_bytes():
                raise ValueError("LUT/Bit-plane cache inputs differ")
        values = np.random.default_rng(893).normal(size=(8, 1, tokens, 128))[:, :, 4:]
        attention = {name: softmax(score) @ values for name, score in scores.items()}
        report["sequences"][str(seq)] = {
            name: {
                "qk": errors(scores[after], scores[before]),
                "synthetic_attention": errors(attention[after], attention[before]),
            }
            for name, before, after in (
                ("codebook_change", "lm", "structured_lut"),
                ("computation_change", "structured_lut", "bitplane"),
            )
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
