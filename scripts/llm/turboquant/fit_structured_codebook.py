# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Fit four shared beta constants on Gaussian coordinates, never LM centroids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qai_hub_models.models.templates.llm.turboquant.structured import (
    STRUCTURED,
    centroids_from_beta,
    fit_beta,
    gap_transform,
    parameter_digest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--samples", type=int, default=262144)
    parser.add_argument("--starts", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite a frozen fit or report")
    rng = np.random.default_rng(args.seed)
    d = 128
    train = rng.normal(0, 1 / np.sqrt(d), args.samples)
    candidates = []
    for start in range(args.starts):
        gaps = np.full(4, 0.015) if start == 0 else rng.uniform(0.003, 0.04, 4)
        beta, history = fit_beta(train, gap_transform() @ gaps)
        candidates.append(
            {
                "start": start,
                "beta": beta.tolist(),
                "mse": history[-1],
                "history": history,
            }
        )
    best = min(candidates, key=lambda c: c["mse"])
    # Export beta as exactly representable float32 constants for the native op.
    beta = np.asarray(best["beta"], dtype=np.float32).astype(np.float64)
    c = centroids_from_beta(beta)
    boundaries = (c[:-1] + c[1:]) / 2
    data = {
        "name": STRUCTURED,
        "block_size": d,
        "beta": beta.tolist(),
        "centroids": c.tolist(),
        "boundaries": boundaries.tolist(),
        "bit_order": "natural_binary_lsb_beta0",
        "seed": args.seed,
        "distribution": "N(0,1/d)",
        "training_samples": args.samples,
        "starts": args.starts,
        "selected_start": best["start"],
        "objective": "scalar_mse",
        "min_gap": 1e-6,
        "max_iterations": 200,
        "tolerance": 1e-10,
        "global_optimum_claimed": False,
    }
    data["sha256"] = parameter_digest(data)
    evaluation = np.random.default_rng(args.seed + 1)
    held = evaluation.normal(0, 1 / np.sqrt(d), args.samples)
    idx = np.searchsorted(boundaries, held, side="left")
    # Independent isotropic vectors: normalization/scale and FP16 LUT rounding.
    vectors = evaluation.normal(size=(4096, d))
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    y = vectors / norms
    index = np.searchsorted(
        boundaries.astype(np.float16), y.astype(np.float16), side="left"
    )
    selected = c.astype(np.float16)[index].astype(np.float64)
    scale = (norms / np.linalg.norm(selected, axis=-1, keepdims=True)).astype(
        np.float16
    )
    reconstructed = (
        (selected.astype(np.float32) * scale.astype(np.float32))
        .astype(np.float16)
        .astype(np.float64)
    )
    report = {
        "parameters_sha256": data["sha256"],
        "candidates": candidates,
        "evaluation_seed": args.seed + 1,
        "heldout_scalar_mse": float(np.mean((held - c[idx]) ** 2)),
        "heldout_kv_mse_fp16_scale_lut": float(np.mean((vectors - reconstructed) ** 2)),
        "heldout_kv_relative_mse": float(
            np.mean(np.sum((vectors - reconstructed) ** 2, axis=-1) / norms[:, 0] ** 2)
        ),
        "kv_evaluation_note": "Synthetic rotated-domain vectors; FP16 coordinates/thresholds/LUT/scale/product; norm reduction is float64. Not a full HTP encoder simulation.",
        "selection": "training scalar MSE only; heldout data never selects coefficients",
    }
    for path, payload in ((args.out, data), (args.report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
