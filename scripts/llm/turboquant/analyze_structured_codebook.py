# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Held-out scalar and post-norm KV comparison, with no coefficient fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qai_hub_models.models.templates.llm.turboquant.reference import load_codebook
from qai_hub_models.models.templates.llm.turboquant.structured import load_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve previous analysis")
    params = load_parameters()
    seed = params["seed"] + 1
    rng = np.random.default_rng(seed)
    scalar = rng.normal(0, 1 / np.sqrt(128), params["training_samples"])
    vectors = rng.normal(size=(4096, 128))
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    y = (vectors / norms).astype(np.float16)
    results = {}
    for name, book in (("lm", "lloyd_max"), ("structured", "structured4_v1")):
        centroids = load_codebook(4, 128, book)
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        indices = np.searchsorted(boundaries, scalar, side="left")
        selected = centroids.astype(np.float16)[
            np.searchsorted(boundaries.astype(np.float16), y, side="left")
        ]
        scale = (
            norms / np.linalg.norm(selected.astype(np.float64), axis=-1, keepdims=True)
        ).astype(np.float16)
        restored = (
            (selected.astype(np.float32) * scale.astype(np.float32))
            .astype(np.float16)
            .astype(np.float64)
        )
        squared = (vectors - restored) ** 2
        results[name] = {
            "heldout_scalar_mse": float(np.mean((scalar - centroids[indices]) ** 2)),
            "heldout_kv_mse_fp16_scale_lut": float(np.mean(squared)),
            "heldout_kv_relative_mse": float(
                np.mean(squared.sum(axis=-1) / norms[:, 0] ** 2)
            ),
            "heldout_kv_cosine_mean": float(
                np.mean(
                    (vectors * restored).sum(axis=-1)
                    / norms[:, 0]
                    / np.linalg.norm(restored, axis=-1)
                )
            ),
        }
    report = {
        "parameters_sha256": params["sha256"],
        "heldout_seed": seed,
        "scalar_samples": len(scalar),
        "kv_vectors": len(vectors),
        "results": results,
        "note": "No fitting/selection on held-out data. Synthetic rotated-domain vectors; FP16 coordinates/thresholds/LUT/scale/product, float64 norm reduction. Not full HTP encoder simulation.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
