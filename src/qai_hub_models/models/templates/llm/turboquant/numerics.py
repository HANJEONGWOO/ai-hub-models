# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Comparison of executed codec graphs against the float64 oracle.

Byte packing is always exact. Float execution may flip an index only to an
adjacent centroid and only when the oracle's rotated coordinate lies within
``boundary_distance`` of the boundary between them; norms and reconstructions
are compared relative to the oracle norm. Tolerances are fixed per backend in
``tutorials/llm/turboquant_design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qai_hub_models.models.templates.llm.turboquant.packing import unpack_indices
from qai_hub_models.models.templates.llm.turboquant.reference import (
    PolarQuantReference,
)

TINY = 1e-30


@dataclass(frozen=True)
class Tolerance:
    boundary_distance: float
    norm_rel: float
    decode_rel: float


# float32 execution (e.g. onnxruntime CPU) of the exported graphs.
FLOAT32_GRAPH = Tolerance(boundary_distance=1e-5, norm_rel=1e-5, decode_rel=1e-5)
# HTP executes float graphs with float16 math. One wrong centroid costs ~2e-2 decode_rel.
HTP_FP16 = Tolerance(boundary_distance=1e-3, norm_rel=2e-3, decode_rel=5e-3)


def compare_encode(
    codec: PolarQuantReference,
    x: np.ndarray,
    packed: np.ndarray,
    norms: np.ndarray,
    tol: Tolerance,
) -> dict[str, Any]:
    """Check executed ``(packed, norms)`` for input ``x`` against the oracle."""
    bits = codec.spec.bits
    y, ref_norms = codec.rotate_normalized(x)
    ref_idx = np.searchsorted(codec.boundaries, y, side="left")
    dev_idx = unpack_indices(
        np.asarray(packed, np.uint8), bits, codec.block_size
    ).astype(np.int64)
    mismatch = dev_idx != ref_idx
    # Distance to the boundary between the two centroids involved, not to any boundary.
    between = np.clip(np.minimum(dev_idx, ref_idx), 0, len(codec.boundaries) - 1)
    distance = np.abs(y - codec.boundaries[between])
    allowed = (np.abs(dev_idx - ref_idx) == 1) & (distance <= tol.boundary_distance)
    unexplained = mismatch & ~allowed

    norms = np.asarray(norms, np.float64)
    norm_rel = np.abs(norms - ref_norms) / np.maximum(ref_norms, TINY)
    zero_rows = ref_norms[..., 0] == 0
    report = {
        "values": int(ref_idx.size),
        "index_mismatches": int(mismatch.sum()),
        "index_mismatch_fraction": float(mismatch.mean()),
        "unexplained_index_mismatches": int(unexplained.sum()),
        "max_boundary_distance_of_mismatch": float(distance[mismatch].max())
        if mismatch.any()
        else 0.0,
        "norm_max_rel_error": float(norm_rel[~zero_rows].max())
        if (~zero_rows).any()
        else 0.0,
        "zero_vectors": int(zero_rows.sum()),
        "zero_vector_norms_exact": bool(np.all(norms[zero_rows] == 0)),
    }
    report["passed"] = bool(
        report["unexplained_index_mismatches"] == 0
        and report["norm_max_rel_error"] <= tol.norm_rel
        and report["zero_vector_norms_exact"]
    )
    return report


def compare_decode(
    codec: PolarQuantReference,
    packed: np.ndarray,
    norms: np.ndarray,
    x_hat: np.ndarray,
    tol: Tolerance,
) -> dict[str, Any]:
    """Check an executed decode against the oracle decode of the same inputs.

    The error is ``||x_hat - expected||_2 / norm`` per vector; a max-element metric
    would dilute a single wrong centroid by the rotation's ``1/sqrt(d)`` spread.
    """
    indices = unpack_indices(
        np.asarray(packed, np.uint8), codec.spec.bits, codec.block_size
    )
    norms = np.asarray(norms, np.float64)
    expected = codec.decode(indices, norms)
    err = np.linalg.norm(
        np.asarray(x_hat, np.float64) - expected, axis=-1, keepdims=True
    )
    zero_rows = norms == 0
    rel = err / np.maximum(norms, TINY)
    report = {
        "vectors": int(norms.size),
        "decode_max_rel_error": float(rel[~zero_rows].max())
        if (~zero_rows).any()
        else 0.0,
        "zero_vectors_decode_exact": bool(np.all(err[zero_rows] == 0)),
    }
    report["passed"] = bool(
        report["decode_max_rel_error"] <= tol.decode_rel
        and report["zero_vectors_decode_exact"]
    )
    return report


def reconstruction_stats(x: np.ndarray, x_hat: np.ndarray) -> dict[str, float]:
    """Codec quality on real data: relative MSE and cosine similarity per vector."""
    x = np.asarray(x, np.float64)
    x_hat = np.asarray(x_hat, np.float64)
    sq_norm = np.sum(x * x, axis=-1)
    valid = sq_norm > 0
    rel_mse = np.sum((x - x_hat) ** 2, axis=-1)[valid] / sq_norm[valid]
    cos = np.sum(x * x_hat, axis=-1)[valid] / np.sqrt(
        sq_norm[valid] * np.sum(x_hat * x_hat, axis=-1)[valid]
    )
    return {
        "rel_mse_mean": float(rel_mse.mean()),
        "rel_mse_p99": float(np.quantile(rel_mse, 0.99)),
        "cosine_mean": float(cos.mean()),
        "cosine_min": float(cos.min()),
    }
