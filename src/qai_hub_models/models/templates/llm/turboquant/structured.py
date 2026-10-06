# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Frozen K-only additive binary codebook. No Lloyd-Max centroid fitting.

Bit 0 is the least significant bit of the *natural* nibble code. Strict
ordering is equivalent to beta[m] > sum(beta[:m]) (including beta[0] > 0).
The offline optimizer parameterizes these four positive gaps explicitly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

STRUCTURED = "structured4_v1"
PARAMETERS = Path(__file__).with_name("structured4_v1.json")


def bit_signs() -> np.ndarray:
    return 2.0 * ((np.arange(16)[:, None] >> np.arange(4)) & 1) - 1.0


def gap_transform() -> np.ndarray:
    transform = np.eye(4)
    for m in range(1, 4):
        transform[m] += transform[:m].sum(axis=0)
    return transform


def centroids_from_beta(beta: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape != (4,) or not np.all(np.isfinite(beta)):
        raise ValueError("Expected four finite beta coefficients")
    if any(beta[m] <= beta[:m].sum() for m in range(4)):
        raise ValueError("Natural binary codebook must be strictly increasing")
    # Explicit mirrored construction also fixes floating-point symmetry.
    positive = bit_signs()[8:] @ beta
    return np.r_[-positive[::-1], positive]


def parameter_digest(data: dict) -> str:
    payload = {k: v for k, v in data.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_parameters() -> dict:
    # Deliberately revalidate the file: edited constants must not reuse a bundle.
    data = json.loads(PARAMETERS.read_text())
    if data["sha256"] != parameter_digest(data):
        raise ValueError("Structured codebook parameter hash mismatch")
    c = centroids_from_beta(np.asarray(data["beta"]))
    if data["block_size"] != 128 or data["name"] != STRUCTURED:
        raise ValueError("Unsupported structured codebook")
    if not np.array_equal(c, data["centroids"]):
        raise ValueError("Structured centroid/bit-code mismatch")
    if not np.array_equal((c[:-1] + c[1:]) / 2, data["boundaries"]):
        raise ValueError("Structured midpoint mismatch")
    return data


def fit_beta(
    samples: np.ndarray,
    initial: np.ndarray,
    *,
    min_gap: float = 1e-6,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, list[float]]:
    """Constrained Lloyd iteration on raw samples, not existing LM centroids.

    Sufficient statistics give the exact fixed-assignment least-squares
    problem in only 16 rows. NNLS on positive gaps enforces code order.
    Only scalar MSE is optimized; post-scale KV error is reported separately.
    """
    from scipy.optimize import nnls

    x = np.asarray(samples, dtype=np.float64).ravel()
    if not x.size or not np.isfinite(x).all() or min_gap <= 0:
        raise ValueError("Need finite nonempty samples and a positive minimum gap")
    transform = gap_transform()
    design = bit_signs() @ transform
    beta = np.asarray(initial, dtype=np.float64).copy()
    history = []
    for _ in range(max_iterations):
        c = centroids_from_beta(beta)
        assigned = np.searchsorted((c[:-1] + c[1:]) / 2, x, side="left")
        counts = np.bincount(assigned, minlength=16)
        sums = np.bincount(assigned, weights=x, minlength=16)
        weights = np.sqrt(counts)
        means = sums / np.maximum(counts, 1)
        a = weights[:, None] * design
        rhs = weights * means - a @ np.full(4, min_gap)
        gaps = nnls(a, rhs)[0] + min_gap
        updated = transform @ gaps
        new_c = centroids_from_beta(updated)
        idx = np.searchsorted((new_c[:-1] + new_c[1:]) / 2, x, side="left")
        mse = float(np.mean((x - new_c[idx]) ** 2))
        history.append(mse)
        change = np.max(np.abs(updated - beta))
        beta = updated
        if change < tolerance:
            break
    return beta, history
