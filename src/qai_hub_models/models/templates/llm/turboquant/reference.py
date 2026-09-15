# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Float64 CPU oracle for QJL-off PolarQuant.

Re-implements turboquant_plus ``PolarQuant`` numerics (commit in
``constants.REFERENCE_COMMIT``):

- encode: ``norm = ||x||``; ``y = R (x / norm)`` (divide by 1 for zero vectors);
  ``idx = searchsorted(midpoints(c), y, side="left")``.
- decode: ``y_hat = c[idx]``; optional ``y_hat /= ||y_hat||``; ``x_hat = R^T y_hat * norm``.

``Rotation.FWHT`` composes the reference ``random_rotation_fast`` pieces
(forward ``D2 H D1``, inverse ``D1 H D2``, ``H`` Sylvester-ordered and scaled by
``1/sqrt(n)``). ``Rotation.DENSE_QR`` is the Haar rotation the reference class
itself uses and exists only to cross-check against that class.
"""

from __future__ import annotations

import hashlib
from functools import cache

import numpy as np

from qai_hub_models.models.templates.llm.turboquant.config import (
    KVCodecSpec,
    Rotation,
)
from qai_hub_models.models.templates.llm.turboquant.constants import (
    CODEBOOK_HEX,
    CODEBOOK_SHA256,
    FWHT_SIGNS,
    FWHT_SIGNS_SHA256,
)

# Reference norm-correction guard (polar_quant.py); unreachable for d=128 codebooks.
NORM_CORRECTION_EPS = 1e-10


@cache
def load_codebook(bits: int, block_size: int) -> np.ndarray:
    """Frozen centroids, float64 ascending. Verified against the recorded digest."""
    centroids = np.array(
        [float.fromhex(h) for h in CODEBOOK_HEX[(bits, block_size)]], dtype="<f8"
    )
    digest = hashlib.sha256(centroids.tobytes()).hexdigest()
    if digest != CODEBOOK_SHA256[(bits, block_size)]:
        raise RuntimeError(f"Codebook ({bits}, {block_size}) digest mismatch.")
    centroids.setflags(write=False)
    return centroids


@cache
def load_boundaries(bits: int, block_size: int) -> np.ndarray:
    centroids = load_codebook(bits, block_size)
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    boundaries.setflags(write=False)
    return boundaries


@cache
def load_fwht_signs(seed: int, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Frozen ``(signs1, signs2)`` as float64 +-1. Verified against the recorded digest."""
    signs = tuple(
        np.array([1.0 if c == "+" else -1.0 for c in s], dtype=np.float64)
        for s in FWHT_SIGNS[(seed, block_size)]
    )
    payload = signs[0].astype(np.int8).tobytes() + signs[1].astype(np.int8).tobytes()
    if hashlib.sha256(payload).hexdigest() != FWHT_SIGNS_SHA256[(seed, block_size)]:
        raise RuntimeError(f"FWHT signs ({seed}, {block_size}) digest mismatch.")
    for s in signs:
        s.setflags(write=False)
    return signs[0], signs[1]


def fwht(x: np.ndarray) -> np.ndarray:
    """Orthonormal Walsh-Hadamard transform over the last axis (Sylvester order)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"FWHT length must be a power of two, got {n}.")
    lead = x.shape[:-1]
    y = x.reshape(-1, n).copy()
    h = 1
    while h < n:
        pairs = y.reshape(-1, n // (2 * h), 2, h)
        a = pairs[:, :, 0, :]
        b = pairs[:, :, 1, :]
        y = np.stack((a + b, a - b), axis=2).reshape(-1, n)
        h *= 2
    return (y / np.sqrt(n)).reshape(*lead, n)


class FWHTRotation:
    """``R = D2 H D1`` with frozen sign vectors."""

    def __init__(self, seed: int, block_size: int) -> None:
        self.signs1, self.signs2 = load_fwht_signs(seed, block_size)
        self.block_size = block_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.signs2 * fwht(self.signs1 * x)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return self.signs1 * fwht(self.signs2 * y)

    def matrix(self) -> np.ndarray:
        """Dense ``R`` such that ``forward(x) == x @ R.T``."""
        return self.forward(np.eye(self.block_size)).T


class DenseQRRotation:
    """Haar rotation matching turboquant_plus ``random_rotation_dense``.

    QR output depends on the LAPACK build, so compare against it with a tolerance.
    """

    def __init__(self, seed: int, block_size: int) -> None:
        rng = np.random.default_rng(seed)
        gaussian = rng.standard_normal((block_size, block_size))
        q, r = np.linalg.qr(gaussian)
        signs = np.sign(np.diag(r))
        signs[signs == 0] = 1.0
        q = q * signs[np.newaxis, :]
        det_sign, _ = np.linalg.slogdet(q)
        if det_sign < 0:
            q[:, 0] = -q[:, 0]
        self.q = q
        self.block_size = block_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.q.T

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64) @ self.q

    def matrix(self) -> np.ndarray:
        return self.q


def make_rotation(
    rotation: Rotation, seed: int, block_size: int
) -> FWHTRotation | DenseQRRotation:
    if rotation == Rotation.FWHT:
        return FWHTRotation(seed, block_size)
    return DenseQRRotation(seed, block_size)


def nearest_centroid_indices(values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid; a value exactly on a boundary maps to the lower one."""
    return np.searchsorted(boundaries, values, side="left").astype(np.uint8)


class PolarQuantReference:
    """Float64 PolarQuant encode/decode for one of K or V."""

    def __init__(
        self,
        spec: KVCodecSpec,
        block_size: int = 128,
        rotation: Rotation = Rotation.FWHT,
        norm_correction: bool = True,
    ) -> None:
        if not spec.is_polar:
            raise ValueError("PolarQuantReference needs a POLAR codec spec.")
        self.spec = spec
        self.block_size = block_size
        self.norm_correction = norm_correction
        self.centroids = load_codebook(spec.bits, block_size)
        self.boundaries = load_boundaries(spec.bits, block_size)
        self.rotation = make_rotation(rotation, spec.seed, block_size)

    def _check_input(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.block_size:
            raise ValueError(
                f"Last axis must be block_size={self.block_size}, got {x.shape}."
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("PolarQuant input contains NaN or Inf.")
        return x

    def rotate_normalized(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(y, norms)`` where ``y`` is the rotated unit vector the indices come from."""
        x = self._check_input(x)
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        safe = np.where(norms > 0, norms, 1.0)
        return self.rotation.forward(x / safe), norms

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(indices uint8 [..., d], norms float64 [..., 1])``."""
        y, norms = self.rotate_normalized(x)
        return nearest_centroid_indices(y, self.boundaries), norms

    def decode(self, indices: np.ndarray, norms: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices)
        if indices.size and int(indices.max()) >= len(self.centroids):
            raise ValueError(
                f"Index {int(indices.max())} out of range for {self.spec.bits}-bit codebook."
            )
        y_hat = self.centroids[indices.astype(np.intp)]
        if self.norm_correction:
            lengths = np.linalg.norm(y_hat, axis=-1, keepdims=True)
            y_hat = y_hat / np.where(lengths > NORM_CORRECTION_EPS, lengths, 1.0)
        return self.rotation.inverse(y_hat) * np.asarray(norms, dtype=np.float64)
