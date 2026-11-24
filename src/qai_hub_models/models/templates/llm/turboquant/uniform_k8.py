# SPDX-License-Identifier: BSD-3-Clause
"""Frozen opt-in Gaussian-MSE Uniform K8; the offset is exactly half a step."""

from __future__ import annotations

import hashlib
from functools import cache

import numpy as np

QUANTIZER = "uniform_k8_gaussian_v1"
COEFFICIENT_HEX = ("0x1.6480000000000p-10", "0x1.6480000000000p-9")
CODEBOOK_SHA256 = "a4f590625adbf11125d703869c906e144b246a264fe31ce83a37f8d6549cb523"
FP16_SHA256 = "30d0dfd34829196d478b0ec8b77e71880f8f217ac236a189d7e74805bffa905e"


def coefficients() -> np.ndarray:
    return np.array([float.fromhex(x) for x in COEFFICIENT_HEX], np.float64)


@cache
def codebook() -> np.ndarray:
    offset, step = coefficients().astype(np.float16)
    positive = (np.float16(np.arange(128, dtype=np.float16) * step) + offset).astype(
        np.float16
    )
    values = np.r_[-positive[::-1], positive].astype("<f8")
    if (
        hashlib.sha256(values.tobytes()).hexdigest() != CODEBOOK_SHA256
        or hashlib.sha256(values.astype("<f2").tobytes()).hexdigest() != FP16_SHA256
    ):
        raise RuntimeError("Uniform K8 FP16 arithmetic/codebook digest mismatch")
    values.setflags(write=False)
    return values


def boundaries() -> np.ndarray:
    positive = np.arange(1, 128, dtype=np.float64) * coefficients()[1]
    return np.r_[-positive[::-1], 0.0, positive]


def indices(values: np.ndarray, *, fp16: bool = False) -> np.ndarray:
    dtype = np.float16 if fp16 else np.float64
    y = np.asarray(values, dtype=dtype).astype(np.float64)
    reciprocal = np.asarray(1 / coefficients()[1], dtype=dtype).astype(np.float64)
    position = np.asarray(np.abs(y) * reciprocal, dtype=dtype).astype(np.float64)
    local = np.clip(np.floor(position), 0, 127).astype(np.int32)
    return np.where(y >= 0, 128 + local, 127 - local).astype(np.uint8)


def metadata() -> dict:
    return {
        "quantizer": QUANTIZER,
        "coefficient_hex": list(COEFFICIENT_HEX),
        "codebook_sha256": CODEBOOK_SHA256,
        "codebook_fp16_sha256": FP16_SHA256,
        "distribution": "N(0,1/128)",
        "levels": 256,
        "index": "clip(floor(abs(y)*reciprocal_delta),0,127); reflected negative codes",
        "centroid": "FP16(FP16(local*delta)+delta/2); sign then FP16 effective-scale product",
        "optimizer": "one FP64 spacing, three fixed bounded brackets; nearest FP16; no real-data fitting",
    }
