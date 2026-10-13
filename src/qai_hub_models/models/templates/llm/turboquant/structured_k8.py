# SPDX-License-Identifier: BSD-3-Clause
"""Frozen opt-in sign/region/local K8; coefficients fit Gaussian MSE offline.

Logical magnitude code is 64*r+k. Negative byte codes are reflected (127-j)
to preserve the ascending, symmetric Native LUT ABI; positive codes are 128+j.
The region/local fields of negative bytes are therefore complemented.
"""

from __future__ import annotations

import hashlib
from functools import cache

import numpy as np

QUANTIZER = "structured_k8_two_region_v1"
COEFFICIENT_HEX = (
    "0x1.d400000000000p-11",
    "0x1.0080000000000p-9",
    "0x1.03c0000000000p-3",
    "0x1.ed00000000000p-9",
)
CODEBOOK_SHA256 = "c407f658ea5293ea6ba13c741534cf813d3f9ee363f8020479caa387fbcd4181"


def coefficients() -> np.ndarray:
    return np.array([float.fromhex(h) for h in COEFFICIENT_HEX], dtype=np.float64)


@cache
def codebook() -> np.ndarray:
    a, h, b, t = coefficients().astype(np.float16)
    k = np.arange(64, dtype=np.float16)
    positive = np.r_[(k * h + a).astype(np.float16), (k * t + b).astype(np.float16)]
    values = np.r_[-positive[::-1], positive].astype("<f8")
    if hashlib.sha256(values.tobytes()).hexdigest() != CODEBOOK_SHA256:
        raise RuntimeError("Structured K8 centroid/FP16 arithmetic hash mismatch")
    values.setflags(write=False)
    return values


def region_boundary() -> float:
    values = codebook()
    return float((values[191] + values[192]) / 2)


def boundaries() -> np.ndarray:
    """Actual arithmetic decision edges, not rounded-LUT nearest midpoints."""
    a, h, b, t = coefficients()
    positive = np.r_[
        a + (np.arange(63) + 0.5) * h, region_boundary(), b + (np.arange(63) + 0.5) * t
    ]
    return np.r_[-positive[::-1], 0.0, positive]


def metadata() -> dict:
    return {
        "quantizer": QUANTIZER,
        "coefficient_hex": list(COEFFICIENT_HEX),
        "codebook_sha256": CODEBOOK_SHA256,
        "distribution": "N(0,1/128)",
        "index": "j=64*r+clip(floor((abs(y)-offset[r])/step[r]+0.5),0,63); y>=0?128+j:127-j",
        "region_boundary": region_boundary(),
        "centroid": "FP16(FP16(local*step[r])+offset[r]); symmetric signed LUT",
        "negative_fields": "region/local complemented to retain ascending LUT ABI",
        "optimizer": "nine fixed initializations; analytic Gaussian scalar MSE; no real-data tuning",
    }


def indices(values: np.ndarray, *, fp16: bool = False) -> np.ndarray:
    """Arithmetic rule; optional FP16 operation-boundary emulator, not HTP exact."""
    a, h, b, t = coefficients()

    def rounded(x: np.ndarray | float) -> np.ndarray:
        return np.asarray(x, dtype=np.float16 if fp16 else np.float64).astype(
            np.float64
        )

    y = rounded(values)
    m = np.abs(y)
    r = m > rounded(region_boundary())
    offset = np.where(r, b, a)
    reciprocal = np.where(r, rounded(1 / t), rounded(1 / h))
    position = rounded(rounded(rounded(m - offset) * reciprocal) + 0.5)
    local = np.clip(np.floor(position), 0, 63).astype(np.int32)
    j = r.astype(np.int32) * 64 + local
    return np.where(y >= 0, 128 + j, 127 - j).astype(np.uint8)
