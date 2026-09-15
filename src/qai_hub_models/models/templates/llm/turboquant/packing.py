# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Byte packing for TurboQuant centroid indices and block norms.

Format version 1 packs the indices of one block (last axis) as a continuous
MSB-first bit stream, the same order as turboquant_plus ``pack_indices``: the
first index occupies the most significant bits of byte 0 and the tail of the
last byte is zero-padded. For 4-bit this is ``byte = idx[2k] << 4 | idx[2k+1]``.
"""

from __future__ import annotations

import numpy as np

FLOAT16_MAX = float(np.finfo(np.float16).max)
FLOAT16_SMALLEST_NORMAL = float(np.finfo(np.float16).smallest_normal)


def packed_nbytes(num_values: int, bits: int) -> int:
    _check_bits(bits)
    return (num_values * bits + 7) // 8


def _check_bits(bits: int) -> None:
    if not 1 <= bits <= 8:
        raise ValueError(f"Bit width must be in [1, 8], got {bits}.")


def pack_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    """Pack ``[..., n]`` indices into ``[..., ceil(n * bits / 8)]`` uint8 bytes."""
    _check_bits(bits)
    indices = np.asarray(indices)
    if indices.dtype.kind not in "ui":
        raise TypeError(f"Indices must be integers, got {indices.dtype}.")
    if indices.size and (indices.min() < 0 or indices.max() >= 1 << bits):
        raise ValueError(f"Index out of range for {bits}-bit packing.")
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint8)
    bit_planes = (indices.astype(np.uint8)[..., np.newaxis] >> shifts) & 1
    stream = bit_planes.reshape(*indices.shape[:-1], indices.shape[-1] * bits)
    return np.packbits(stream, axis=-1, bitorder="big")


def unpack_indices(
    packed: np.ndarray, bits: int, num_values: int, strict: bool = True
) -> np.ndarray:
    """Inverse of :func:`pack_indices`; ``strict`` rejects non-zero tail padding."""
    _check_bits(bits)
    packed = np.asarray(packed)
    if packed.dtype != np.uint8:
        raise TypeError(f"Packed data must be uint8, got {packed.dtype}.")
    expected = packed_nbytes(num_values, bits)
    if packed.shape[-1] != expected:
        raise ValueError(
            f"Expected {expected} bytes for {num_values} {bits}-bit values, "
            f"got {packed.shape[-1]}."
        )
    stream = np.unpackbits(packed, axis=-1, bitorder="big")
    if strict and np.any(stream[..., num_values * bits :]):
        raise ValueError("Non-zero padding bits; packed data is corrupt or mislabeled.")
    planes = stream[..., : num_values * bits].reshape(
        *packed.shape[:-1], num_values, bits
    )
    weights = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint16)
    return (planes.astype(np.uint16) @ weights).astype(np.uint8)


def to_storage_norms(norms: np.ndarray, dtype: str) -> np.ndarray:
    """Cast block norms to the storage dtype, rejecting values that would not survive.

    Exact zeros are allowed; non-zero norms outside the dtype's normal range
    raise instead of silently rounding to zero, infinity or a coarse subnormal.
    """
    norms = np.asarray(norms)
    if not np.all(np.isfinite(norms)) or np.any(norms < 0):
        raise ValueError("Block norms must be finite and non-negative.")
    if dtype != "float16" or not norms.size:
        return norms.astype(dtype)
    if float(norms.max()) > FLOAT16_MAX:
        raise OverflowError(
            f"Block norm {float(norms.max())} exceeds float16 range; "
            "use norm_dtype='float32'."
        )
    nonzero = norms[norms > 0]
    if nonzero.size and float(nonzero.min()) < FLOAT16_SMALLEST_NORMAL:
        raise ValueError(
            f"Block norm {float(nonzero.min())} underflows float16; "
            "use norm_dtype='float32'."
        )
    return norms.astype(dtype)
