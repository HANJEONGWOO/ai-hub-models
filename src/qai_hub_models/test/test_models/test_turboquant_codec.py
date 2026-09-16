# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""TurboQuant codec oracle, packing and config tests.

The golden fixture was produced by running turboquant_plus at the pinned commit
(``scripts/llm/turboquant/verify_reference.py --write-golden``), so these tests
pin the oracle to the reference without needing that repo in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.linalg import hadamard

from qai_hub_models.models.templates.llm.turboquant.config import (
    BASELINE,
    PROFILES,
    CodecKind,
    KVCodecSpec,
    Rotation,
    TurboQuantConfig,
    get_profile,
)
from qai_hub_models.models.templates.llm.turboquant.constants import CODEBOOK_HEX
from qai_hub_models.models.templates.llm.turboquant.packing import (
    FLOAT16_MAX,
    pack_indices,
    packed_nbytes,
    to_storage_norms,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.reference import (
    FWHTRotation,
    PolarQuantReference,
    fwht,
    load_boundaries,
    load_codebook,
    nearest_centroid_indices,
)

GOLDEN_PATH = Path(__file__).parent / "turboquant_golden_v1.json"
D = 128
K4 = KVCodecSpec(CodecKind.POLAR, bits=4, seed=42)
V4 = KVCodecSpec(CodecKind.POLAR, bits=4, seed=542)
V3 = KVCodecSpec(CodecKind.POLAR, bits=3, seed=542)


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text())


# ----------------------------------------------------------------- rotation


def test_fwht_matches_scipy_hadamard() -> None:
    x = np.random.default_rng(0).standard_normal((5, D))
    expected = x @ (hadamard(D) / np.sqrt(D)).T
    np.testing.assert_allclose(fwht(x), expected, rtol=0, atol=1e-12)


def test_fwht_rejects_non_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        fwht(np.zeros((2, 100)))


@pytest.mark.parametrize("seed", [42, 542])
def test_fwht_rotation_round_trip_and_norm(seed: int) -> None:
    rot = FWHTRotation(seed, D)
    x = np.random.default_rng(1).standard_normal((3, 7, D))
    y = rot.forward(x)
    np.testing.assert_allclose(rot.inverse(y), x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-12
    )
    np.testing.assert_allclose(x @ rot.matrix().T, y, rtol=0, atol=1e-12)
    # Forward is D2 H D1, inverse D1 H D2.
    h = hadamard(D) / np.sqrt(D)
    dense = np.diag(rot.signs2) @ h @ np.diag(rot.signs1)
    np.testing.assert_allclose(rot.matrix(), dense, rtol=0, atol=1e-12)


def test_key_and_value_signs_differ() -> None:
    assert not np.array_equal(FWHTRotation(42, D).signs1, FWHTRotation(542, D).signs1)


# ----------------------------------------------------------------- codebook


@pytest.mark.parametrize("bits", [3, 4])
def test_codebook_sorted_and_frozen(bits: int) -> None:
    c = load_codebook(bits, D)
    assert len(c) == 1 << bits
    assert np.all(np.diff(c) > 0)
    assert np.all(np.diff(c.astype(np.float16)) > 0)
    assert not c.flags.writeable


def test_codebook_is_100_iteration_lloyd_not_doc_table() -> None:
    c = load_codebook(4, D)
    # Doc table (0.1739) is the zero-iteration init; converged Lloyd-Max is 0.241529.
    assert abs(c[-1] - 0.24021009724443104) < 1e-15


@pytest.mark.parametrize("bits", [3, 4])
def test_boundary_ties_go_to_lower_index(bits: int) -> None:
    b = load_boundaries(bits, D)
    k = np.arange(len(b))
    np.testing.assert_array_equal(nearest_centroid_indices(b, b), k)
    np.testing.assert_array_equal(
        nearest_centroid_indices(np.nextafter(b, np.inf), b), k + 1
    )
    np.testing.assert_array_equal(
        nearest_centroid_indices(np.nextafter(b, -np.inf), b), k
    )


def test_nearest_boundary_matches_argmin() -> None:
    c = load_codebook(4, D)
    v = np.random.default_rng(2).standard_normal(20000) / np.sqrt(D)
    np.testing.assert_array_equal(
        nearest_centroid_indices(v, load_boundaries(4, D)),
        np.argmin(np.abs(v[:, None] - c[None, :]), axis=1),
    )


# ----------------------------------------------------------------- oracle


@pytest.mark.parametrize("case", ["4bit_seed42", "4bit_seed542", "3bit_seed542"])
def test_oracle_matches_golden(golden: dict[str, Any], case: str) -> None:
    g = golden["cases"][case]
    spec = KVCodecSpec(CodecKind.POLAR, bits=g["bits"], seed=g["seed"])
    codec = PolarQuantReference(spec, D)
    x = np.array(golden["inputs"])
    idx, norms = codec.encode(x)
    np.testing.assert_array_equal(idx, np.array(g["indices"]))
    np.testing.assert_array_equal(norms[:, 0], np.array(g["norms"]))
    packed = pack_indices(idx, g["bits"])
    assert [bytes(row).hex() for row in packed] == g["packed_hex"]
    np.testing.assert_allclose(
        codec.decode(idx, norms), np.array(g["decoded"]), rtol=0, atol=1e-12
    )


def test_zero_vector_round_trips_to_zero() -> None:
    codec = PolarQuantReference(K4, D)
    idx, norms = codec.encode(np.zeros((2, D)))
    assert np.all(norms == 0)
    assert np.all(codec.decode(idx, norms) == 0)


@pytest.mark.parametrize("scale", [1e-20, 1e-3, 1.0, 1e3, 6e4])
def test_scale_invariance(scale: float) -> None:
    codec = PolarQuantReference(V4, D)
    x = np.random.default_rng(3).standard_normal((50, D))
    idx_unit, _ = codec.encode(x)
    idx_scaled, norms = codec.encode(x * scale)
    np.testing.assert_array_equal(idx_unit, idx_scaled)
    decoded = codec.decode(idx_scaled, norms)
    np.testing.assert_allclose(
        decoded / scale, codec.decode(idx_unit, norms / scale), rtol=1e-9
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input_rejected(bad: float) -> None:
    x = np.ones((1, D))
    x[0, 3] = bad
    with pytest.raises(ValueError, match="NaN or Inf"):
        PolarQuantReference(K4, D).encode(x)


def test_wrong_block_size_rejected() -> None:
    with pytest.raises(ValueError, match="block_size"):
        PolarQuantReference(K4, D).encode(np.ones((1, 64)))


def test_norm_correction_flag_only_changes_decode() -> None:
    x = np.random.default_rng(4).standard_normal((20, D))
    on = PolarQuantReference(V4, D, norm_correction=True)
    off = PolarQuantReference(V4, D, norm_correction=False)
    idx_on, norms = on.encode(x)
    idx_off, _ = off.encode(x)
    np.testing.assert_array_equal(idx_on, idx_off)
    np.testing.assert_allclose(
        np.linalg.norm(on.decode(idx_on, norms), axis=-1), norms[:, 0], rtol=1e-12
    )


def test_quality_regression_bound() -> None:
    """Unit Gaussian d=128, seed-42 FWHT, norm correction: reference mean 9.188e-3."""
    x = np.random.default_rng(123).standard_normal((20000, D))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    codec = PolarQuantReference(K4, D)
    err = np.sum((x - codec.decode(*codec.encode(x))) ** 2, axis=1)
    assert abs(err.mean() - 9.188e-3) / 9.188e-3 < 0.02


def test_dense_qr_rotation_is_orthogonal() -> None:
    codec = PolarQuantReference(K4, D, rotation=Rotation.DENSE_QR)
    q = codec.rotation.matrix()
    np.testing.assert_allclose(q @ q.T, np.eye(D), atol=1e-12)
    assert np.linalg.det(q) > 0


# ----------------------------------------------------------------- packing


@pytest.mark.parametrize("bits", list(range(1, 9)))
@pytest.mark.parametrize("n", [1, 3, 7, 8, 9, 64, 127, 128])
def test_pack_round_trip_all_patterns(bits: int, n: int) -> None:
    rng = np.random.default_rng(bits * 1000 + n)
    levels = 1 << bits
    positions = np.arange(n)
    # Row s holds (p - s) mod levels at position p, so every value appears at every position.
    rows = [((positions - s) % levels).astype(np.uint8) for s in range(levels)]
    rows.append(rng.integers(0, levels, n, dtype=np.uint8))
    idx = np.stack(rows)
    packed = pack_indices(idx, bits)
    assert packed.dtype == np.uint8
    assert packed.shape == (len(rows), packed_nbytes(n, bits))
    np.testing.assert_array_equal(unpack_indices(packed, bits, n), idx)


def test_pack_known_layouts() -> None:
    assert bytes(pack_indices(np.arange(1, 9, dtype=np.uint8), 4)).hex() == "12345678"
    three = np.array([1, 2, 3, 4, 5, 6, 7, 0], dtype=np.uint8)
    assert bytes(pack_indices(three, 3)).hex() == "29cbb8"
    assert (
        bytes(pack_indices(np.array([1, 0, 0, 0, 0, 0, 0, 0, 1], np.uint8), 1)).hex()
        == "8080"
    )


def test_pack_preserves_leading_axes() -> None:
    idx = np.random.default_rng(5).integers(0, 16, (8, 1, 5, D), dtype=np.uint8)
    packed = pack_indices(idx, 4)
    assert packed.shape == (8, 1, 5, 64)
    np.testing.assert_array_equal(packed[3, 0, 2], pack_indices(idx[3, 0, 2], 4))


def test_unpack_rejects_bad_input() -> None:
    packed = pack_indices(np.array([7, 7, 7], np.uint8), 3)
    with pytest.raises(ValueError, match="padding"):
        unpack_indices(packed | 1, 3, 3)
    unpack_indices(packed | 1, 3, 3, strict=False)
    with pytest.raises(ValueError, match="Expected"):
        unpack_indices(packed, 3, 9)
    with pytest.raises(TypeError, match="uint8"):
        unpack_indices(packed.astype(np.int16), 3, 3)
    with pytest.raises(ValueError, match="out of range"):
        pack_indices(np.array([16], np.uint8), 4)
    with pytest.raises(ValueError, match="Bit width"):
        pack_indices(np.array([0], np.uint8), 9)


def test_storage_norms_policy() -> None:
    assert to_storage_norms(np.array([0.0, 1.5, 300.25]), "float16").dtype == np.float16
    with pytest.raises(OverflowError, match="float16"):
        to_storage_norms(np.array([FLOAT16_MAX * 1.01]), "float16")
    with pytest.raises(ValueError, match="underflows"):
        to_storage_norms(np.array([0.0, 1e-6]), "float16")
    assert to_storage_norms(np.array([1e6, 1e-6]), "float32")[0] == 1e6
    with pytest.raises(ValueError, match="finite"):
        to_storage_norms(np.array([np.nan]), "float16")


# ----------------------------------------------------------------- config


def test_profiles() -> None:
    assert not get_profile("baseline_int8").enabled
    assert get_profile("k8_v3").key == BASELINE
    k4_v4 = get_profile("k4_v4")
    assert (k4_v4.key.bits, k4_v4.key.seed, k4_v4.value.seed) == (4, 42, 542)
    assert get_profile("k4_v3").value.bits == 3
    with pytest.raises(NotImplementedError, match="QJL-off"):
        get_profile("qjl_reference")
    with pytest.raises(ValueError, match="Unknown"):
        get_profile("k2_v2")


def test_config_hash_is_stable_and_sensitive() -> None:
    hashes = {name: cfg.config_hash() for name, cfg in PROFILES.items()}
    assert len(set(hashes.values())) == len(PROFILES)
    same = TurboQuantConfig("k4_v4", K4, V4)
    assert same.config_hash() == get_profile("k4_v4").config_hash()
    assert TurboQuantConfig("k4_v4", K4, V4, norm_dtype="float32").config_hash() != (
        same.config_hash()
    )
    assert TurboQuantConfig("k4_v4", K4, V4, norm_correction=False).config_hash() != (
        same.config_hash()
    )
    data = same.to_dict()
    assert data["qjl"] is False
    assert data["key"]["signs_sha256"] != data["value"]["signs_sha256"]


def test_config_rejects_unsupported_settings() -> None:
    with pytest.raises(ValueError, match="3 or 4"):
        KVCodecSpec(CodecKind.POLAR, bits=2, seed=42)
    with pytest.raises(ValueError, match="no bits"):
        KVCodecSpec(CodecKind.BASELINE, bits=4)
    with pytest.raises(ValueError, match="no bits"):
        KVCodecSpec(CodecKind.INT16, bits=4)
    with pytest.raises(ValueError, match="FWHT signs"):
        TurboQuantConfig("x", KVCodecSpec(CodecKind.POLAR, bits=4, seed=7), V4)
    with pytest.raises(ValueError, match="codebook"):
        TurboQuantConfig("x", K4, V4, block_size=64)
    with pytest.raises(ValueError, match="bit order"):
        TurboQuantConfig("x", K4, V4, bit_order="lsb_first")
    with pytest.raises(ValueError, match="format version"):
        TurboQuantConfig("x", K4, V4, format_version=2)
    assert (4, D) in CODEBOOK_HEX


# Qwen3 checkpoint config values: (layers, attention heads, kv heads, hidden, head_dim).
QWEN3_SHAPES = {
    "qwen3_0_6b": (28, 16, 8, 1024, 128),
    "qwen3_1_7b": (28, 16, 8, 2048, 128),
    "qwen3_4b": (36, 32, 8, 2560, 128),
    "qwen3_8b": (36, 32, 8, 4096, 128),
}


@pytest.mark.parametrize("model_id", sorted(QWEN3_SHAPES))
@pytest.mark.parametrize("profile", ["k4_v4", "k8_v3", "k4_v3"])
def test_qwen3_sizes_share_one_config(model_id: str, profile: str) -> None:
    layers, heads, kv_heads, hidden, head_dim = QWEN3_SHAPES[model_id]
    get_profile(profile).validate_for_model(layers, kv_heads, head_dim)
    if model_id in ("qwen3_0_6b", "qwen3_4b"):
        # head_dim is explicit in these checkpoints; hidden // heads would be wrong.
        assert hidden // heads != head_dim


def test_validate_for_model_rejects_bad_head_dim() -> None:
    cfg = get_profile("k4_v4")
    with pytest.raises(ValueError, match="power of two"):
        cfg.validate_for_model(28, 8, 100)
    with pytest.raises(ValueError, match="block_size"):
        cfg.validate_for_model(28, 8, 64)
    get_profile("baseline_int8").validate_for_model(28, 8, 100)
