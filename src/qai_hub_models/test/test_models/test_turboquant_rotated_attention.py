# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Rotated attention, effective scales and legacy format isolation."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import Rotation, get_profile
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)
from qai_hub_models.models.templates.llm.turboquant.numerics import (
    HTP_FP16,
    compare_decode,
    compare_encode,
)
from qai_hub_models.models.templates.llm.turboquant.packing import (
    pack_indices,
    to_storage_norms,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.reference import (
    PolarQuantReference,
    load_codebook,
)
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)
from qai_hub_models.test.test_models.test_turboquant_tiled_attention import (
    CONTEXT,
    HEADS,
    D,
    attention_part,
    feeds,
)

CONFIG = get_profile("k4_v4_scaled")


def test_scaled_numeric_checker_preserves_tolerances() -> None:
    codec = PolarQuantReference(CONFIG.key, D, precomputed_norm=True)
    values = np.random.default_rng(92).normal(size=(HEADS, 1, 5, D))
    indices, scales = codec.encode(values)
    packed = pack_indices(indices, 4)
    good = compare_encode(codec, values, packed, scales, HTP_FP16)
    assert good["passed"] and good["stored_scalar"] == "effective_scale"
    assert not compare_encode(codec, values, packed, scales * 1.003, HTP_FP16)["passed"]
    decoded = codec.decode(indices, scales)
    assert compare_decode(codec, packed, scales, decoded, HTP_FP16)["passed"]
    assert not compare_decode(codec, packed, scales, decoded * 1.01, HTP_FP16)["passed"]


@pytest.mark.parametrize("rotation", list(Rotation))
def test_effective_scale_not_raw_norm_must_fit_storage(rotation: Rotation) -> None:
    codec = PolarQuantReference(CONFIG.key, D, rotation, precomputed_norm=True)
    # Construct the adversarial direction in rotated coordinates so the
    # overflow test does not depend on WHT spreading a one-hot vector evenly.
    values = codec.rotation.inverse(np.ones((1, D)) / np.sqrt(D)) * 65000
    _, scales = codec.encode(values)
    assert np.linalg.norm(values) < np.finfo(np.float16).max
    assert scales.max() > np.finfo(np.float16).max
    with pytest.raises(OverflowError, match="float16 range"):
        to_storage_norms(scales, "float16")


def run(model: onnx.ModelProto, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return dict(
        zip(
            (o.name for o in session.get_outputs()),
            session.run(None, data),
            strict=True,
        )
    )


@pytest.mark.parametrize("seq", [1, 3])
@pytest.mark.parametrize("tile", [7, 16, 256])
@pytest.mark.parametrize("valid", [0, 13, 32])
def test_rotated_attention_and_scale_cache(seq: int, tile: int, valid: int) -> None:
    model, encodings = attention_part(seq)
    legacy = apply_kv_profile(model, encodings, get_profile("k4_v4"), seq, CONTEXT)
    scaled = apply_kv_profile(model, encodings, CONFIG, seq, CONTEXT)
    transformed = tile_kv_attention(scaled, CONFIG, tile, rotated=True)
    onnx.checker.check_model(transformed.model, full_check=True)
    data = feeds(seq, min(valid, CONTEXT - seq))
    expected = run(legacy.model, data)
    for kind in ("key", "value"):
        idx = unpack_indices(data[f"tq_{kind}_0_packed_in"], 4, D)
        length = np.linalg.norm(load_codebook(4, D)[idx], axis=-1, keepdims=True)
        data[f"tq_{kind}_0_scale_in"] = (
            data.pop(f"tq_{kind}_0_norm_in") / length
        ).astype(np.float32)
    actual = run(transformed.model, data)
    normal = run(scaled.model, data)
    for name, value in actual.items():
        np.testing.assert_allclose(value, normal[name], rtol=3e-5, atol=3e-6)
        if name in expected:
            np.testing.assert_allclose(value, expected[name], rtol=3e-5, atol=3e-6)
    for kind in ("key", "value"):
        reference = PolarQuantReference(getattr(CONFIG, kind), D, precomputed_norm=True)
        _, scale = reference.encode(data[f"new_{kind}"])
        np.testing.assert_allclose(
            actual[f"tq_{kind}_0_scale_out"], scale, rtol=3e-5, atol=3e-6
        )
    for node in transformed.model.graph.node:
        if "_dec_" in node.name:
            assert node.op_type not in ("MatMul", "ReduceSum", "Sqrt", "Div")
    acts = {e["name"] for e in transformed.encodings["activation_encodings"]}
    assert not any(
        n.endswith(("_rotated", "_key_cat", "_value_cat", "_partial")) for n in acts
    )
    assert transformed.attention_tiles[0]["strategy"] == "rotated_precomputed_scale"


def test_effective_scale_reference_and_format_isolation() -> None:
    rng = np.random.default_rng(91)
    values = rng.normal(size=(HEADS, 1, 7, D))
    values[0, 0, 0] = 0
    for kind in ("key", "value"):
        spec = getattr(CONFIG, kind)
        old = PolarQuantReference(spec, D)
        new = PolarQuantReference(spec, D, precomputed_norm=True)
        a, norm = old.encode(values)
        b, scale = new.encode(values)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(
            old.decode(a, norm), new.decode(b, scale), atol=1e-14
        )
        assert scale[0, 0, 0, 0] == 0
    old_cache = TurboQuantKVCache(get_profile("k4_v4"), 1, HEADS, D, CONTEXT)
    new_cache = TurboQuantKVCache(CONFIG, 1, HEADS, D, CONTEXT)
    assert old_cache.state_dict()["format_version"] == 1
    assert new_cache.state_dict()["format_version"] == 2
    with pytest.raises(ValueError, match="format"):
        new_cache.load_state_dict(old_cache.state_dict())
    with pytest.raises(ValueError, match="format"):
        old_cache.load_state_dict(new_cache.state_dict())


def test_rotated_requires_effective_scale() -> None:
    model, encodings = attention_part(1)
    config = get_profile("k4_v4")
    with pytest.raises(ValueError, match="precomputed-scale"):
        tile_kv_attention(
            apply_kv_profile(model, encodings, config, 1, CONTEXT),
            config,
            7,
            rotated=True,
        )


@pytest.mark.parametrize("context", [8, 17, 35])
def test_bucket_padding_preserves_attention(context: int) -> None:
    model, encodings = attention_part(1)
    data = feeds(1, 3)
    for kind in ("key", "value"):
        idx = unpack_indices(data[f"tq_{kind}_0_packed_in"], 4, D)
        length = np.linalg.norm(load_codebook(4, D)[idx], axis=-1, keepdims=True)
        data[f"tq_{kind}_0_scale_in"] = (
            data.pop(f"tq_{kind}_0_norm_in") / length
        ).astype(np.float32)
    full = tile_kv_attention(
        apply_kv_profile(model, encodings, CONFIG, 1, CONTEXT), CONFIG, 7, rotated=True
    )
    expected = run(full.model, data)
    for value in model.graph.input:
        if value.name == "mask":
            value.type.tensor_type.shape.dim[3].dim_value = context
    bucket = tile_kv_attention(
        apply_kv_profile(model, encodings, CONFIG, 1, context), CONFIG, 7, rotated=True
    )
    for name in data:
        if name.endswith(("_packed_in", "_scale_in")):
            data[name] = data[name][:, :, -(context - 1) :]
    data["mask"] = data["mask"][..., -context:]
    for name, value in run(bucket.model, data).items():
        np.testing.assert_allclose(value, expected[name], rtol=3e-5, atol=3e-6)


def test_scaled_cache_append_reset_roundtrip() -> None:
    cache = TurboQuantKVCache(CONFIG, 1, HEADS, D, CONTEXT)
    rng = np.random.default_rng(12)
    keys = rng.normal(size=(HEADS, 1, D, 7)).astype(np.float32)
    values = rng.normal(size=(HEADS, 1, 7, D)).astype(np.float32)
    cache.append([keys[..., :3], values[:, :, :3]])
    first = cache.layers[0][0].arrays()["norms"][:, :, :3].copy()
    cache.append([keys[..., 3:], values[:, :, 3:]])
    np.testing.assert_array_equal(first, cache.layers[0][0].arrays()["norms"][:, :, :3])
    copy = TurboQuantKVCache(CONFIG, 1, HEADS, D, CONTEXT)
    copy.load_state_dict(cache.state_dict())
    for actual, expected in zip(copy.layer_float(0), cache.layer_float(0), strict=True):
        np.testing.assert_array_equal(actual, expected)
    oracle = PolarQuantReference(CONFIG.value, D, precomputed_norm=True)
    np.testing.assert_allclose(
        cache.layer_float(0)[1],
        oracle.decode(*oracle.encode(values)),
        atol=0.002,
        rtol=0.002,
    )
    copy.reset()
    assert copy.get_seq_length() == 0
    assert all(
        not a.any()
        for pair in copy.layers
        for store in pair
        for a in store.arrays().values()
    )
