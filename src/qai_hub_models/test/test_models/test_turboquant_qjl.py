# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""K3+1 orthogonal QJL, native nibble ABI and independent full-attention oracle."""

from dataclasses import replace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import Rotation, get_profile
from qai_hub_models.models.templates.llm.turboquant.export import build_encode_model
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)
from qai_hub_models.models.templates.llm.turboquant.native_decoder import (
    use_native_decoder,
    with_reference_decoder,
)
from qai_hub_models.models.templates.llm.turboquant.packing import (
    pack_indices,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.qjl import (
    QJLKeyReference,
    add_qjl_attention,
    build_qjl_encode_model,
    dequantize_residual,
    projection,
    quantize_residual,
)
from qai_hub_models.models.templates.llm.turboquant.reference import PolarQuantReference
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)
from qai_hub_models.test.test_models.test_turboquant_rotated_attention import run
from qai_hub_models.test.test_models.test_turboquant_tiled_attention import (
    CONTEXT,
    GROUPS,
    HEADS,
    D,
    attention_part,
    feeds,
)

CONFIG = get_profile("k3qjl_v4_scaled")


def test_projection_signs_coefficient() -> None:
    s = projection(D, 1042)
    np.testing.assert_allclose(s @ s.T, np.eye(D), atol=2e-15)
    r = np.random.default_rng(3).normal(size=(13, D))
    r[0] = 0
    signs, norms = quantize_residual(r)
    np.testing.assert_array_equal(signs, np.where(r @ s.T >= 0, 1, -1))
    assert np.all(signs[0] == 1)
    expected = (signs @ s) * norms * (np.sqrt(np.pi / 2) / np.sqrt(D))
    np.testing.assert_allclose(dequantize_residual(signs, norms), expected, atol=1e-15)
    # Exactly one bit per coordinate, without silently applying MMSE shrinkage.
    np.testing.assert_allclose(
        np.linalg.norm(expected, axis=-1), np.sqrt(np.pi / 2) * norms[:, 0], atol=1e-14
    )


@pytest.mark.parametrize("seq", [1, 3])
@pytest.mark.parametrize("tile", [7, 256])
@pytest.mark.parametrize("valid", [0, 13, 32])
def test_qjl_tiled_attention(seq: int, tile: int, valid: int) -> None:
    model, encodings = attention_part(seq)
    result = use_native_decoder(
        tile_kv_attention(
            apply_kv_profile(model, encodings, CONFIG, seq, CONTEXT),
            CONFIG,
            tile,
            rotated=True,
        ),
        CONFIG,
    )
    original = result.model.SerializeToString()
    corrected = add_qjl_attention(result, CONFIG)
    assert result.model.SerializeToString() == original
    oracle = with_reference_decoder(corrected.model)
    # ORT's CPU FP16 emulation otherwise elides internal product rounding.
    # Expose Native outputs so the independent oracle exercises the DSP ABI.
    native_outputs = {n.output[0] for n in oracle.graph.node if n.op_type == "Decode4"}
    oracle.graph.output.extend(
        v for v in oracle.graph.value_info if v.name in native_outputs
    )
    for info in list(oracle.graph.value_info):
        if info.name.endswith("_native_fp16"):
            oracle.graph.output.append(
                helper.make_tensor_value_info(
                    info.name.removesuffix("native_fp16") + "restored",
                    TensorProto.FLOAT,
                    [dim.dim_value for dim in info.type.tensor_type.shape.dim],
                )
            )
    onnx.checker.check_model(oracle, full_check=True)
    assert len({n.name for n in oracle.graph.node}) == len(oracle.graph.node)
    data = feeds(seq, min(valid, CONTEXT - seq))
    rng = np.random.default_rng(5)
    reference = QJLKeyReference(CONFIG)
    past = rng.normal(size=(HEADS, 1, CONTEXT - seq, D))
    past[:, :, : CONTEXT - seq - min(valid, CONTEXT - seq)] = 0
    packed, scale, qscale = reference.encode(past)
    data["tq_key_0_packed_in"] = packed
    data["tq_key_0_scale_in"] = scale.astype(np.float32)
    data["tq_key_0_qjlscale_in"] = qscale.astype(np.float32)
    del data["tq_key_0_norm_in"]
    value_ref = PolarQuantReference(CONFIG.value, D, precomputed_norm=True)
    vpast = rng.normal(size=past.shape)
    vi, vs = value_ref.encode(vpast)
    vs = vs.astype(np.float16)
    data["tq_value_0_packed_in"] = pack_indices(vi, 4)
    data["tq_value_0_scale_in"] = vs.astype(np.float32)
    del data["tq_value_0_norm_in"]
    actual = run(oracle, data)
    k = reference.decode(packed, scale, qscale)
    v = value_ref.rotation.inverse(
        (value_ref.centroids[vi].astype(np.float16) * vs).astype(np.float16)
    )
    for h in range(HEADS):
        keys = np.concatenate((k[h : h + 1], data["new_key"][h : h + 1]), axis=2)
        values = np.concatenate((v[h : h + 1], data["new_value"][h : h + 1]), axis=2)
        for g in range(GROUPS):
            scores = data[f"h{h}g{g}_q"] @ keys.swapaxes(-1, -2) + data["mask"]
            probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            np.testing.assert_allclose(
                actual[f"h{h}g{g}_out"], probs @ values, rtol=2e-4, atol=2e-6
            )
    ep, es, eqs = reference.encode(data["new_key"])
    np.testing.assert_array_equal(actual["tq_key_0_packed_out"], ep)
    np.testing.assert_allclose(
        actual["tq_key_0_scale_out"].astype(np.float16), es, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        actual["tq_key_0_qjlscale_out"].astype(np.float16), eqs, rtol=0, atol=0
    )
    assert ep.shape[-1] == D // 2
    assert es.nbytes + eqs.nbytes == HEADS * seq * 4
    # V encoder is unchanged, including packed bytes and effective scale.
    vi_new, vs_new = value_ref.encode(data["new_value"])
    np.testing.assert_array_equal(
        unpack_indices(actual["tq_value_0_packed_out"], 4, D), vi_new
    )
    np.testing.assert_allclose(actual["tq_value_0_scale_out"], vs_new, rtol=3e-6)


def test_qjl_profile_isolation() -> None:
    assert (
        get_profile("k4_v4_scaled").config_hash()
        == "a1bd2907c7f1352c7e3472d7022dcb4c11cf6cf85b361f5316a0d155697c3c62"
    )
    assert not get_profile("k4_v4_scaled").qjl
    assert CONFIG.format_version == 3
    assert CONFIG.key.bits + 1 == CONFIG.value.bits == 4


@pytest.mark.parametrize("heads", [1, 8])
def test_qjl_encoder_zero_and_storage(heads: int) -> None:
    model = with_reference_decoder(build_qjl_encode_model(CONFIG, 3, heads))
    x = np.random.default_rng(36).normal(size=(heads, 1, 3, D)).astype(np.float32)
    x[:, :, 0] = 0
    # ReferenceEvaluator preserves FP16 Cast/Mul rounding. ORT's CPU cast
    # transformer can bypass the encoder-internal scale rounding even with
    # optimization disabled and its FP16 output exposed.
    actual = dict(
        zip(
            (v.name for v in model.graph.output),
            ReferenceEvaluator(model).run(None, {"tq_key_0_present_tokens_last": x}),
            strict=True,
        )
    )
    packed, scale, qscale = QJLKeyReference(CONFIG).encode(x)
    np.testing.assert_array_equal(actual["tq_key_0_packed_out"], packed)
    np.testing.assert_array_equal(
        actual["tq_key_0_scale_out"].astype(np.float16), scale
    )
    np.testing.assert_array_equal(
        actual["tq_key_0_qjlscale_out"].astype(np.float16), qscale
    )
    np.testing.assert_array_equal(
        QJLKeyReference(CONFIG).decode(packed, scale, qscale)[:, :, 0], 0
    )
    assert np.all(unpack_indices(packed, 4, D)[:, :, 0] >= 8)


def test_qjl_rejects_incompatible_paths() -> None:
    for changes in (
        {"rotation": Rotation.FWHT},
        {"norm_correction": False},
        {"norm_dtype": "float32"},
        {"format_version": 2},
    ):
        with pytest.raises(ValueError, match=r"QJL requires|Config format"):
            replace(CONFIG, **changes)
    with pytest.raises(NotImplementedError, match=r"QJLKeyReference"):
        TurboQuantKVCache(CONFIG, 1, HEADS, D, CONTEXT)
    with pytest.raises(ValueError, match=r"combined K3\+1"):
        build_encode_model(CONFIG, CONFIG.key, HEADS, 1)
    model, encodings = attention_part(1)
    result = use_native_decoder(
        tile_kv_attention(
            apply_kv_profile(model, encodings, CONFIG, 1, CONTEXT),
            CONFIG,
            7,
            rotated=True,
        ),
        CONFIG,
    )
    corrected = add_qjl_attention(result, CONFIG)
    with pytest.raises(ValueError, match=r"applied once"):
        add_qjl_attention(corrected, CONFIG)
    for node in result.model.graph.node:
        if node.output[0] == "tq_attn_0_tile0_head0_q0_score":
            node.output[0] = "unrecognized_score_name"
            break
    with pytest.raises(ValueError, match=r"Incomplete QJL"):
        add_qjl_attention(result, CONFIG)
