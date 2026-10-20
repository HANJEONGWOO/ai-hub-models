# SPDX-License-Identifier: BSD-3-Clause
"""Uniform's opt-in arithmetic ABI, exact FP16 round boundaries and old hashes."""

from dataclasses import replace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.current_attention import (
    quantize_current_attention,
)
from qai_hub_models.models.templates.llm.turboquant.export import build_encode_model
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)
from qai_hub_models.models.templates.llm.turboquant.native_decoder import (
    UNIFORM_COEFFICIENTS,
    UNIFORM_OP,
    use_native_decoder,
    with_reference_decoder,
)
from qai_hub_models.models.templates.llm.turboquant.numerics import (
    FLOAT32_GRAPH,
    compare_encode,
)
from qai_hub_models.models.templates.llm.turboquant.reference import PolarQuantReference
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)
from qai_hub_models.models.templates.llm.turboquant.uniform_k8 import (
    boundaries,
    codebook,
    coefficients,
    indices,
)
from qai_hub_models.test.test_models import test_turboquant_tiled_attention as fixture


def test_uniform_constants_and_index() -> None:
    c, (offset, delta) = codebook(), coefficients()
    assert len(c) == 256 and offset == delta / 2
    assert np.array_equal(c, -c[::-1]) and np.all(np.diff(c) > 0)
    x = np.r_[-1e6, boundaries() - 1e-9, boundaries() + 1e-9, 0, 1e6]
    x.sort()
    idx = indices(x)
    expected_mag = np.clip(np.floor(np.abs(x) / delta), 0, 127).astype(int)
    np.testing.assert_array_equal(
        idx, np.where(x >= 0, 128 + expected_mag, 127 - expected_mag)
    )
    assert idx.min() == 0 and idx.max() == 255 and np.all(np.diff(idx.astype(int)) >= 0)
    assert indices(np.array([0.0, -0.0])).tolist() == [128, 128]
    nonzero = np.geomspace(1e-8, 1e3, 4096)
    np.testing.assert_array_equal(indices(-nonzero), 255 - indices(nonzero))


def test_profiles_budget_cache_and_legacy_hashes() -> None:
    old = get_profile("mixed_k6_v2_uniform_structured_k8")
    new = get_profile("mixed_k6_v2_uniform_uniform_k8")
    assert (
        old.config_hash()
        == "b68d0d5534b516363c892ecf65bd98f44789969d4e07cc31de73e7bdc3d11ed9"
    )
    assert (
        get_profile("k4_v4_scaled").config_hash()
        == "d7ac74594b85ffd81086018773add722de8e6b9f1eee7339b929f8510374be6e"
    )
    for layer in range(28):
        assert new.codec_for("value", layer) == old.codec_for("value", layer)
        if layer % 2 == 0:
            assert new.codec_for("key", layer) == old.codec_for("key", layer)
        else:
            assert new.codec_for("key", layer).quantizer == "uniform_k8_gaussian_v1"
    caches = [TurboQuantKVCache(c, 28, 8, 128, 1024) for c in (old, new)]
    assert all(c.memory_report().allocated_bytes == 30277632 for c in caches)
    with pytest.raises(ValueError, match="different TurboQuant config"):
        caches[1].load_state_dict(caches[0].state_dict())
    for kwargs in ({"structured_k8": True}, {"converged_k8": True}, {"qjl": True}):
        with pytest.raises(ValueError, match="Uniform K8 requires"):
            replace(new, **kwargs)
    probe = get_profile("uniform_k8_v4_scaled")
    with pytest.raises(ValueError, match="does not match"):
        replace(probe, uniform_k8=False, structured_k8=True)


@pytest.mark.parametrize("tokens", [1, 3, 128])
def test_encoder_cpu_reference(tokens: int) -> None:
    c = get_profile("uniform_k8_v4_scaled")
    model = build_encode_model(c, c.key, 8, tokens)
    assert not any(
        "tq_scalar_b8_" in t.name or "tq_centroid_b8_" in t.name
        for t in model.graph.initializer
    )
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    x = (
        np.random.default_rng(tokens)
        .normal(size=(1, 8, tokens, 128))
        .astype(np.float32)
    )
    packed, scale = session.run(None, {"x": x})
    reference = PolarQuantReference(c.key, precomputed_norm=True)
    assert compare_encode(reference, x, packed, scale, FLOAT32_GRAPH)["passed"]


@pytest.mark.parametrize("tokens", [1, 3, 256, 1023])
def test_native_arithmetic_fp16_exact(tokens: int) -> None:
    packed = np.resize(np.arange(256, dtype=np.uint8), (8, 1, tokens, 128))
    edges = np.array(
        [
            0,
            0x8000,
            1,
            0x3FF,
            0x400,
            0x401,
            0x3BFF,
            0x3C00,
            0x3C01,
            0x4000,
            0x7BFF,
            0x8001,
            0xBC00,
            0xFBFF,
        ],
        np.uint16,
    ).view(np.float16)
    scale = np.resize(np.repeat(edges, 2), (8, 1, tokens, 1))
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    UNIFORM_OP,
                    ["packed", "scale", "coefficients"],
                    ["decoded"],
                    domain="turboquant",
                )
            ],
            "test",
            [
                helper.make_tensor_value_info(
                    "packed", TensorProto.UINT8, list(packed.shape)
                ),
                helper.make_tensor_value_info(
                    "scale", TensorProto.FLOAT16, list(scale.shape)
                ),
            ],
            [
                helper.make_tensor_value_info(
                    "decoded", TensorProto.FLOAT16, list(packed.shape)
                )
            ],
            [
                numpy_helper.from_array(
                    coefficients().astype(np.float16).reshape(1, 1, 1, 2),
                    "coefficients",
                )
            ],
        ),
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("turboquant", 1),
        ],
        ir_version=9,
    )
    model = with_reference_decoder(model)
    onnx.checker.check_model(model, full_check=True)
    output = ReferenceEvaluator(model).run(None, {"packed": packed, "scale": scale})[0]
    expected = (
        codebook()[packed].astype(np.float32) * scale.astype(np.float32)
    ).astype(np.float16)
    np.testing.assert_array_equal(output.view(np.uint16), expected.view(np.uint16))


@pytest.mark.parametrize("seq", [1, 3, 128])
def test_current_past_native_routing(seq: int, monkeypatch: pytest.MonkeyPatch) -> None:
    context = max(35, seq + 17)
    monkeypatch.setattr(fixture, "CONTEXT", context)
    c = get_profile("uniform_k8_v4_scaled")
    model, enc = fixture.attention_part(seq)
    result = quantize_current_attention(
        use_native_decoder(
            tile_kv_attention(
                apply_kv_profile(model, enc, c, seq, context), c, 7, rotated=True
            ),
            c,
        ),
        c,
    )
    ops = [n for n in result.model.graph.node if n.domain == "turboquant"]
    assert any(n.op_type == UNIFORM_OP and "current" in n.name for n in ops)
    assert any(n.op_type == UNIFORM_OP and "tile" in n.name for n in ops)
    assert all(
        n.input[2] == UNIFORM_COEFFICIENTS for n in ops if n.op_type == UNIFORM_OP
    )
    assert all(n.op_type == "Decode4" for n in ops if "value" in n.name)
    onnx.checker.check_model(with_reference_decoder(result.model), full_check=True)
