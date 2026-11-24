# SPDX-License-Identifier: BSD-3-Clause
"""Structured K8 is opt-in; legacy LM and all V paths remain unchanged."""

from dataclasses import replace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.export import (
    Subgraph,
    _structured_arithmetic,
    build_encode_model,
)
from qai_hub_models.models.templates.llm.turboquant.reference import PolarQuantReference
from qai_hub_models.models.templates.llm.turboquant.structured_k8 import (
    codebook,
    coefficients,
    indices,
    region_boundary,
)


def test_frozen_codebook_and_gaussian_parameters() -> None:
    c = codebook()
    assert c.shape == (256,) and np.all(np.diff(c) > 0)
    np.testing.assert_array_equal(c, -c[::-1])
    assert coefficients()[1] < coefficients()[3]
    assert c[191] < region_boundary() < c[192]


def test_arithmetic_boundaries_range_symmetry_and_saturation() -> None:
    a, h, b, t = coefficients()
    transitions = np.r_[
        a + (np.arange(63) + 0.5) * h, region_boundary(), b + (np.arange(63) + 0.5) * t
    ]
    values = np.sort(np.r_[-transitions, transitions, 0, -1e6, 1e6])
    codes = indices(values)
    assert codes.min() == 0 and codes.max() == 255
    assert np.all(np.diff(codes.astype(int)) >= 0)
    assert indices(np.array([0.0, -0.0])).tolist() == [128, 128]
    positive = transitions + 1e-8
    np.testing.assert_array_equal(indices(-positive), 255 - indices(positive))
    for i, edge in enumerate(transitions):
        assert indices(np.array([edge - 1e-8, edge + 1e-8])).tolist() == [
            128 + i,
            129 + i,
        ]


def test_arithmetic_centroid_exactly_matches_native_lut() -> None:
    sg = Subgraph()
    out = _structured_arithmetic(sg, "index", "test_", centroid=True)
    graph = helper.make_graph(
        sg.nodes,
        "centroids",
        [helper.make_tensor_value_info("index", TensorProto.FLOAT, [256])],
        [helper.make_tensor_value_info(out, TensorProto.FLOAT, [256])],
        list(sg.initializers.values()),
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    result = session.run(None, {"index": np.arange(256, dtype=np.float32)})[0]
    np.testing.assert_array_equal(result, codebook())


def test_mixed_plan_budget_legacy_hash_and_cache_rejection() -> None:
    lm = get_profile("mixed_k6_v2_uniform")
    structured = get_profile("mixed_k6_v2_uniform_structured_k8")
    assert (
        lm.config_hash()
        == "0c5924e389479226063bb62e5e812dcf406d25695ac9dcb28a0405bb567f1da4"
    )
    assert (
        get_profile("k4_v4_scaled").config_hash()
        == "d7ac74594b85ffd81086018773add722de8e6b9f1eee7339b929f8510374be6e"
    )
    assert structured.layer_bits == lm.layer_bits
    for layer in range(28):
        assert structured.codec_for("value", layer) == lm.codec_for("value", layer)
        assert (structured.codec_for("key", layer).quantizer != "lloyd_max") == bool(
            layer % 2
        )
        if not layer % 2:
            a, b = (
                build_encode_model(c.for_layer(layer), c.codec_for("key", layer), 8, 1)
                for c in (lm, structured)
            )
            assert a.SerializeToString() == b.SerializeToString()
    a, b = (TurboQuantKVCache(c, 28, 8, 128, 1024) for c in (lm, structured))
    assert (
        a.memory_report().allocated_bytes
        == b.memory_report().allocated_bytes
        == 30277632
    )
    with pytest.raises(ValueError, match="different TurboQuant config"):
        b.load_state_dict(a.state_dict())


def test_structured_graph_small_no_tree_no_repacking() -> None:
    config = get_profile("structured_k8_v4_scaled")
    graph = build_encode_model(config, config.key, 8, 128).graph
    assert len(graph.node) < 100
    assert not any(n.op_type in ("Slice", "Concat", "Gather") for n in graph.node)
    assert not any("level" in n.name for n in graph.node)
    assert sum(n.op_type == "MatMul" for n in graph.node) == 1
    x = np.random.default_rng(53).normal(size=(1, 8, 128, 128))
    ref = PolarQuantReference(config.key, precomputed_norm=True)
    packed, scale = ref.encode(x)
    assert np.isfinite(ref.decode(packed, scale)).all()
    with pytest.raises(ValueError, match="K-only"):
        replace(config, value=config.key)


def test_mixed_current_past_packing_and_reset() -> None:
    config = get_profile("mixed_k6_v2_uniform_structured_k8")
    caches = [TurboQuantKVCache(config, 28, 1, 128, 5) for _ in range(2)]
    rng = np.random.default_rng(64)
    kv = [
        rng.normal(size=shape).astype(np.float32)
        for _ in range(28)
        for shape in ((1, 1, 128, 5), (1, 1, 5, 128))
    ]
    caches[0].append(kv)
    for i in range(5):
        caches[1].append(
            [
                v[..., i : i + 1] if j % 2 == 0 else v[:, :, i : i + 1]
                for j, v in enumerate(kv)
            ]
        )
    for a, b in zip(caches[0].layers, caches[1].layers, strict=True):
        for key, value in zip(a, b, strict=True):
            for name, values in key.arrays().items():
                np.testing.assert_array_equal(values, value.arrays()[name])
    saved = caches[0].state_dict()
    caches[0].reset()
    assert caches[0].get_seq_length() == 0
    assert all(
        not v.any()
        for pair in caches[0].layers
        for store in pair
        for v in store.arrays().values()
    )
    caches[0].load_state_dict(saved)
    assert caches[0].get_seq_length() == 5


def test_numeric_gate_uses_arithmetic_rule_without_relaxing_tolerance() -> None:
    from qai_hub_models.models.templates.llm.turboquant.numerics import (
        FLOAT32_GRAPH,
        compare_encode,
    )

    config = get_profile("structured_k8_v4_scaled")
    model = build_encode_model(config, config.key, 8, 3)
    x = np.random.default_rng(90).normal(size=(1, 8, 3, 128)).astype(np.float32)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    packed, scale = session.run(None, {"x": x})
    oracle = PolarQuantReference(config.key, precomputed_norm=True)
    assert compare_encode(oracle, x, packed, scale, FLOAT32_GRAPH)["passed"]
