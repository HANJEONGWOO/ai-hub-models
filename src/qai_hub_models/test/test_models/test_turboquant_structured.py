# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Structured K codebook, frozen identity and packed QK correctness contracts."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

from qai_hub_models.models.templates.llm.turboquant.bitplane_attention import (
    bitplane_reference,
    use_bitplane_qk,
)
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
    use_native_decoder,
    with_reference_decoder,
)
from qai_hub_models.models.templates.llm.turboquant.packing import (
    pack_indices,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.reference import (
    PolarQuantReference,
    load_boundaries,
    load_codebook,
)
from qai_hub_models.models.templates.llm.turboquant.structured import (
    bit_signs,
    centroids_from_beta,
    fit_beta,
    gap_transform,
    load_parameters,
    parameter_digest,
)
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)
from qai_hub_models.test.test_models import test_turboquant_tiled_attention as fixture


class BitplaneQK4(OpRun):
    op_domain = "turboquant"

    def _run(
        self, packed: np.ndarray, scale: np.ndarray, query: np.ndarray, beta: np.ndarray
    ) -> tuple[np.ndarray]:
        return (bitplane_reference(packed, scale, query, beta.ravel().view("<f4")),)


def test_structured_identity_and_constraints() -> None:
    data = load_parameters()
    c = centroids_from_beta(np.array(data["beta"]))
    assert np.all(np.diff(c) > 0)
    np.testing.assert_array_equal(c, -c[::-1])
    np.testing.assert_allclose(c, bit_signs() @ data["beta"], rtol=0, atol=1e-16)
    a, b = get_profile("k4s_v4_scaled"), get_profile("k4s_v4_bitplane")
    assert a.key == b.key and a.value == b.value == get_profile("k4_v4_scaled").value
    assert a.config_hash() != b.config_hash()
    assert a.to_dict()["structured_key"] == b.to_dict()["structured_key"] == data
    assert (
        get_profile("k4_v4_scaled").config_hash()
        == "d7ac74594b85ffd81086018773add722de8e6b9f1eee7339b929f8510374be6e"
    )
    with pytest.raises(ValueError, match="K-only"):
        replace(a, key=a.value, value=a.key)
    with pytest.raises(ValueError, match="strictly"):
        centroids_from_beta(np.ones(4))


def test_constrained_lloyd_decreases_scalar_objective() -> None:
    x = np.random.default_rng(81).normal(0, 1 / np.sqrt(128), 8192)
    beta, history = fit_beta(x, gap_transform() @ np.full(4, 0.01), max_iterations=30)
    assert np.all(np.diff(history) <= 1e-15)
    assert np.all(np.diff(centroids_from_beta(beta)) > 0)


def test_structured_file_revalidated_after_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qai_hub_models.models.templates.llm.turboquant import structured

    data = load_parameters()
    original = load_codebook(4, 128, "structured4_v1")
    path = tmp_path / "parameters.json"
    monkeypatch.setattr(structured, "PARAMETERS", path)
    data["beta"][0] += 0.00001
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_codebook(4, 128, "structured4_v1")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_boundaries(4, 128, "structured4_v1")
    data["centroids"] = centroids_from_beta(np.array(data["beta"])).tolist()
    c = np.array(data["centroids"])
    data["boundaries"] = ((c[:-1] + c[1:]) / 2).tolist()
    data["sha256"] = parameter_digest(data)
    path.write_text(json.dumps(data))
    changed = load_codebook(4, 128, "structured4_v1")
    np.testing.assert_array_equal(changed, c)
    assert not np.array_equal(original, changed)


@pytest.mark.parametrize("source", ["k4_v4_scaled", "k4s_v4_scaled", "k4s_v4_bitplane"])
@pytest.mark.parametrize("target", ["k4_v4_scaled", "k4s_v4_scaled", "k4s_v4_bitplane"])
def test_structured_cache_identity_and_storage(source: str, target: str) -> None:
    a = TurboQuantKVCache(get_profile(source), 1, 2, 128, 16)
    b = TurboQuantKVCache(get_profile(target), 1, 2, 128, 16)
    rng = np.random.default_rng(940)
    a.append([rng.normal(size=(2, 1, 128, 3)), rng.normal(size=(2, 1, 3, 128))])
    state = a.state_dict()
    if source == target:
        b.load_state_dict(state)
        assert b.get_seq_length() == 3
        np.testing.assert_array_equal(a.layers[0][0].packed, b.layers[0][0].packed)
    else:
        with pytest.raises(ValueError, match="different TurboQuant config"):
            b.load_state_dict(state)
    assert a.memory_report().allocated_bytes == 1 * 2 * 2 * 16 * 66


@pytest.mark.parametrize("tokens", [1, 128])
def test_structured_encoder_scale_and_value_unchanged(tokens: int) -> None:
    cfg = get_profile("k4s_v4_scaled")
    x = np.random.default_rng(62).normal(size=(1, 2, tokens, 128)).astype(np.float32)
    x[0, 0, 0] = 0
    model = build_encode_model(cfg, cfg.key, 2, tokens)
    packed, scale = ReferenceEvaluator(model).run(None, {model.graph.input[0].name: x})
    codec = PolarQuantReference(cfg.key, precomputed_norm=True)
    indices, expected_scale = codec.encode(x)
    np.testing.assert_array_equal(unpack_indices(packed, 4, 128), indices)
    np.testing.assert_array_equal(pack_indices(indices, 4), packed)
    np.testing.assert_allclose(scale, expected_scale, rtol=1e-5, atol=1e-6)
    baseline = get_profile("k4_v4_scaled")
    # Same V graph, not just a close reconstruction.
    assert (
        build_encode_model(cfg, cfg.value, 2, tokens).SerializeToString()
        == build_encode_model(baseline, baseline.value, 2, tokens).SerializeToString()
    )


def test_structured_tree_fp16_domain_and_boundaries() -> None:
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    from qai_hub_models.models.templates.llm.turboquant.export import (
        Subgraph,
        _finish,
        _scalar_index_tree,
    )

    cfg = get_profile("k4s_v4_scaled")
    boundaries = load_boundaries(4, 128, cfg.key.codebook).astype(np.float16)
    x = np.arange(65536, dtype=np.uint16).view(np.float16)
    x = np.sort(x[np.isfinite(x)]).reshape(1, 1, 1, -1)
    sg = Subgraph()
    output = _scalar_index_tree(sg, "y", 4, 128, "test_", cfg.key.codebook)
    for tensor in sg.initializers.values():
        if tensor.data_type == TensorProto.FLOAT:
            tensor.CopyFrom(
                numpy_helper.from_array(
                    numpy_helper.to_array(tensor).astype(np.float16), tensor.name
                )
            )
    for node in sg.nodes:
        for attr in node.attribute:
            if (
                node.op_type == "Cast"
                and attr.name == "to"
                and attr.i == TensorProto.FLOAT
            ):
                attr.i = TensorProto.FLOAT16
    model = _finish(
        sg,
        "test",
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, x.shape)],
        [helper.make_tensor_value_info(output, TensorProto.INT32, x.shape)],
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    index = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {"y": x})[0]
    np.testing.assert_array_equal(index, np.searchsorted(boundaries, x, side="left"))
    assert index.min() == 0 and index.max() == 15
    assert np.all(index[x == 0] == 7)


@pytest.mark.parametrize("seq", [1, 3, 128])
@pytest.mark.parametrize("divisor", [None, float(np.sqrt(128))])
def test_bitplane_gqa_current_past_attention(
    seq: int, divisor: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = max(35, seq + 17)
    monkeypatch.setattr(fixture, "CONTEXT", context)
    model, enc = fixture.attention_part(seq, divisor)
    cfg = get_profile("k4s_v4_bitplane")
    surgery = apply_kv_profile(model, enc, cfg, seq, context)
    lut = quantize_current_attention(
        use_native_decoder(tile_kv_attention(surgery, cfg, 7, rotated=True), cfg), cfg
    )
    bp = use_bitplane_qk(lut, cfg)
    assert bp.encodings["param_encodings"] == lut.encodings["param_encodings"]
    assert not any(
        n.op_type == "Decode4" and "key" in n.name for n in bp.model.graph.node
    )
    assert not any(
        n.name.startswith("tq_key_") and n.name.endswith(("restored", "native_fp16"))
        for n in bp.model.graph.node
    )

    # V/AV and mask/softmax operations are identical.
    def retained(m: onnx.ModelProto) -> list[bytes]:
        return sorted(
            n.SerializeToString()
            for n in m.graph.node
            if n.op_type == "Softmax"
            or n.name.startswith("tq_value_")
            or (n.op_type == "MatMul" and n.name.endswith(("_partial", "_out")))
        )

    assert retained(bp.model) == retained(lut.model)
    rng = np.random.default_rng(73)
    feeds = fixture.feeds(seq, min(context - seq, 13))
    for kind in ("key", "value"):
        feeds.pop(f"tq_{kind}_0_norm_in")
        codec = PolarQuantReference(getattr(cfg, kind), precomputed_norm=True)
        vectors = rng.normal(size=(2, 1, context - seq, 128))
        vectors[..., :2, :] = 0
        indices, scale = codec.encode(vectors)
        feeds[f"tq_{kind}_0_packed_in"] = pack_indices(indices, 4)
        feeds[f"tq_{kind}_0_scale_in"] = scale.astype(np.float16).astype(np.float32)
    outputs = []
    for result in (lut, bp):
        oracle = with_reference_decoder(result.model)
        onnx.checker.check_model(oracle)
        outputs.append(
            ReferenceEvaluator(oracle, new_ops=[BitplaneQK4]).run(None, feeds)
        )
    for info, expected, actual in zip(bp.model.graph.output, *outputs, strict=True):
        if info.name.startswith("tq_"):
            np.testing.assert_array_equal(expected, actual)
        else:
            np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.02)


def test_bitplane_real_arithmetic_equivalence() -> None:
    beta = np.asarray(load_parameters()["beta"])
    c = load_codebook(4, 128, "structured4_v1")
    rng = np.random.default_rng(291)
    indices = rng.integers(0, 16, (3, 128))
    query = rng.normal(size=(4, 128))
    expected = query @ c[indices].T
    actual = sum(beta[m] * (query @ (2 * ((indices >> m) & 1) - 1).T) for m in range(4))
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-14)
