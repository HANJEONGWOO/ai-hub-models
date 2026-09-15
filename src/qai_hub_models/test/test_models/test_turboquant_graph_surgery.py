# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""TurboQuant codec insertion into a synthetic delta-KV part, checked with onnxruntime."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import apply_kv_codec
from qai_hub_models.models.templates.llm.turboquant.numerics import (
    FLOAT32_GRAPH,
    compare_decode,
    compare_encode,
)
from qai_hub_models.models.templates.llm.turboquant.packing import pack_indices
from qai_hub_models.models.templates.llm.turboquant.reference import (
    PolarQuantReference,
)

HEADS, D, SEQ, CTX = 2, 128, 3, 8
PAST = CTX - SEQ


def synthetic_part() -> tuple[onnx.ModelProto, dict[str, Any]]:
    """Attention-like consumer of past KV plus new-token present outputs, hub layout."""
    f = TensorProto.FLOAT
    graph = helper.make_graph(
        [
            helper.make_node("Concat", ["past_key_0_in", "new_k"], ["attn_k"], axis=3),
            helper.make_node(
                "Concat", ["past_value_0_in", "new_v"], ["attn_v"], axis=2
            ),
            helper.make_node("Identity", ["new_k"], ["past_key_0_out"]),
            helper.make_node("Identity", ["new_v"], ["past_value_0_out"]),
        ],
        "part",
        [
            helper.make_tensor_value_info("past_key_0_in", f, [HEADS, 1, D, PAST]),
            helper.make_tensor_value_info("past_value_0_in", f, [HEADS, 1, PAST, D]),
            helper.make_tensor_value_info("new_k", f, [HEADS, 1, D, SEQ]),
            helper.make_tensor_value_info("new_v", f, [HEADS, 1, SEQ, D]),
        ],
        [
            helper.make_tensor_value_info("attn_k", f, [HEADS, 1, D, CTX]),
            helper.make_tensor_value_info("attn_v", f, [HEADS, 1, CTX, D]),
            helper.make_tensor_value_info("past_key_0_out", f, [HEADS, 1, D, SEQ]),
            helper.make_tensor_value_info("past_value_0_out", f, [HEADS, 1, SEQ, D]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    def enc(name: str) -> dict[str, Any]:
        return {"name": name, "bw": 8, "dtype": "INT", "enc_type": "PER_TENSOR",
                "is_sym": True, "offset": [-128.0], "scale": [0.1]}  # fmt: skip

    names = ["past_key_0_in", "past_value_0_in", "attn_k", "attn_v", "past_key_0_out"]
    encodings = {
        "version": "1.0.0",
        "activation_encodings": [enc(n) for n in names],
        "param_encodings": [],
    }
    return model, encodings


def run(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, session.run(None, feeds), strict=True))


@pytest.mark.parametrize("profile", ["k4_v4", "k8_v4"])
def test_codec_insertion_matches_oracle(profile: str) -> None:
    config = get_profile(profile)
    model, encodings = synthetic_part()
    new_model, new_enc, codec_io = apply_kv_codec(model, encodings, config, SEQ, CTX)
    onnx.checker.check_model(new_model, full_check=True)

    rng = np.random.default_rng(0)
    past_k = rng.standard_normal((HEADS, 1, D, PAST)).astype(np.float32) * 20
    past_v = rng.standard_normal((HEADS, 1, PAST, D)).astype(np.float32)
    new_k = rng.standard_normal((HEADS, 1, D, SEQ)).astype(np.float32) * 20
    new_v = rng.standard_normal((HEADS, 1, SEQ, D)).astype(np.float32)
    feeds = {"new_k": new_k, "new_v": new_v}
    expected_kinds = {"value"} if profile == "k8_v4" else {"key", "value"}
    assert {io.kind for io in codec_io} == expected_kinds

    stored = {"key": np.swapaxes(past_k, 2, 3), "value": past_v}
    present = {"key": np.swapaxes(new_k, 2, 3), "value": new_v}
    codecs = {}
    for io in codec_io:
        spec = getattr(config, io.kind)
        codec = PolarQuantReference(spec, D)
        codecs[io.kind] = codec
        idx, norms = codec.encode(stored[io.kind])
        feeds[io.packed_in] = pack_indices(idx, spec.bits)
        feeds[io.norm_in] = norms.astype(np.float32)
    if "key" not in expected_kinds:
        feeds["past_key_0_in"] = past_k
    outputs = run(new_model, feeds)

    for io in codec_io:
        codec = codecs[io.kind]
        attn = outputs["attn_k" if io.kind == "key" else "attn_v"]
        restored = (
            np.swapaxes(attn[..., :PAST], 2, 3)
            if io.kind == "key"
            else attn[:, :, :PAST]
        )
        report = compare_decode(
            codec, feeds[io.packed_in], feeds[io.norm_in], restored, FLOAT32_GRAPH
        )
        assert report["passed"], (io.kind, report)
        report = compare_encode(
            codec,
            present[io.kind],
            outputs[io.packed_out],
            outputs[io.norm_out],
            FLOAT32_GRAPH,
        )
        assert report["passed"], (io.kind, report)
        assert outputs[io.packed_out].shape == (HEADS, 1, SEQ, D // 2)
    np.testing.assert_array_equal(outputs["attn_k"][..., PAST:], new_k)
    np.testing.assert_array_equal(outputs["attn_v"][:, :, PAST:], new_v)
    if profile == "k8_v4":
        np.testing.assert_array_equal(outputs["attn_k"][..., :PAST], past_k)
        assert "past_key_0_out" in outputs

    removed = {f"past_{kind}_0_in" for kind in expected_kinds}
    kept = {e["name"] for e in new_enc["activation_encodings"]}
    assert kept == {e["name"] for e in encodings["activation_encodings"]} - removed
    assert {i.name for i in new_model.graph.input}.isdisjoint(removed)
    assert len(model.graph.input) == 4, "input model must not be mutated"


def test_rejects_baseline_profile_and_bad_lengths() -> None:
    model, encodings = synthetic_part()
    with pytest.raises(ValueError, match="stores no KV"):
        apply_kv_codec(model, encodings, get_profile("baseline_int8"), SEQ, CTX)
    with pytest.raises(ValueError, match="seq_len"):
        apply_kv_codec(model, encodings, get_profile("k4_v4"), CTX, CTX)
