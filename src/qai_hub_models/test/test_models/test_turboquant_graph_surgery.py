# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""TurboQuant profile surgery on a synthetic delta-KV part, checked with onnxruntime.

The synthetic part mirrors the exported Qwen3 w4a16 KV path: a computing op
(the tap) feeds a value-preserving chain into ``past_*_out`` and, through a
shared tensor, the attention Concat; ``past_*_in`` is sliced per head before
that Concat.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from qai_hub_models.models.templates.llm.turboquant.config import (
    BASELINE,
    PROFILES,
    TurboQuantConfig,
    get_profile,
)
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
    regrid_encoding,
)
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

# Per-tensor 8-bit symmetric grids, as in the exported w4a16 encodings.
CACHE_SCALE = {"key": 0.1, "value": 0.02}
TAP_SCALE = {"key": 0.08, "value": 0.015}
ATTN_SCALE = {"key": 0.09, "value": 0.017}
TAP_BW = {"key": 8, "value": 16}


def _enc(name: str, scale: float, bw: int = 8) -> dict[str, Any]:
    offset = -(2 ** (bw - 1))
    return {"name": name, "bw": bw, "dtype": "INT", "enc_type": "PER_TENSOR",
            "is_sym": True, "offset": [float(offset)], "scale": [scale]}  # fmt: skip


def synthetic_part(opset: int = 17) -> tuple[onnx.ModelProto, dict[str, Any]]:
    f = TensorProto.FLOAT
    i64 = np.int64
    inits = [
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one"),
        numpy_helper.from_array(np.array([0], dtype=i64), "s0"),
        numpy_helper.from_array(np.array([PAST], dtype=i64), "e_past"),
        numpy_helper.from_array(np.array([3], dtype=i64), "ax3"),
        numpy_helper.from_array(np.array([2], dtype=i64), "ax2"),
    ]
    nodes = [
        # Key tap (stands in for the R3 MatMul): tokens last after a transpose
        # that is shared by the cache write and the attention Concat.
        helper.make_node("Mul", ["new_k", "one"], ["k_tap"]),
        helper.make_node("Transpose", ["k_tap"], ["k_hub"], perm=[0, 1, 3, 2]),
        helper.make_node("Identity", ["k_hub"], ["past_key_0_out"]),
        helper.make_node("Slice", ["past_key_0_in", "s0", "e_past", "ax3"], ["k_past"]),
        helper.make_node("Concat", ["k_past", "k_hub"], ["attn_k"], axis=3),
        # Value tap (stands in for v_proj): already in hub layout.
        helper.make_node("Mul", ["new_v", "one"], ["v_tap"]),
        helper.make_node("Identity", ["v_tap"], ["past_value_0_out"]),
        helper.make_node(
            "Slice", ["past_value_0_in", "s0", "e_past", "ax2"], ["v_past"]
        ),
        helper.make_node("Concat", ["v_past", "v_tap"], ["attn_v"], axis=2),
        helper.make_node("Mul", ["attn_v", "one"], ["unrelated"]),
    ]
    graph = helper.make_graph(
        nodes,
        "part",
        [
            helper.make_tensor_value_info("past_key_0_in", f, [HEADS, 1, D, PAST]),
            helper.make_tensor_value_info("past_value_0_in", f, [HEADS, 1, PAST, D]),
            helper.make_tensor_value_info("new_k", f, [HEADS, 1, SEQ, D]),
            helper.make_tensor_value_info("new_v", f, [HEADS, 1, SEQ, D]),
        ],
        [
            helper.make_tensor_value_info("attn_k", f, [HEADS, 1, D, CTX]),
            helper.make_tensor_value_info("attn_v", f, [HEADS, 1, CTX, D]),
            helper.make_tensor_value_info("unrelated", f, [HEADS, 1, CTX, D]),
            helper.make_tensor_value_info("past_key_0_out", f, [HEADS, 1, D, SEQ]),
            helper.make_tensor_value_info("past_value_0_out", f, [HEADS, 1, SEQ, D]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    encodings = {
        "version": "1.0.0",
        "activation_encodings": [
            _enc("k_tap", TAP_SCALE["key"], TAP_BW["key"]),
            _enc("k_hub", TAP_SCALE["key"]),
            _enc("past_key_0_out", CACHE_SCALE["key"]),
            _enc("past_key_0_in", CACHE_SCALE["key"]),
            _enc("k_past", CACHE_SCALE["key"]),
            _enc("attn_k", ATTN_SCALE["key"]),
            _enc("v_tap", TAP_SCALE["value"], TAP_BW["value"]),
            _enc("past_value_0_out", CACHE_SCALE["value"]),
            _enc("past_value_0_in", CACHE_SCALE["value"]),
            _enc("v_past", CACHE_SCALE["value"]),
            _enc("attn_v", ATTN_SCALE["value"]),
            _enc("unrelated", 0.5, 16),
        ],
        "param_encodings": [],
    }
    return model, encodings


def run(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, session.run(None, feeds), strict=True))


def inputs(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "past_key_0_in": rng.standard_normal((HEADS, 1, D, PAST)).astype(np.float32)
        * 20,
        "past_value_0_in": rng.standard_normal((HEADS, 1, PAST, D)).astype(np.float32),
        "new_k": rng.standard_normal((HEADS, 1, SEQ, D)).astype(np.float32) * 20,
        "new_v": rng.standard_normal((HEADS, 1, SEQ, D)).astype(np.float32),
    }


def edits_by_tensor(result: Any) -> dict[str, str]:
    return {e.tensor: e.action for e in result.edits}


def enc_map(encodings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {e["name"]: e for e in encodings["activation_encodings"]}


def node_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {n.output[0]: n for n in model.graph.node}


# Value-only codec (exported affine int8 K); not a shipped profile.
V_ONLY = TurboQuantConfig("v_only", BASELINE, get_profile("k4_v4").value)

# Expected encoding edits per KV tensor a profile changes.
PRE_INT8_EDITS = {
    "key": {
        "past_key_0_in": "drop",
        "past_key_0_out": "drop",
        "k_hub": "duplicate",
        "k_tap": "regrid",
        "k_tap_tq_cache": "guard",
    },
    "value": {
        "past_value_0_in": "drop",
        "past_value_0_out": "drop",
        "v_tap_tq_cache": "guard",
    },
}
# Nodes the cache branch adds per pre-int8 kind: a guard, plus the shared Transpose copy for K.
CACHE_NODES = {"key": 2, "value": 1}


def check_cache_chain(model: onnx.ModelProto, after: dict[str, Any]) -> None:
    """The cache branch starts at a guard on the tap; attention keeps the original chain."""
    nodes = node_map(model)
    guard = nodes["k_tap_tq_cache"]
    assert guard.op_type == "Max" and list(guard.input) == ["k_tap", "k_tap"]
    copy_node = nodes["k_hub_tq_cache"]
    assert copy_node.op_type == "Transpose"
    assert list(copy_node.input) == ["k_tap_tq_cache"]
    assert list(nodes["past_key_0_out"].input) == ["k_hub_tq_cache"]
    assert list(nodes["attn_k"].input) == ["k_past", "k_hub"]
    assert list(nodes["past_value_0_out"].input) == ["v_tap_tq_cache"]
    assert list(nodes["attn_v"].input) == ["v_past", "v_tap"]
    for name in (
        "k_tap_tq_cache",
        "k_hub_tq_cache",
        "v_tap_tq_cache",
        "past_key_0_out",
    ):
        assert name not in after
    assert after["k_hub"]["scale"] == [TAP_SCALE["key"]] and after["k_hub"]["bw"] == 8


@pytest.mark.parametrize("opset", [17, 18])
@pytest.mark.parametrize(
    ("config", "expected_kinds"),
    [(get_profile("k4_v4"), {"key", "value"}), (V_ONLY, {"value"})],
    ids=["k4_v4", "v_only"],
)
def test_codec_insertion_matches_oracle(
    config: TurboQuantConfig, expected_kinds: set[str], opset: int
) -> None:
    # The shipped Qwen3 parts are opset 18, where ReduceMax takes axes as an input.
    model, encodings = synthetic_part(opset)
    result = apply_kv_profile(model, encodings, config, SEQ, CTX)
    onnx.checker.check_model(result.model, full_check=True)
    assert {io.kind for io in result.codec_io} == expected_kinds

    raw = inputs()
    feeds = {"new_k": raw["new_k"], "new_v": raw["new_v"]}
    stored = {
        "key": np.swapaxes(raw["past_key_0_in"], 2, 3),
        "value": raw["past_value_0_in"],
    }
    present = {"key": raw["new_k"], "value": raw["new_v"]}
    codecs = {}
    for io in result.codec_io:
        spec = getattr(config, io.kind)
        codec = PolarQuantReference(spec, D)
        codecs[io.kind] = codec
        idx, norms = codec.encode(stored[io.kind])
        feeds[io.packed_in] = pack_indices(idx, spec.bits)
        feeds[io.norm_in] = norms.astype(np.float32)
    if "key" not in expected_kinds:
        feeds["past_key_0_in"] = raw["past_key_0_in"]
    outputs = run(result.model, feeds)

    for io in result.codec_io:
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
            codec, present[io.kind], outputs[io.packed_out], outputs[io.norm_out],
            FLOAT32_GRAPH,
        )  # fmt: skip
        assert report["passed"], (io.kind, report)
        assert outputs[io.packed_out].shape == (HEADS, 1, SEQ, D // 2)
    np.testing.assert_array_equal(
        outputs["attn_k"][..., PAST:], np.swapaxes(raw["new_k"], 2, 3)
    )
    np.testing.assert_array_equal(outputs["attn_v"][:, :, PAST:], raw["new_v"])
    if "key" not in expected_kinds:
        np.testing.assert_array_equal(
            outputs["attn_k"][..., :PAST], raw["past_key_0_in"]
        )
        assert "past_key_0_out" in outputs

    expected_edits: dict[str, str] = {}
    for kind in expected_kinds:
        expected_edits.update(PRE_INT8_EDITS[kind])
    assert edits_by_tensor(result) == expected_edits
    before, after = enc_map(encodings), enc_map(result.encodings)
    dropped = {t for t, a in expected_edits.items() if a == "drop"}
    assert set(after) == set(before) - dropped
    for name in after:
        if expected_edits.get(name) == "regrid":
            assert after[name] == regrid_encoding(before[name])
        else:
            assert after[name] == before[name]
    if "key" in expected_kinds:
        check_cache_chain(result.model, after)
    assert {i.name for i in result.model.graph.input}.isdisjoint(
        {f"past_{kind}_0_in" for kind in expected_kinds}
    )
    assert len(model.graph.input) == 4, "input model must not be mutated"
    assert len(model.graph.node) == 10, "input model must not be mutated"
    assert enc_map(encodings) == before, "input encodings must not be mutated"


def test_pre_int8_paths_report_taps_and_kept_attention_encodings() -> None:
    model, encodings = synthetic_part()
    result = apply_kv_profile(model, encodings, get_profile("k4_v4"), SEQ, CTX)
    paths = {p.kind: p for p in result.paths}
    assert paths["key"].taps == ("k_tap",) and paths["key"].tap_ops == ("Mul",)
    assert paths["value"].taps == ("v_tap",)
    assert paths["key"].kept_consumers == ("attn_k",)
    assert paths["value"].kept_consumers == ("attn_v",)
    after = enc_map(result.encodings)
    # The attention-side grid consumed by the 16x8 MatMul is untouched, and so
    # are the per-head slices of the past (QAIRT leaves them float).
    assert after["attn_k"]["scale"] == [ATTN_SCALE["key"]]
    assert after["attn_v"]["scale"] == [ATTN_SCALE["value"]]
    assert after["k_past"] == enc_map(encodings)["k_past"]
    # An 8-bit tap is widened over the same range; a 16-bit tap is left alone.
    assert after["k_tap"]["bw"] == 16
    assert after["k_tap"]["offset"] == [-32768.0]
    assert after["k_tap"]["scale"] == [TAP_SCALE["key"] / 256]
    assert after["v_tap"] == enc_map(encodings)["v_tap"]
    check_cache_chain(result.model, after)


def test_int16_kv_profile_stores_the_tap_grid_exactly() -> None:
    model, encodings = synthetic_part()
    result = apply_kv_profile(
        model, encodings, get_profile("baseline_int16_kv"), SEQ, CTX
    )
    onnx.checker.check_model(result.model, full_check=True)
    assert result.codec_io == []
    assert [i.name for i in result.model.graph.input] == [
        i.name for i in model.graph.input
    ]
    assert [o.name for o in result.model.graph.output] == [
        o.name for o in model.graph.output
    ]
    # Only the shared K Transpose is copied; there is no float guard.
    assert len(result.model.graph.node) == len(model.graph.node) + 1
    assert edits_by_tensor(result) == {
        "past_key_0_in": "set",
        "k_past": "set",
        "past_key_0_out": "set",
        "k_hub": "duplicate",
        "k_tap": "regrid",
        "past_value_0_in": "set",
        "v_past": "set",
        "past_value_0_out": "set",
    }
    before, after = enc_map(encodings), enc_map(result.encodings)

    def grid(enc: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in enc.items() if k != "name"}

    # Every tensor from the K tap to the cache input slices shares the tap's
    # 16-bit grid; the V tap already was 16-bit, so its grid is reused as is.
    k_grid = grid(regrid_encoding(before["k_tap"]))
    for name in (
        "k_tap",
        "k_hub_tq_cache",
        "past_key_0_out",
        "past_key_0_in",
        "k_past",
    ):
        assert grid(after[name]) == k_grid, name
    for name in ("v_tap", "past_value_0_out", "past_value_0_in", "v_past"):
        assert grid(after[name]) == grid(before["v_tap"]), name
    assert after["k_hub"] == before["k_hub"] and after["attn_k"] == before["attn_k"]
    assert after["attn_v"] == before["attn_v"]
    nodes = node_map(result.model)
    assert list(nodes["k_hub_tq_cache"].input) == ["k_tap"]
    assert list(nodes["past_key_0_out"].input) == ["k_hub_tq_cache"]
    assert list(nodes["past_value_0_out"].input) == ["v_tap"]
    raw = inputs(1)
    outs_before, outs_after = run(model, raw), run(result.model, raw)
    for name in outs_before:
        np.testing.assert_array_equal(outs_before[name], outs_after[name])


def test_regrid_keeps_range() -> None:
    enc = _enc("t", 0.4)
    new = regrid_encoding(enc)
    assert new["bw"] == 16
    assert min(new["offset"]) * new["scale"][0] == pytest.approx(-128 * 0.4)
    assert (65535 + new["offset"][0]) * new["scale"][0] >= 127 * 0.4
    assert enc["bw"] == 8, "input must not be mutated"


def test_profiles_are_distinct_artifacts() -> None:
    hashes = {name: cfg.config_hash() for name, cfg in PROFILES.items()}
    assert len(set(hashes.values())) == len(PROFILES)
    assert get_profile("k4_v4").to_dict()["value"]["kind"] == "polar"
    assert get_profile("baseline_int16_kv").modifies_graph
    assert not get_profile("baseline_int16_kv").enabled


def test_rejects_baseline_profile_and_bad_lengths() -> None:
    model, encodings = synthetic_part()
    with pytest.raises(ValueError, match="keeps the exported KV path"):
        apply_kv_profile(model, encodings, get_profile("baseline_int8"), SEQ, CTX)
    with pytest.raises(ValueError, match="seq_len"):
        apply_kv_profile(model, encodings, get_profile("k4_v4"), CTX, CTX)
