# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""FP16 cache/attention control: precision, current KV, causal masking and ABI."""

from __future__ import annotations

import copy
import importlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import (
    BASELINE,
    FP16,
    TurboQuantConfig,
    get_profile,
)
from qai_hub_models.models.templates.llm.turboquant.fp16_attention import (
    use_fp16_kv_attention,
)
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)
from qai_hub_models.test.test_models.test_turboquant_tiled_attention import (
    CONTEXT,
    GROUPS,
    HEADS,
    D,
    attention_part,
)

CONFIG = get_profile("baseline_fp16_kv_fp16_attn")


@pytest.mark.parametrize("seq", [1, 3])
@pytest.mark.parametrize("valid_past", [0, 13, 32])
@pytest.mark.parametrize("divisor", [None, 8.0])
def test_fp16_attention_oracle(
    seq: int, valid_past: int, divisor: float | None
) -> None:
    model, encodings = attention_part(seq, divisor)
    before = model.SerializeToString(), copy.deepcopy(encodings)
    result = use_fp16_kv_attention(
        apply_kv_profile(model, encodings, CONFIG, seq, CONTEXT), CONFIG
    )
    assert (model.SerializeToString(), encodings) == before
    onnx.checker.check_model(result.model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(result.model)
    types = {
        v.name: v.type.tensor_type.elem_type
        for v in [
            *inferred.graph.input,
            *inferred.graph.output,
            *inferred.graph.value_info,
        ]
    }
    producers = {o: n for n in result.model.graph.node for o in n.output}
    for record in result.fp16_attention:
        for kind in ("qk", "av"):
            op = producers[record[kind]["output"]]
            assert op.op_type == "MatMul"
            assert [types[t] for t in op.input] == [TensorProto.FLOAT16] * 2
        for kind in ("key", "value"):
            current = f"fp16_attn_0_head{record['head']}_{kind}_current"
            assert producers[current].input[0] == f"past_{kind}_0_out"
    original_acts = {e["name"]: e for e in encodings["activation_encodings"]}
    acts = {e["name"]: e for e in result.encodings["activation_encodings"]}
    for name, enc in acts.items():
        if not name.endswith("_tap"):
            assert enc == original_acts[name]
    assert not any(n.startswith(("past_", "fp16_")) for n in acts)
    assert result.encodings["param_encodings"] == encodings["param_encodings"]
    rng = np.random.default_rng(31)
    data = {}
    for v in result.model.graph.input:
        shape = [d.dim_value for d in v.type.tensor_type.shape.dim]
        dtype = (
            np.float16
            if v.type.tensor_type.elem_type == TensorProto.FLOAT16
            else np.float32
        )
        data[v.name] = (rng.standard_normal(shape) * 0.1).astype(dtype)
    mask = data["mask"].astype(np.float32)
    data["mask"] = mask
    mask[:] = 0
    mask[..., : max(0, CONTEXT - seq - valid_past)] = -10000
    for row in range(seq):
        mask[..., row, CONTEXT - seq + row + 1 :] = -10000
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        result.model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    actual = dict(
        zip(
            [v.name for v in result.model.graph.output],
            session.run(None, data),
            strict=True,
        )
    )
    k = data["new_key"].transpose(0, 1, 3, 2).astype(np.float16)
    v = data["new_value"].astype(np.float16)
    np.testing.assert_array_equal(actual["past_key_0_out"], k)
    np.testing.assert_array_equal(actual["past_value_0_out"], v)
    k = np.concatenate([data["past_key_0_in"], k], axis=3)
    v = np.concatenate([data["past_value_0_in"], v], axis=2)
    if divisor is not None:
        k = (k / np.float16(divisor)).astype(np.float16)
    for h in range(HEADS):
        for g in range(GROUPS):
            prefix = f"h{h}g{g}"
            q = data[prefix + "_q"].astype(np.float16)
            score = (q.astype(np.float32) @ k[h : h + 1].astype(np.float32)).astype(
                np.float16
            ).astype(np.float32) + mask
            prob = np.exp(score - np.max(score, axis=-1, keepdims=True))
            prob = (prob / np.sum(prob, axis=-1, keepdims=True)).astype(np.float16)
            expected = (
                (prob.astype(np.float32) @ v[h : h + 1].astype(np.float32))
                .astype(np.float16)
                .astype(np.float32)
            )
            np.testing.assert_allclose(
                actual[prefix + "_out"], expected, atol=6e-5, rtol=2e-3
            )


def test_fp16_cache_rounding_memory_reset_and_snapshot() -> None:
    cache = TurboQuantKVCache(CONFIG, 1, HEADS, D, CONTEXT)
    rng = np.random.default_rng(3)
    kv = [
        rng.standard_normal(shape).astype(np.float32)
        for shape in ((HEADS, 1, D, 3), (HEADS, 1, 3, D))
    ]
    cache.append(kv)
    for actual, expected in zip(cache.layer_float(0), kv, strict=True):
        np.testing.assert_array_equal(
            actual, expected.astype(np.float16).astype(np.float32)
        )
    assert (
        sum(s.arrays()["raw"].nbytes for s in cache.layers[0])
        == 2 * HEADS * D * CONTEXT * 2
    )
    state = cache.state_dict()
    restored = TurboQuantKVCache(CONFIG, 1, HEADS, D, CONTEXT)
    restored.load_state_dict(state)
    for actual, expected in zip(
        restored.layer_float(0), cache.layer_float(0), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    cache.reset()
    assert cache.get_seq_length() == 0
    with pytest.raises(ValueError, match="config"):
        TurboQuantKVCache(
            get_profile("baseline_int16_kv"), 1, HEADS, D, CONTEXT
        ).load_state_dict(state)


def test_fp16_configuration_and_fail_closed() -> None:
    assert CONFIG.fp16_attention and not CONFIG.enabled
    with pytest.raises(ValueError, match="both K and V"):
        TurboQuantConfig("invalid", FP16, BASELINE)
    model, enc = attention_part(1)
    first = apply_kv_profile(model, enc, CONFIG, 1, CONTEXT)
    with pytest.raises(ValueError, match="fresh"):
        use_fp16_kv_attention(use_fp16_kv_attention(first, CONFIG), CONFIG)
    node = next(n for n in first.model.graph.node if n.output[0] == "h0g0_qk")
    node.op_type = "Add"
    with pytest.raises(ValueError, match="Unsupported QK"):
        use_fp16_kv_attention(first, CONFIG)


@pytest.mark.parametrize(
    "fault", [None, "kv_io", "qk", "hidden_int8", "raw_current", "missing_product"]
)
def test_compiled_audit_rejects_precision_and_wiring_regressions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str | None
) -> None:
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    monkeypatch.syspath_prepend(str(scripts))
    audit = importlib.import_module("verify_fp16_attention")
    boundary = importlib.import_module("verify_kv_boundary")
    model, enc = attention_part(1)
    result = use_fp16_kv_attention(
        apply_kv_profile(model, enc, CONFIG, 1, CONTEXT), CONFIG
    )
    inferred = onnx.shape_inference.infer_shapes(result.model)
    half = {
        v.name
        for v in [
            *inferred.graph.input,
            *inferred.graph.output,
            *inferred.graph.value_info,
        ]
        if v.type.tensor_type.elem_type == TensorProto.FLOAT16
    }
    static = {t.name for t in inferred.graph.initializer}

    def tensor(name: str) -> Any:
        return boundary.Tensor(
            name,
            "Float_16" if name in half else "uFxp_16",
            "",
            "STATIC" if name in static else "NATIVE",
        )

    ops, producers, consumers = [], {}, defaultdict(list)
    for n in result.model.graph.node:
        op = boundary.Op(
            n.name,
            n.name,
            "StridedSlice" if n.op_type == "Slice" else n.op_type,
            [tensor(t) for t in n.input],
            [tensor(t) for t in n.output],
        )
        ops.append(op)
        for t in op.outputs:
            producers[t.name] = op
        for t in op.inputs:
            consumers[t.name].append(op)
    io = {
        side: {
            v.name: {"dtype": "Float_16", "encoding": "No encoding"}
            for v in values
            if v.name.startswith("past_")
        }
        for side, values in (
            ("input", result.model.graph.input),
            ("output", result.model.graph.output),
        )
    }
    if fault == "kv_io":
        io["input"]["past_key_0_in"]["dtype"] = "uFxp_8"
    elif fault == "qk":
        producers["fp16_attn_0_head0_q0_qk_half"].inputs[1].dtype = "uFxp_8"
    elif fault == "hidden_int8":
        producers["fp16_attn_0_head0_key_past"].outputs[0].dtype = "uFxp_8"
    elif fault == "raw_current":
        producers["fp16_attn_0_head0_key_current"].inputs[0].name = "new_key"
    elif fault == "missing_product":
        ops.remove(producers["fp16_attn_0_head0_q0_qk_half"])
    info = boundary.DlcInfo(ops, producers, consumers, io)
    monkeypatch.setattr(audit, "parse_dlcinfo", lambda _: info)
    monkeypatch.setattr(audit.onnx, "load", lambda *args, **kwargs: result.model)
    (tmp_path / "graph.kv_edits.json").write_text(json.dumps(result.report()))
    report = audit.verify_graph(tmp_path, "graph")
    assert bool(report["violations"]) == (fault is not None)
