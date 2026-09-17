# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tiled attention equivalence, masking, ABI and fail-closed pattern matching."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.export import Subgraph
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)
from qai_hub_models.models.templates.llm.turboquant.packing import pack_indices
from qai_hub_models.models.templates.llm.turboquant.reference import PolarQuantReference
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)

HEADS, GROUPS, D, CONTEXT = 2, 2, 128, 35
CONFIG = get_profile("k4_v4")


def attention_part(seq: int) -> tuple[onnx.ModelProto, dict[str, Any]]:
    sg, qks, softmaxes, avs = Subgraph(), Subgraph(), Subgraph(), Subgraph()
    acts = []
    inputs = []
    outputs = []

    def value(name: str, shape: list[int]) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    def enc(name: str, bits: int = 16) -> None:
        acts.append(
            {
                "name": name,
                "bw": bits,
                "dtype": "INT",
                "enc_type": "PER_TENSOR",
                "is_sym": True,
                "offset": [-float(2 ** (bits - 1))],
                "scale": [4.0 / (2 ** (bits - 1))],
            }
        )

    one = sg.const("one", np.array(1.0, dtype=np.float32))
    for kind in ("key", "value"):
        dims = (
            [HEADS, 1, D, CONTEXT - seq]
            if kind == "key"
            else [HEADS, 1, CONTEXT - seq, D]
        )
        inputs += [
            value(f"past_{kind}_0_in", dims),
            value(f"new_{kind}", [HEADS, 1, seq, D]),
        ]
        cache_heads = []
        for h in range(HEADS):
            stem = f"{kind}_{h}"
            sg.node(
                "Slice",
                [f"new_{kind}", sg.shape([h]), sg.shape([h + 1]), sg.shape([0])],
                [stem + "_raw"],
            )
            sg.node("Mul", [stem + "_raw", one], [stem + "_tap"])
            enc(stem + "_tap", 8)
            present = stem + "_tap"
            if kind == "key":
                sg.node("Transpose", [present], [stem + "_hub"], perm=[0, 1, 3, 2])
                present = stem + "_hub"
                enc(present, 8)
            cache_heads.append(present)
            sg.node(
                "Slice",
                [f"past_{kind}_0_in", sg.shape([h]), sg.shape([h + 1]), sg.shape([0])],
                [stem + "_past"],
            )
            sg.node(
                "Concat",
                [stem + "_past", present],
                [stem + "_cat"],
                axis=3 if kind == "key" else 2,
            )
            enc(stem + "_past", 8)
            enc(stem + "_cat", 8)
        out = f"past_{kind}_0_out"
        sg.node("Concat", cache_heads, [out], axis=0)
        outputs.append(
            value(out, [HEADS, 1, D, seq] if kind == "key" else [HEADS, 1, seq, D])
        )
        enc(f"past_{kind}_0_in", 8)
        enc(out, 8)
    inputs.append(value("mask", [1, 1, seq, CONTEXT]))
    for h in range(HEADS):
        for g in range(GROUPS):
            p = f"h{h}g{g}"
            inputs.append(value(p + "_q", [1, 1, seq, D]))
            qks.node("MatMul", [p + "_q", f"key_{h}_cat"], [p + "_qk"])
            softmaxes.node("Add", [p + "_qk", "mask"], [p + "_masked"])
            softmaxes.node("Softmax", [p + "_masked"], [p + "_prob"], axis=-1)
            avs.node("MatMul", [p + "_prob", f"value_{h}_cat"], [p + "_out"])
            outputs.append(value(p + "_out", [1, 1, seq, D]))
            for suffix in ("_q", "_qk", "_masked", "_prob", "_out"):
                enc(p + suffix)
    for sub in (qks, softmaxes, avs):
        sg.extend(sub)
    graph = helper.make_graph(
        sg.nodes, "attention", inputs, outputs, list(sg.initializers.values())
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model, {
        "version": "1.0.0",
        "activation_encodings": acts,
        "param_encodings": [],
    }


def feeds(seq: int, valid_past: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    data = {}
    for kind in ("key", "value"):
        codec = PolarQuantReference(getattr(CONFIG, kind), D)
        past = rng.standard_normal((HEADS, 1, CONTEXT - seq, D))
        # Unused slots must be harmless even when packed bytes are arbitrary.
        past[:, :, : CONTEXT - seq - valid_past] = 0
        indices, norm = codec.encode(past)
        data[f"tq_{kind}_0_packed_in"] = pack_indices(indices, 4)
        data[f"tq_{kind}_0_norm_in"] = norm.astype(np.float32)
        data[f"new_{kind}"] = rng.standard_normal((HEADS, 1, seq, D)).astype(np.float32)
    mask = np.zeros((1, 1, seq, CONTEXT), dtype=np.float32)
    mask[..., : CONTEXT - seq - valid_past] = -10000
    for row in range(seq):
        mask[..., row, CONTEXT - seq + row + 1 :] = -10000
    data["mask"] = mask
    for h in range(HEADS):
        for g in range(GROUPS):
            data[f"h{h}g{g}_q"] = (
                rng.standard_normal((1, 1, seq, D)) / np.sqrt(D)
            ).astype(np.float32)
    return data


@pytest.mark.parametrize("seq", [1, 3])
@pytest.mark.parametrize("tile", [1, 7, 16])
@pytest.mark.parametrize("valid_past", [0, 13, 32])
def test_tiled_attention_matches_full_restore(
    seq: int, tile: int, valid_past: int
) -> None:
    model, encodings = attention_part(seq)
    full = apply_kv_profile(model, encodings, CONFIG, seq, CONTEXT)
    original = full.model.SerializeToString()
    tiled = tile_kv_attention(full, CONFIG, tile)
    assert full.model.SerializeToString() == original
    onnx.checker.check_model(tiled.model, full_check=True)
    data = feeds(seq, min(valid_past, CONTEXT - seq))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = [
        ort.InferenceSession(
            m.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        for m in (full.model, tiled.model)
    ]
    expected, actual = [s.run(None, data) for s in sessions]
    for out, ref in zip(actual, expected, strict=True):
        np.testing.assert_allclose(out, ref, rtol=2e-5, atol=2e-6)
    for attr in ("input", "output"):
        assert [v.SerializeToString() for v in getattr(full.model.graph, attr)] == [
            v.SerializeToString() for v in getattr(tiled.model.graph, attr)
        ]
    assert full.encodings["param_encodings"] == tiled.encodings["param_encodings"]
    node_names = {o for n in tiled.model.graph.node for o in n.output}
    assert not {"tq_key_0_restored", "tq_value_0_restored"}.intersection(node_names)
    shapes = onnx.shape_inference.infer_shapes(tiled.model)
    for info in shapes.graph.value_info:
        if "_tile" in info.name and info.name.endswith("_restored"):
            dims = [d.dim_value for d in info.type.tensor_type.shape.dim]
            assert dims[2] <= tile and dims[3] == D
    acts = {e["name"]: e for e in tiled.encodings["activation_encodings"]}
    for report in tiled.attention_tiles:
        for chunk in report["tiles"]:
            for kind, names in chunk["attention_concats"].items():
                for h, name in enumerate(names):
                    original_enc = next(
                        e
                        for e in encodings["activation_encodings"]
                        if e["name"] == f"{kind}_{h}_cat"
                    )
                    assert acts[name] == {**original_enc, "name": name}


@pytest.mark.parametrize("tile", [0, -1, CONTEXT])
def test_invalid_tile_rejected(tile: int) -> None:
    model, encodings = attention_part(1)
    with pytest.raises(ValueError, match=r"positive tile size|does not tile"):
        tile_kv_attention(
            apply_kv_profile(model, encodings, CONFIG, 1, CONTEXT), CONFIG, tile
        )


def test_unrecognized_consumers_and_missing_encodings_rejected() -> None:
    model, encodings = attention_part(1)
    full = apply_kv_profile(model, encodings, CONFIG, 1, CONTEXT)
    broken = copy.deepcopy(full)
    next(n for n in broken.model.graph.node if n.output[0] == "h0g0_out").input[1] = (
        "value_1_cat"
    )
    with pytest.raises(ValueError, match="Mismatched"):
        tile_kv_attention(broken, CONFIG, 7)
    for missing in ("h0g0_qk", "key_0_past", "value_0_past"):
        broken = copy.deepcopy(full)
        broken.encodings["activation_encodings"] = [
            e for e in broken.encodings["activation_encodings"] if e["name"] != missing
        ]
        with pytest.raises(ValueError, match="Missing calibrated"):
            tile_kv_attention(broken, CONFIG, 7)
