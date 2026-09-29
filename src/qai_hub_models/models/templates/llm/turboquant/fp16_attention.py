# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Uncompressed FP16 KV and FP16-input QK/AV control for TurboQuant experiments.

The W4A16 producers and score/softmax/output encodings are retained. This is
not an all-FP16 model and does not prescribe the hardware accumulation dtype.
Current KV is read from the exact FP16 tensors exported to the host cache.
"""

from __future__ import annotations

import copy

from onnx import TensorProto

from qai_hub_models.models.templates.llm.turboquant.config import TurboQuantConfig
from qai_hub_models.models.templates.llm.turboquant.current_attention import (
    _topological_sort,
)
from qai_hub_models.models.templates.llm.turboquant.export import Subgraph
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    SurgeryResult,
    _GraphIndex,
)
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    _heads,
    _prune,
    _slice,
)


def use_fp16_kv_attention(
    result: SurgeryResult, config: TurboQuantConfig
) -> SurgeryResult:
    """Apply once, after KV tap surgery; reject unsupported attention patterns."""
    if not config.fp16_attention or result.fp16_attention or result.codec_io:
        raise ValueError("Requires a fresh, uncompressed FP16 KV surgery result.")
    result = copy.deepcopy(result)
    graph = result.model.graph
    index = _GraphIndex(graph)
    inputs = {v.name: v for v in graph.input}
    outputs = {v.name: v for v in graph.output}
    acts = {e["name"]: e for e in result.encodings["activation_encodings"]}
    sg = Subgraph()
    removed: set[str] = set()

    for layer in sorted({p.layer for p in result.paths}):
        sources = {kind: f"past_{kind}_{layer}_in" for kind in ("key", "value")}
        count = inputs[sources["key"]].type.tensor_type.shape.dim[0].dim_value
        heads = _heads(result, index, layer, count, sources)
        for kind in ("key", "value"):
            source, present = sources[kind], f"past_{kind}_{layer}_out"
            inputs[source].type.tensor_type.elem_type = TensorProto.FLOAT16
            outputs[present].type.tensor_type.elem_type = TensorProto.FLOAT16
            producer = index.producer[present]
            # The cache output is a delta (current chunk), not the whole context.
            intermediate = f"fp16_{kind}_{layer}_present_float"
            producer.output[list(producer.output).index(present)] = intermediate
            producer.name = intermediate
            sg.node("Cast", [intermediate], [present], to=TensorProto.FLOAT16)
            for info in graph.value_info:
                if info.name == present:
                    info.type.tensor_type.elem_type = TensorProto.FLOAT16
            acts.pop(source, None)
            acts.pop(present, None)

        for head in heads:
            prefix = f"fp16_attn_{layer}_head{head.head}_"
            kv = {}
            for kind, axis in (("key", 3), ("value", 2)):
                past = _slice(
                    sg,
                    sources[kind],
                    prefix + kind + "_past",
                    head.head,
                    head.head + 1,
                    0,
                )
                current = _slice(
                    sg,
                    f"past_{kind}_{layer}_out",
                    prefix + kind + "_current",
                    head.head,
                    head.head + 1,
                    0,
                )
                kv[kind] = prefix + kind + "_cat"
                sg.node("Concat", [past, current], [kv[kind]], axis=axis)
            if head.key_scale is not None:
                divisor = prefix + "divisor"
                sg.node(
                    "Cast", [head.key_scale.input[1]], [divisor], to=TensorProto.FLOAT16
                )
                sg.node("Div", [kv["key"], divisor], [prefix + "key_scaled"])
                kv["key"] = prefix + "key_scaled"
            for group, (qk, av) in enumerate(head.products):
                report = {
                    "layer": layer,
                    "head": head.head,
                    "group": group,
                    "kv": kv.copy(),
                }
                for kind, node, rhs in (("qk", qk, kv["key"]), ("av", av, kv["value"])):
                    out = node.output[0]
                    if out not in acts or node.input[0] not in acts:
                        raise ValueError(
                            f"Missing original {kind} boundary encoding: {out}"
                        )
                    stem = prefix + f"q{group}_{kind}"
                    sg.node(
                        "Cast", [node.input[0]], [stem + "_lhs"], to=TensorProto.FLOAT16
                    )
                    sg.node("MatMul", [stem + "_lhs", rhs], [stem + "_half"])
                    # Preserve calibrated score/final-output grids, as TurboQuant does.
                    sg.node("Cast", [stem + "_half"], [out], to=TensorProto.FLOAT)
                    removed.add(out)
                    report[kind] = {
                        "lhs": stem + "_lhs",
                        "rhs": rhs,
                        "output": stem + "_half",
                        "boundary": out,
                    }
                result.fp16_attention.append(report)
    kept = [n for n in graph.node if not removed.intersection(n.output)]
    del graph.node[:]
    graph.node.extend(kept + sg.nodes)
    existing = {t.name for t in graph.initializer}
    graph.initializer.extend(
        t for name, t in sg.initializers.items() if name not in existing
    )
    _topological_sort(graph)
    _prune(result.model)
    live = {v.name for v in graph.input} | {o for n in graph.node for o in n.output}
    result.encodings["activation_encodings"] = [
        e for name, e in acts.items() if name in live
    ]
    return result
