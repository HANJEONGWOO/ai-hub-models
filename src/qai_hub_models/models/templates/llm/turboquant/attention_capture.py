# SPDX-License-Identifier: BSD-3-Clause
"""Offline-only instrumentation of the W4A16 FP16-KV reference graph."""

from __future__ import annotations

from collections import defaultdict

from onnx import TensorProto, helper

from qai_hub_models.models.templates.llm.turboquant.graph_surgery import SurgeryResult


def add_attention_capture(result: SurgeryResult, sequence_length: int) -> dict:
    if not result.fp16_attention:
        raise ValueError("Capture requires the FP16-KV attention control")
    groups = defaultdict(list)
    for item in result.fp16_attention:
        groups[item["layer"]].append(item)
    metadata = {}
    for layer, items in sorted(groups.items()):
        items.sort(key=lambda x: (x["head"], x["group"]))
        for kind, sources in (
            ("query", [x["qk"]["lhs"] for x in items]),
            ("output", [x["av"]["output"] for x in items]),
        ):
            name = f"capture_attn_{layer}_{kind}"
            result.model.graph.node.append(
                helper.make_node("Concat", sources, [name], name=name, axis=0)
            )
            result.model.graph.output.append(
                helper.make_tensor_value_info(
                    name, TensorProto.FLOAT16, [len(items), 1, sequence_length, 128]
                )
            )
        metadata[str(layer)] = items
    return metadata
