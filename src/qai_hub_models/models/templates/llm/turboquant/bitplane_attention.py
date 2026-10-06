# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""HTP packed-K -> score replacement; no restored K or bit-plane graph output."""

from __future__ import annotations

import copy
import re

import numpy as np
from onnx import TensorProto, helper

from qai_hub_models.models.templates.llm.turboquant.config import TurboQuantConfig
from qai_hub_models.models.templates.llm.turboquant.current_attention import (
    _topological_sort,
)
from qai_hub_models.models.templates.llm.turboquant.export import Subgraph
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import SurgeryResult
from qai_hub_models.models.templates.llm.turboquant.native_decoder import NATIVE_DOMAIN
from qai_hub_models.models.templates.llm.turboquant.structured import load_parameters
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    _prune,
    _slice,
)

BITPLANE_OP = "BitplaneQK4"


def use_bitplane_qk(result: SurgeryResult, config: TurboQuantConfig) -> SurgeryResult:
    """Final pass after current-KV quantization. Unknown patterns fail closed.

    Keep query rotation/scaling, score dtype/encoding boundary, global mask/softmax,
    V/AV, and per-head GQA mapping exactly as they were. Current packed K is
    read from the very same output that the host appends to the cache.
    """
    if (
        not config.bitplane_qk
        or not result.current_kv_attention
        or not result.attention_tiles
    ):
        raise ValueError(
            "Bit-plane QK requires structured K and quantized current Native attention"
        )
    if any(t.get("decoder") != "native_unpack_lut_v1" for t in result.attention_tiles):
        raise ValueError("Bit-plane QK requires Native V decoding")
    result = copy.deepcopy(result)
    graph = result.model.graph
    sg = Subgraph()
    # Raw shared constant bytes prevent QAIRT float fallback from silently
    # rounding beta to FP16. This is not per-token metadata.
    beta = sg.const(
        "tq_bitplane_beta_f32_bytes",
        np.asarray(load_parameters()["beta"], "<f4").view(np.uint8).reshape(1, 1, 4, 4),
    )
    ios = {io.layer: io for io in result.codec_io if io.kind == "key"}
    layers = {t["layer"]: t for t in result.attention_tiles}
    sources: dict[tuple[int, int, int], tuple[str, str, int]] = {}
    current_scores: dict[tuple[int, int, int], str] = {}
    current_sources: dict[tuple[int, int], tuple[str, str]] = {}
    replacements = {}

    def score(
        packed: str, scale: str, query: str, prefix: str, seq: int, tokens: int
    ) -> str:
        query16, output16, output = (
            prefix + "query16",
            prefix + "score16",
            prefix + "score",
        )
        sg.node("Cast", [query], [query16], to=TensorProto.FLOAT16)
        sg.node(
            BITPLANE_OP,
            [packed, scale, query16, beta],
            [output16],
            domain=NATIVE_DOMAIN,
        )
        sg.node("Cast", [output16], [output], to=TensorProto.FLOAT)
        graph.value_info.append(
            helper.make_tensor_value_info(
                output16, TensorProto.FLOAT16, [1, 1, seq, tokens]
            )
        )
        return output

    for node in graph.node:
        match = re.fullmatch(
            r"tq_attn_(\d+)_tile(\d+)_head(\d+)_q(\d+)_score", node.output[0]
        )
        if not match:
            continue
        layer, start, head, group = map(int, match.groups())
        if node.op_type != "MatMul":
            raise ValueError("Unexpected QK score producer")
        io, info = ios[layer], layers[layer]
        tile = next(t for t in info["tiles"] if t["start"] == start)
        key = layer, start, head
        if key not in sources:
            prefix = f"tq_bitplane_{layer}_tile{start}_head{head}_"
            inputs = []
            for kind, past in (
                ("packed", io.packed_in),
                ("scale", io.norm_in),
            ):
                past_head = _slice(
                    sg, past, prefix + kind + "_past_head", head, head + 1, 0
                )
                past_tile = _slice(
                    sg, past_head, prefix + kind + "_past", start, tile["stop"], 2
                )
                if kind == "scale":
                    sg.node(
                        "Cast", [past_tile], [past_tile + "16"], to=TensorProto.FLOAT16
                    )
                    past_tile += "16"
                inputs.append(past_tile)
            sources[key] = inputs[0], inputs[1], tile["stop"] - start
        if (layer, head) not in current_sources:
            prefix = f"tq_bitplane_{layer}_head{head}_current_"
            packed = _slice(sg, io.packed_out, prefix + "packed", head, head + 1, 0)
            scale = _slice(sg, io.norm_out, prefix + "scale", head, head + 1, 0)
            sg.node("Cast", [scale], [scale + "16"], to=TensorProto.FLOAT16)
            current_sources[layer, head] = packed, scale + "16"
        if (layer, head, group) not in current_scores:
            packed, scale = current_sources[layer, head]
            current_scores[layer, head, group] = score(
                packed,
                scale,
                node.input[0],
                f"tq_attn_{layer}_head{head}_q{group}_current_bitplane_",
                io.new_tokens,
                io.new_tokens,
            )
        packed, scale, tokens = sources[key]
        prefix = node.output[0] + "_bitplane_"
        past_score = score(packed, scale, node.input[0], prefix, io.new_tokens, tokens)
        # HTP does not support raw UINT8 Concat. Compute past/current scores
        # separately, then concatenate at the original score dtype/encoding boundary.
        # Current QK is shared across tiles, and neither path restores K.
        replacements[node.output[0]] = [
            helper.make_node(
                "Concat",
                [past_score, current_scores[layer, head, group]],
                list(node.output),
                name=node.name,
                axis=3,
            ),
        ]
    expected = sum(t["query_heads"] * len(t["tiles"]) for t in layers.values())
    if len(replacements) != expected:
        raise ValueError(
            f"Incomplete Bit-plane QK replacement: {len(replacements)} != {expected}"
        )
    nodes = [
        replacement
        for node in graph.node
        for replacement in replacements.get(node.output[0], [node])
    ]
    del graph.node[:]
    graph.node.extend(nodes + sg.nodes)
    existing = {t.name for t in graph.initializer}
    graph.initializer.extend(
        t for name, t in sg.initializers.items() if name not in existing
    )
    _topological_sort(graph)
    _prune(result.model)
    if any(
        n.op_type == "Decode4"
        and n.input[2] == "tq_native_key_structured_centroids_fp16"
        for n in graph.node
    ):
        raise ValueError("Restored structured K is still live")
    for info in result.attention_tiles:
        info["qk"] = "native_bitplane_v2"
        for tile in info["tiles"]:
            tile["key_restored"] = None
            tile["attention_concats"]["key"] = []
    for info in result.current_kv_attention:
        if info["kind"] == "key":
            info.update(
                decoder="native_bitplane_qk", restored=None, attention_concats=[]
            )
    live = {v.name for v in graph.input} | {o for n in graph.node for o in n.output}
    result.encodings["activation_encodings"] = [
        e for e in result.encodings["activation_encodings"] if e["name"] in live
    ]
    return result


def bitplane_reference(
    packed: np.ndarray, scale: np.ndarray, query: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """CPU correctness oracle only; never attached to a deployment graph."""
    indices = np.stack((packed >> 4, packed & 15), axis=-1).reshape(
        *packed.shape[:-1], 128
    )
    score = np.zeros((*query.shape[:-1], packed.shape[-2]), dtype=np.float32)
    for m in range(4):
        signs = (2 * ((indices.astype(np.int32) >> m) & 1) - 1).astype(np.float32)
        score += np.float32(beta[m]) * (
            query.astype(np.float32) @ signs.swapaxes(-1, -2)
        )
    return (score * scale.astype(np.float32).swapaxes(-1, -2)).astype(np.float16)
