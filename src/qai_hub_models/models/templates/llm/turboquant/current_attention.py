# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Read current-token KV from the same packed outputs that are appended to cache.

Apply this final pass after tiling, Native replacement and optional QJL. The
encoder is shared with cache output, not duplicated for each query head/tile.
Both decode (AR1) and every token of a prefill chunk use the stored precision.
"""

from __future__ import annotations

import copy
import heapq
import re

import onnx
from onnx import TensorProto, helper

from qai_hub_models.models.templates.llm.turboquant.config import TurboQuantConfig
from qai_hub_models.models.templates.llm.turboquant.export import (
    Subgraph,
    decode_subgraph,
)
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    SurgeryResult,
    _GraphIndex,
)
from qai_hub_models.models.templates.llm.turboquant.native_decoder import (
    NATIVE_DOMAIN,
    NATIVE_OP,
)
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    _head_concat,
    _prune,
    _slice,
)


def _topological_sort(graph: onnx.GraphProto) -> None:
    """Move append encoders before attention without crossing layer dependencies."""
    nodes = list(graph.node)
    producers: dict[str, int] = {}
    external = (
        {v.name for v in graph.input} | {t.name for t in graph.initializer} | {""}
    )
    for i, node in enumerate(nodes):
        for output in node.output:
            if output in producers or output in external:
                raise ValueError(f"Duplicate graph tensor: {output}")
            producers[output] = i
    waiting = []
    consumers: dict[int, list[int]] = {}
    ready: list[int] = []
    for i, node in enumerate(nodes):
        missing = set(node.input) - external - producers.keys()
        if missing:
            raise ValueError(f"Missing graph inputs for {node.name}: {sorted(missing)}")
        parents = {producers[n] for n in node.input if n not in external}
        waiting.append(len(parents))
        if not parents:
            heapq.heappush(ready, i)
        for parent in parents:
            consumers.setdefault(parent, []).append(i)
    ordered = []
    while ready:
        i = heapq.heappop(ready)
        ordered.append(nodes[i])
        for child in consumers.get(i, []):
            waiting[child] -= 1
            if not waiting[child]:
                heapq.heappush(ready, child)
    if len(ordered) != len(nodes):
        raise ValueError("Current KV rewrite introduced a dependency cycle.")
    del graph.node[:]
    graph.node.extend(ordered)


def quantize_current_attention(
    result: SurgeryResult, config: TurboQuantConfig
) -> SurgeryResult:
    """Replace raw current KV branches with decode(cache outputs), without ABI changes."""
    if not result.codec_io or result.current_kv_attention:
        raise ValueError("Current KV attention requires codec I/O and is applied once.")
    result = copy.deepcopy(result)
    graph = result.model.graph
    index = _GraphIndex(graph)
    layers = {item["layer"]: item for item in result.attention_tiles}
    if config.qjl and (
        not layers or any(not item.get("qjl") for item in layers.values())
    ):
        raise ValueError("Apply QJL attention before current KV quantization.")
    sg = Subgraph()
    acts = {e["name"]: e for e in result.encodings["activation_encodings"]}

    def native(
        packed: str, scale: str, table: str, prefix: str, heads: int, seq: int
    ) -> str:
        sg.node("Cast", [scale], [prefix + "scale16"], to=TensorProto.FLOAT16)
        sg.node(
            NATIVE_OP,
            [packed, prefix + "scale16", table],
            [prefix + "native_fp16"],
            domain=NATIVE_DOMAIN,
        )
        sg.node(
            "Cast",
            [prefix + "native_fp16"],
            [prefix + "restored"],
            to=TensorProto.FLOAT,
        )
        graph.value_info.append(
            helper.make_tensor_value_info(
                prefix + "native_fp16",
                TensorProto.FLOAT16,
                [heads, 1, seq, config.block_size],
            )
        )
        return prefix + "restored"

    for io in result.codec_io:
        layer, kind, heads, seq = io.layer, io.kind, io.num_kv_heads, io.new_tokens
        info = layers.get(layer)
        rotated = bool(info and info["strategy"] == "rotated_precomputed_scale")
        use_native = bool(info and info.get("decoder") == "native_unpack_lut_v1")
        prefix = f"tq_{kind}_{layer}_current_"
        if use_native:
            restored = native(
                io.packed_out,
                io.norm_out,
                "tq_native_key3_centroids_fp16"
                if config.qjl and kind == "key"
                else "tq_native_centroids_fp16",
                prefix,
                heads,
                seq,
            )
        else:
            scale = io.norm_out
            # Match the host cache's stored scalar, including on CPU ONNX oracles.
            if config.norm_dtype == "float16":
                sg.node("Cast", [scale], [prefix + "scale16"], to=TensorProto.FLOAT16)
                sg.node(
                    "Cast",
                    [prefix + "scale16"],
                    [prefix + "scale"],
                    to=TensorProto.FLOAT,
                )
                scale = prefix + "scale"
            restored = prefix + "restored"
            sg.extend(
                decode_subgraph(
                    config,
                    getattr(config, kind),
                    io.packed_out,
                    scale,
                    restored,
                    (heads, 1),
                    seq,
                    prefix + "dec_",
                    rotated=rotated,
                )
            )
        if kind == "key":
            sg.node("Transpose", [restored], [prefix + "hub"], perm=[0, 1, 3, 2])
            restored = prefix + "hub"
        current = [
            _slice(sg, restored, prefix + f"head{head}", head, head + 1, 0)
            for head in range(heads)
        ]
        concats: list[tuple[int, onnx.NodeProto]] = []
        if info:
            for tile in info["tiles"]:
                names = tile["attention_concats"][kind]
                if len(names) != heads:
                    raise ValueError("Incomplete current KV head mapping.")
                concats.extend(
                    (head, index.producer[name]) for head, name in enumerate(names)
                )
        else:
            path = next(p for p in result.paths if p.layer == layer and p.kind == kind)
            past = f"tq_{kind}_{layer}_restored" + ("_hub" if kind == "key" else "")
            concats = [
                _head_concat(result.model, index, name, past, 3 if kind == "key" else 2)
                for name in path.kept_consumers
            ]
            if len(concats) != heads or {head for head, _ in concats} != set(
                range(heads)
            ):
                raise ValueError("Incomplete current KV attention consumers.")
        for head, cat in concats:
            if cat.op_type != "Concat" or len(cat.input) != 2:
                raise ValueError(f"Unsupported current KV consumer: {cat.name}")
            cat.input[1] = current[head]
            if not rotated:
                # Preserve the legacy attention int8 boundary on both inputs.
                source = cat.input[0]
                if source not in acts:
                    raise ValueError(f"Missing attention encoding: {source}")
                acts[current[head]] = {
                    **copy.deepcopy(acts[source]),
                    "name": current[head],
                }

        if config.qjl and kind == "key":
            residual = native(
                io.packed_out,
                f"tq_key_{layer}_qjlscale_out",
                "tq_qjl_sign_lut_fp16",
                prefix + "qjl_",
                heads,
                seq,
            )
            sg.node("Transpose", [residual], [prefix + "qjl_hub"], perm=[0, 1, 3, 2])
            residual_heads = [
                _slice(sg, prefix + "qjl_hub", prefix + f"qjl_head{h}", h, h + 1, 0)
                for h in range(heads)
            ]
            replaced = 0
            for node in graph.node:
                match = re.fullmatch(
                    rf"tq_attn_{layer}_tile\d+_head(\d+)_q\d+_score_qjl_cat",
                    node.output[0],
                )
                if match:
                    if node.op_type != "Concat" or len(node.input) != 2:
                        raise ValueError("Unsupported QJL current correction.")
                    node.input[1] = residual_heads[int(match.group(1))]
                    replaced += 1
            assert info is not None
            if replaced != info["query_heads"] * len(info["tiles"]):
                raise ValueError("Incomplete QJL current correction.")

        result.current_kv_attention.append(
            {
                "layer": layer,
                "kind": kind,
                "tokens": seq,
                "packed": io.packed_out,
                "scale": io.norm_out,
                "restored": restored,
                "decoder": "native" if use_native else "graph",
                "rotated": rotated,
                "qjl": config.qjl and kind == "key",
                "attention_concats": [cat.output[0] for _, cat in concats],
            }
        )
    graph.node.extend(sg.nodes)
    existing = {tensor.name for tensor in graph.initializer}
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
