# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Two-pass tiled KV restore/attention for exported single-head Qwen graphs.

K tiles feed QK immediately; only scores are concatenated for the unchanged
global masked softmax. V tiles feed partial AV products, which are summed.
The opt-in rotated variant moves inverse rotations from cached KV to Q/output
and uses format-2 effective scales without decode-time norm correction.
No full restored K/V tensor is retained. This is graph tiling, not a claim
that QAIRT fuses the resulting operations into a single kernel.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from qai_hub_models.models.templates.llm.turboquant.config import TurboQuantConfig
from qai_hub_models.models.templates.llm.turboquant.export import (
    Subgraph,
    _rotation_name,
    decode_subgraph,
)
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    CodecTensorIO,
    SurgeryResult,
    _GraphIndex,
    regrid_encoding,
)


@dataclass
class _Head:
    head: int
    key: onnx.NodeProto
    value: onnx.NodeProto
    products: list[tuple[onnx.NodeProto, onnx.NodeProto]]


def _attribute(node: onnx.NodeProto, name: str) -> Any:
    return next(helper.get_attribute_value(a) for a in node.attribute if a.name == name)


def _constant(model: onnx.ModelProto, index: _GraphIndex, name: str) -> np.ndarray:
    for tensor in model.graph.initializer:
        if tensor.name == name:
            return numpy_helper.to_array(tensor)
    node = index.producer.get(name)
    if node is not None and node.op_type == "Constant":
        value = helper.get_attribute_value(node.attribute[0])
        return (
            numpy_helper.to_array(value)
            if isinstance(value, onnx.TensorProto)
            else np.asarray(value)
        )
    raise ValueError(f"Expected a constant slice parameter: {name}.")


def _head_concat(
    model: onnx.ModelProto, index: _GraphIndex, name: str, restored: str, axis: int
) -> tuple[int, onnx.NodeProto]:
    node = index.producer[name]
    if (
        node.op_type != "Concat"
        or len(node.input) != 2
        or _attribute(node, "axis") != axis
    ):
        raise ValueError(f"Unsupported attention cache consumer: {name}.")
    sliced = index.producer[node.input[0]]
    if sliced.op_type != "Slice" or sliced.input[0] != restored:
        raise ValueError(f"Expected a direct per-head slice before {name}.")
    starts, ends, axes = (
        _constant(model, index, s).reshape(-1).tolist() for s in sliced.input[1:4]
    )
    if len(starts) != 1 or axes != [0] or ends != [starts[0] + 1]:
        raise ValueError(f"Expected a single-head slice before {name}.")
    if len(sliced.input) == 5 and _constant(model, index, sliced.input[4]).tolist() != [
        1
    ]:
        raise ValueError(f"Unsupported slice step before {name}.")
    return int(starts[0]), node


def _only_consumer(index: _GraphIndex, name: str, op: str) -> onnx.NodeProto:
    nodes = index.consumers.get(name, [])
    if len(nodes) != 1 or nodes[0].op_type != op:
        raise ValueError(f"Expected one {op} consumer of {name}.")
    return nodes[0]


def _heads(
    result: SurgeryResult, index: _GraphIndex, layer: int, count: int
) -> list[_Head]:
    paths = {p.kind: p for p in result.paths if p.layer == layer}
    concats: dict[str, dict[int, onnx.NodeProto]] = {}
    for kind, axis in (("key", 3), ("value", 2)):
        restored = f"tq_{kind}_{layer}_restored" + ("_hub" if kind == "key" else "")
        entries = [
            _head_concat(result.model, index, n, restored, axis)
            for n in paths[kind].kept_consumers
        ]
        concats[kind] = dict(entries)
        if len(entries) != count or set(concats[kind]) != set(range(count)):
            raise ValueError(
                f"Layer {layer}: not every {kind} head has exactly one attention path."
            )
    heads = []
    for head in range(count):
        key, value = concats["key"][head], concats["value"][head]
        products = []
        for qk in index.consumers[key.output[0]]:
            if qk.op_type != "MatMul" or qk.input[1] != key.output[0]:
                raise ValueError(f"Unsupported QK consumer: {qk.name}.")
            add = _only_consumer(index, qk.output[0], "Add")
            softmax = _only_consumer(index, add.output[0], "Softmax")
            if _attribute(softmax, "axis") not in (-1, 3):
                raise ValueError("Attention softmax must reduce the context axis.")
            av = _only_consumer(index, softmax.output[0], "MatMul")
            if list(av.input) != [softmax.output[0], value.output[0]]:
                raise ValueError(f"Mismatched QK/AV head in layer {layer}.")
            products.append((qk, av))
        if not products or len(index.consumers[value.output[0]]) != len(products):
            raise ValueError(f"Unsupported V consumers in layer {layer}.")
        heads.append(_Head(head, key, value, products))
    return heads


def _slice(sg: Subgraph, src: str, dst: str, start: int, stop: int, axis: int) -> str:
    sg.node(
        "Slice", [src, sg.shape([start]), sg.shape([stop]), sg.shape([axis])], [dst]
    )
    return dst


def _restore_tile(
    sg: Subgraph,
    io: CodecTensorIO,
    config: TurboQuantConfig,
    start: int,
    stop: int,
    rotated: bool = False,
) -> str:
    prefix = f"tq_{io.kind}_{io.layer}_tile{start}_"
    packed = _slice(sg, io.packed_in, prefix + "packed", start, stop, 2)
    norm = _slice(sg, io.norm_in, prefix + "norm", start, stop, 2)
    restored = prefix + "restored"
    sg.extend(
        decode_subgraph(
            config,
            getattr(config, io.kind),
            packed,
            norm,
            restored,
            (io.num_kv_heads, 1),
            stop - start,
            prefix + "dec_",
            rotated=rotated,
        )
    )
    if io.kind == "key":
        sg.node("Transpose", [restored], [prefix + "restored_hub"], perm=[0, 1, 3, 2])
        return prefix + "restored_hub"
    return restored


def _prune(model: onnx.ModelProto) -> None:
    graph = model.graph
    needed = {v.name for v in graph.output}
    kept = []
    for node in reversed(graph.node):
        if needed.intersection(node.output):
            needed.update(node.input)
            kept.append(node)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    tensors = [t for t in graph.initializer if t.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(tensors)
    infos = [v for v in graph.value_info if v.name in needed]
    del graph.value_info[:]
    graph.value_info.extend(infos)
    available = (
        {v.name for v in graph.input} | {t.name for t in graph.initializer} | {""}
    )
    for node in graph.node:
        missing = set(node.input) - available
        if missing:
            raise ValueError(
                f"Unsupported attention node ordering: {node.name} reads {missing}."
            )
        available.update(node.output)


def tile_kv_attention(
    result: SurgeryResult,
    config: TurboQuantConfig,
    tile_tokens: int,
    *,
    rotated: bool = False,
) -> SurgeryResult:
    """Return a tiled copy; reject unsupported attention patterns instead of falling back."""
    if not config.key.is_polar or not config.value.is_polar:
        raise ValueError("Tiled attention requires both K and V to use the codec.")
    if tile_tokens < 1 or not result.codec_io:
        raise ValueError("Tiled attention needs a positive tile size and codec I/O.")
    if rotated and not config.precomputed_norm:
        raise ValueError(
            "Rotated attention requires the precomputed-scale cache format."
        )
    result = copy.deepcopy(result)
    graph = result.model.graph
    index = _GraphIndex(graph)
    acts = {e["name"]: e for e in result.encodings["activation_encodings"]}
    added_encodings: dict[str, dict[str, Any]] = {}
    all_sg = Subgraph()
    insertions: dict[str, list[onnx.NodeProto]] = {}
    removed: set[str] = set()
    positions = {n.output[0]: i for i, n in enumerate(graph.node)}

    def encoding(src: str, dst: str) -> None:
        if src not in acts:
            raise ValueError(f"Missing calibrated attention encoding: {src}.")
        added_encodings[dst] = {**copy.deepcopy(acts[src]), "name": dst}

    for layer in sorted({io.layer for io in result.codec_io}):
        ios = {io.kind: io for io in result.codec_io if io.layer == layer}
        key_io, value_io = ios["key"], ios["value"]
        past, seq = key_io.past_tokens, key_io.new_tokens
        if tile_tokens >= past and not rotated:
            raise ValueError(
                f"Tile size {tile_tokens} does not tile the {past}-token cache."
            )
        heads = _heads(result, index, layer, key_io.num_kv_heads)
        pairs = [(qk, av) for head in heads for qk, av in head.products]
        keys, values = Subgraph(), Subgraph()
        score_chunks: dict[str, list[str]] = {qk.output[0]: [] for qk, _ in pairs}
        accumulated: dict[str, str] = {}
        tile_report: list[dict[str, Any]] = []
        queries: dict[str, str] = {}
        current: dict[tuple[int, str], str] = {}
        if rotated:
            for head in heads:
                prefix = f"tq_attn_{layer}_head{head.head}_"
                for kind, sg, original in (
                    ("key", keys, head.key),
                    ("value", values, head.value),
                ):
                    src = original.input[1]
                    if src in acts and acts[src]["bw"] == 8:
                        acts[src] = regrid_encoding(acts[src])
                    if kind == "key":
                        sg.node(
                            "Transpose",
                            [src],
                            [prefix + "current_key"],
                            perm=[0, 1, 3, 2],
                        )
                        src = prefix + "current_key"
                    dest = prefix + kind + "_rotated"
                    sg.node(
                        "MatMul",
                        [src, _rotation_name(sg, config, getattr(config, kind), True)],
                        [dest],
                    )
                    if kind == "key":
                        sg.node("Transpose", [dest], [dest + "_hub"], perm=[0, 1, 3, 2])
                        dest += "_hub"
                    current[head.head, kind] = dest
                for group, (qk, _) in enumerate(head.products):
                    dest = prefix + f"q{group}_rotated"
                    keys.node(
                        "MatMul",
                        [qk.input[0], _rotation_name(keys, config, config.key, True)],
                        [dest],
                    )
                    queries[qk.output[0]] = dest
        for start in range(0, past, tile_tokens):
            stop = min(start + tile_tokens, past)
            last = stop == past
            k = _restore_tile(keys, key_io, config, start, stop, rotated)
            v = _restore_tile(values, value_io, config, start, stop, rotated)
            tile_cats: dict[str, list[str]] = {"key": [], "value": []}
            for head in heads:
                prefix = f"tq_attn_{layer}_tile{start}_head{head.head}_"
                cats = {}
                for kind, sg, restored, original, axis in (
                    ("key", keys, k, head.key, 3),
                    ("value", values, v, head.value, 2),
                ):
                    sliced = _slice(
                        sg,
                        restored,
                        prefix + kind + "_past",
                        head.head,
                        head.head + 1,
                        0,
                    )
                    # QAIRT needs the Slice grid too; tagging only Concat promotes KV to 16-bit.
                    if not rotated:
                        encoding(original.input[0], sliced)
                    cat = prefix + kind + "_cat"
                    # The legacy path needs mixed-input Concat for its int8 boundary.
                    present = current[head.head, kind] if rotated else original.input[1]
                    sg.node("Concat", [sliced, present], [cat], axis=axis)
                    if not rotated:
                        encoding(original.output[0], cat)
                    cats[kind] = cat
                    tile_cats[kind].append(cat)
                for group, (qk, av) in enumerate(head.products):
                    stem = prefix + f"q{group}_"
                    score = stem + "score"
                    query = queries[qk.output[0]] if rotated else qk.input[0]
                    keys.node("MatMul", [query, cats["key"]], [score])
                    encoding(qk.output[0], score)
                    if not last:
                        score = _slice(
                            keys, score, stem + "past_score", 0, stop - start, 3
                        )
                        encoding(qk.output[0], score)
                    score_chunks[qk.output[0]].append(score)
                    prob = _slice(
                        values,
                        av.input[0],
                        stem + "prob",
                        start,
                        past + seq if last else stop,
                        3,
                    )
                    encoding(av.input[0], prob)
                    if not last:
                        zeros = values.const(
                            f"tq_zero_prob_ar{seq}",
                            np.zeros((1, 1, seq, seq), dtype=np.float32),
                        )
                        padded = stem + "prob_padded"
                        values.node("Concat", [prob, zeros], [padded], axis=3)
                        encoding(av.input[0], padded)
                        prob = padded
                    partial = stem + "partial"
                    values.node("MatMul", [prob, cats["value"]], [partial])
                    if not rotated:
                        encoding(av.output[0], partial)
                    previous = accumulated.get(av.output[0])
                    if previous is None:
                        accumulated[av.output[0]] = partial
                    else:
                        total = av.output[0] if last and not rotated else stem + "sum"
                        values.node("Add", [previous, partial], [total])
                        accumulated[av.output[0]] = total
            tile_report.append(
                {
                    "start": start,
                    "stop": stop,
                    "key_restored": k,
                    "value_restored": v,
                    "attention_concats": tile_cats,
                }
            )
        for qk, _ in pairs:
            keys.node("Concat", score_chunks[qk.output[0]], list(qk.output), axis=3)
        if rotated:
            for _, av in pairs:
                values.node(
                    "MatMul",
                    [
                        accumulated[av.output[0]],
                        _rotation_name(values, config, config.value, False),
                    ],
                    list(av.output),
                )
        qks, avs = [qk for qk, _ in pairs], [av for _, av in pairs]
        insertions[max(qks, key=lambda n: positions[n.output[0]]).output[0]] = (
            keys.nodes
        )
        insertions[max(avs, key=lambda n: positions[n.output[0]]).output[0]] = (
            values.nodes
        )
        removed.update(n.output[0] for n in qks + avs)
        all_sg.extend(keys)
        all_sg.extend(values)
        result.attention_tiles.append(
            {
                "layer": layer,
                "tile_tokens": tile_tokens,
                "past_tokens": past,
                "new_tokens": seq,
                "kv_heads": key_io.num_kv_heads,
                "query_heads": len(pairs),
                "strategy": "rotated_precomputed_scale"
                if rotated
                else "two_pass_tiled_global_softmax",
                "tiles": tile_report,
            }
        )

    rebuilt = []
    for node in graph.node:
        if node.output[0] not in removed:
            rebuilt.append(node)
        rebuilt.extend(insertions.get(node.output[0], []))
    del graph.node[:]
    graph.node.extend(rebuilt)
    existing = {t.name for t in graph.initializer}
    graph.initializer.extend(
        t for n, t in all_sg.initializers.items() if n not in existing
    )
    _prune(result.model)
    produced = {o for n in graph.node for o in n.output}
    for io in result.codec_io:
        if f"tq_{io.kind}_{io.layer}_restored" in produced:
            raise ValueError(
                "A full KV restore is still live; refusing a partial tiling rewrite."
            )
    live = produced | {v.name for v in graph.input}
    result.encodings["activation_encodings"] = [
        e
        for e in list(acts.values()) + list(added_encodings.values())
        if e["name"] in live
    ]
    return result
