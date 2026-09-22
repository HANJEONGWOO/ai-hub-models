# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Orthogonal QJL oracle and K-only tiled score correction.

Matches the projection/sign/coefficient conventions in turboquant_plus/qjl.py
(reference commit 7f601a135735842a7f12b6bf861561154c410ff4). Unlike the Polar
rotation, QJL does NOT force determinant +1. No MMSE shrinkage is applied.
The deployment residual uses the actually stored FP16 MSE reconstruction.
"""

from __future__ import annotations

import copy
import re
from functools import cache

import numpy as np
import onnx
from onnx import TensorProto, helper

from qai_hub_models.models.templates.llm.turboquant.config import TurboQuantConfig
from qai_hub_models.models.templates.llm.turboquant.export import (
    Subgraph,
    _rotation_name,
    encode_subgraph,
)
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import SurgeryResult
from qai_hub_models.models.templates.llm.turboquant.native_decoder import (
    NATIVE_DOMAIN,
    NATIVE_OP,
)
from qai_hub_models.models.templates.llm.turboquant.packing import (
    pack_indices,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.reference import (
    PolarQuantReference,
    load_codebook,
)
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    _prune,
    _slice,
)


@cache
def projection(d: int, seed: int) -> np.ndarray:
    """Haar orthogonal S, matching the reference QR column-sign convention."""
    if d < 1:
        raise ValueError("QJL dimension must be positive.")
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    q = q * np.sign(np.diag(r))[None, :]
    if np.linalg.norm(q @ q.T - np.eye(d), ord="fro") >= 1e-10:
        raise ValueError("QJL projection is not orthogonal.")
    q.setflags(write=False)
    return q


def quantize_residual(
    residual: np.ndarray, seed: int = 1042
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(residual, dtype=np.float64)
    if not np.all(np.isfinite(residual)):
        raise ValueError("QJL residual contains NaN or Inf.")
    signs = np.where(
        residual @ projection(residual.shape[-1], seed).T >= 0, 1, -1
    ).astype(np.int8)
    return signs, np.linalg.norm(residual, axis=-1, keepdims=True)


def dequantize_residual(
    signs: np.ndarray, norms: np.ndarray, seed: int = 1042
) -> np.ndarray:
    d = signs.shape[-1]
    return (signs @ projection(d, seed)) * norms * (np.sqrt(np.pi / 2) / np.sqrt(d))


class QJLKeyReference:
    """Packed K3+1 oracle with the same FP16 LUT/product rounding as Native Decode4."""

    def __init__(self, config: TurboQuantConfig) -> None:
        if not config.qjl:
            raise ValueError("QJLKeyReference requires a QJL profile.")
        self.config = config
        self.base = PolarQuantReference(
            config.key,
            config.block_size,
            config.rotation,
            config.norm_correction,
            precomputed_norm=True,
        )
        self.coefficient = np.sqrt(np.pi / 2) / np.sqrt(config.block_size)

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices, scales = self.base.encode(x)
        scales = scales.astype(np.float16)
        if not np.all(np.isfinite(scales)):
            raise OverflowError("QJL MSE scale exceeds float16 range.")
        rotated = (self.base.centroids[indices].astype(np.float16) * scales).astype(
            np.float16
        )
        residual = x - self.base.rotation.inverse(rotated)
        signs, norms = quantize_residual(residual, self.config.key.seed + 1000)
        qscale = (norms * self.coefficient).astype(np.float16)
        if not np.all(np.isfinite(qscale)):
            raise OverflowError("QJL residual scale exceeds float16 range.")
        return (
            pack_indices(indices | ((signs > 0).astype(np.uint8) << 3), 4),
            scales,
            qscale,
        )

    def decode(
        self, packed: np.ndarray, scale: np.ndarray, qscale: np.ndarray
    ) -> np.ndarray:
        codes = unpack_indices(packed, 4, self.config.block_size)
        base = (
            self.base.centroids[codes & 7].astype(np.float16) * scale.astype(np.float16)
        ).astype(np.float16)
        signs = np.where(codes >= 8, 1, -1)
        return self.base.rotation.inverse(base) + (signs * qscale) @ projection(
            self.config.block_size, self.config.key.seed + 1000
        )


def _encoder(
    sg: Subgraph, config: TurboQuantConfig, layer: int, heads: int, seq: int, opset: int
) -> None:
    """Encode residual once on append; high nibble bit stores its projected sign."""
    p = f"tq_key_{layer}_"
    e, j = p + "enc_", p + "qjl_"
    d = config.block_size
    sg.node("Cast", [p + "scale_out"], [j + "scale16"], to=TensorProto.FLOAT16)
    # Use precisely the reader's LUT and rounded product, not a second affine
    # approximation of its centroids (which can differ under HTP FP16).
    sg.node(
        NATIVE_OP,
        [p + "packed_out_mse_only", j + "scale16", "tq_native_key3_centroids_fp16"],
        [j + "base_native_fp16"],
        domain=NATIVE_DOMAIN,
    )
    sg.node(
        "Cast", [j + "base_native_fp16"], [j + "base_restored"], to=TensorProto.FLOAT
    )
    sg.node(
        "Reshape", [j + "base_restored", sg.shape([1, heads, seq, d])], [j + "base"]
    )
    sg.node(
        "MatMul",
        [j + "base", _rotation_name(sg, config, config.key, False)],
        [j + "mse_key"],
    )
    sg.node(
        "Reshape",
        [p + "present_tokens_last", sg.shape([1, heads, seq, d])],
        [j + "source"],
    )
    sg.node("Sub", [j + "source", j + "mse_key"], [j + "residual"])
    matrix = sg.const(
        "tq_qjl_projection_t",
        projection(d, config.key.seed + 1000).T.astype(np.float32),
    )
    sg.node("MatMul", [j + "residual", matrix], [j + "projected"])
    zero = sg.const("tq_zero_f", np.array(0, dtype=np.float32))
    one = sg.const("tq_one_f", np.array(1, dtype=np.float32))
    sg.node("GreaterOrEqual", [j + "projected", zero], [j + "positive"])
    sg.node("Cast", [j + "positive"], [j + "positive_i32"], to=TensorProto.INT32)
    eight = sg.const("tq_qjl_eight_i32", np.array(8, dtype=np.int32))
    sg.node("Mul", [j + "positive_i32", eight], [j + "sign_bit"])
    sg.node("Reshape", [e + "index", sg.shape([1, heads, seq, d])], [j + "indices"])
    sg.node("Add", [j + "indices", j + "sign_bit"], [j + "codes"])
    for label, start in (("hi", 0), ("lo", 1)):
        sg.node(
            "Slice",
            [
                j + "codes",
                sg.shape([start]),
                sg.shape([d]),
                sg.shape([3]),
                sg.shape([2]),
            ],
            [j + label],
        )
    sixteen = sg.const("tq_sixteen_i32", np.array(16, dtype=np.int32))
    sg.node("Mul", [j + "hi", sixteen], [j + "hi16"])
    sg.node("Add", [j + "hi16", j + "lo"], [j + "bytes"])
    sg.node(
        "Reshape",
        [j + "bytes", sg.shape([heads, 1, seq, d // 2])],
        [j + "bytes_head_major"],
    )
    # HTP cannot transpose raw UINT8 tensors; establish the storage layout first.
    sg.node("Cast", [j + "bytes_head_major"], [p + "packed_out"], to=TensorProto.UINT8)
    # Overflow-safe norm; HTP Div also needs its divisor below 2**14.
    sg.node("Abs", [j + "residual"], [j + "abs"])
    if opset >= 18:
        sg.node("ReduceMax", [j + "abs", sg.shape([3])], [j + "max"], keepdims=1)
    else:
        sg.node("ReduceMax", [j + "abs"], [j + "max"], axes=[3], keepdims=1)
    threshold = sg.const("tq_prescale_threshold", np.array(256, dtype=np.float32))
    down = sg.const("tq_prescale_down", np.array(2**-8, dtype=np.float32))
    up = sg.const("tq_prescale_up", np.array(2**8, dtype=np.float32))
    sg.node("Greater", [j + "max", threshold], [j + "big"])
    sg.node("Where", [j + "big", down, one], [j + "pre"])
    sg.node("Where", [j + "big", up, one], [j + "post"])
    sg.node("Mul", [j + "residual", j + "pre"], [j + "rpre"])
    sg.node("Mul", [j + "max", j + "pre"], [j + "mpre"])
    sg.node("Greater", [j + "mpre", zero], [j + "nonzero"])
    sg.node("Where", [j + "nonzero", j + "mpre", one], [j + "safe"])
    sg.node("Div", [j + "rpre", j + "safe"], [j + "unit"])
    sg.node("Mul", [j + "unit", j + "unit"], [j + "sq"])
    sg.node("ReduceSum", [j + "sq", sg.shape([3])], [j + "sum"], keepdims=1)
    sg.node("Sqrt", [j + "sum"], [j + "length"])
    sg.node("Mul", [j + "length", j + "mpre"], [j + "norm_pre"])
    sg.node("Mul", [j + "norm_pre", j + "post"], [j + "norm"])
    coeff = sg.const(
        "tq_qjl_coefficient",
        np.array(np.sqrt(np.pi / 2) / np.sqrt(d), dtype=np.float32),
    )
    sg.node("Mul", [j + "norm", coeff], [j + "qscale"])
    sg.node(
        "Reshape", [j + "qscale", sg.shape([heads, 1, seq, 1])], [p + "qjlscale_out"]
    )


def add_qjl_attention(result: SurgeryResult, config: TurboQuantConfig) -> SurgeryResult:
    """Add K-only correction via (q S.T) @ (signs * residual_scale).T per tile.

    The existing Native Decode4 kernel handles both low-three-bit centroids and
    high-one-bit signs using different 16-entry LUTs; no new DSP binary is needed.
    This intermediate pass leaves current K uncompressed with zero correction;
    the default export's final current_attention pass replaces both branches.
    """
    if (
        not config.qjl
        or not config.norm_correction
        or not result.attention_tiles
        or any(
            t.get("decoder") != "native_unpack_lut_v1" or t.get("qjl")
            for t in result.attention_tiles
        )
    ):
        raise ValueError(
            "QJL requires native rotated tiles and norm correction, applied once."
        )
    result = copy.deepcopy(result)
    graph = result.model.graph
    sg, encoder = Subgraph(), Subgraph()
    before: dict[str, list[onnx.NodeProto]] = {}
    after: dict[str, list[onnx.NodeProto]] = {}
    layers = {t["layer"]: t for t in result.attention_tiles}
    query_counts = dict.fromkeys(layers, 0)
    score_counts = dict.fromkeys(layers, 0)
    matrix = sg.const(
        "tq_qjl_projection_t",
        projection(config.block_size, config.key.seed + 1000).T.astype(np.float32),
    )
    lut = sg.const(
        "tq_qjl_sign_lut_fp16",
        np.repeat(np.array([-1, 1], dtype=np.float16), 8).reshape(1, 1, 1, 16),
    )
    for layer, info in layers.items():
        p = f"tq_key_{layer}_"
        heads, past, seq = info["kv_heads"], info["past_tokens"], info["new_tokens"]
        info["qjl"] = {
            "scale_input": p + "qjlscale_in",
            "scale_output": p + "qjlscale_out",
            "mse_bits": 3,
            "sign_bits": 1,
        }
        graph.input.append(
            helper.make_tensor_value_info(
                p + "qjlscale_in", TensorProto.FLOAT, [heads, 1, past, 1]
            )
        )
        graph.output.append(
            helper.make_tensor_value_info(
                p + "qjlscale_out", TensorProto.FLOAT, [heads, 1, seq, 1]
            )
        )
        for node in graph.node:
            if p + "packed_out" in node.output:
                node.output[0] += "_mse_only"
                node.name += "_mse_only"
        opset = max(o.version for o in result.model.opset_import if not o.domain)
        _encoder(encoder, config, layer, heads, seq, opset)
        graph.value_info.append(
            helper.make_tensor_value_info(
                p + "qjl_base_native_fp16",
                TensorProto.FLOAT16,
                [heads, 1, seq, config.block_size],
            )
        )
        for tile in info["tiles"]:
            start, stop = tile["start"], tile["stop"]
            t = p + f"tile{start}_"
            j = t + "qjl_"
            begin = len(sg.nodes)
            _slice(sg, p + "qjlscale_in", j + "scale", start, stop, 2)
            sg.node("Cast", [j + "scale"], [j + "scale16"], to=TensorProto.FLOAT16)
            sg.node(
                NATIVE_OP,
                [t + "packed", j + "scale16", lut],
                [j + "decoded16"],
                domain=NATIVE_DOMAIN,
            )
            sg.node("Cast", [j + "decoded16"], [j + "decoded"], to=TensorProto.FLOAT)
            sg.node("Transpose", [j + "decoded"], [j + "hub"], perm=[0, 1, 3, 2])
            graph.value_info.append(
                helper.make_tensor_value_info(
                    j + "decoded16",
                    TensorProto.FLOAT16,
                    [heads, 1, stop - start, config.block_size],
                )
            )
            after[t + "restored"] = sg.nodes[begin:]
    for node in graph.node:
        query = re.fullmatch(r"tq_attn_(\d+)_head(\d+)_q(\d+)_rotated", node.output[0])
        if query:
            query_counts[int(query.group(1))] += 1
            begin = len(sg.nodes)
            sg.node("MatMul", [node.input[0], matrix], [node.output[0] + "_qjl"])
            after[node.output[0]] = sg.nodes[begin:]
        score = re.fullmatch(
            r"tq_attn_(\d+)_tile(\d+)_head(\d+)_q(\d+)_score", node.output[0]
        )
        if not score:
            continue
        layer, start, head, group = map(int, score.groups())
        score_counts[layer] += 1
        original = node.output[0]
        p = original + "_qjl_"
        begin = len(sg.nodes)
        residual = f"tq_key_{layer}_tile{start}_qjl_hub"
        _slice(sg, residual, p + "head", head, head + 1, 0)
        seq = layers[layer]["new_tokens"]
        zeros = sg.const(
            f"tq_qjl_zero_current_ar{seq}",
            np.zeros((1, 1, config.block_size, seq), dtype=np.float32),
        )
        sg.node("Concat", [p + "head", zeros], [p + "cat"], axis=3)
        q = f"tq_attn_{layer}_head{head}_q{group}_rotated_qjl"
        sg.node("MatMul", [q, p + "cat"], [p + "correction"])
        before[original] = sg.nodes[begin:]
        node.output[0] = original + "_mse"
        node.name = original + "_mse"
        begin = len(sg.nodes)
        sg.node("Add", [node.output[0], p + "correction"], [original])
        after[node.output[0]] = sg.nodes[begin:]
        before[node.output[0]] = before.pop(original)
    for layer, info in layers.items():
        if query_counts[layer] != info["query_heads"] or score_counts[layer] != info[
            "query_heads"
        ] * len(info["tiles"]):
            raise ValueError(f"Incomplete QJL attention rewrite for layer {layer}.")
    nodes = []
    for node in graph.node:
        nodes.extend(before.get(node.output[0], []))
        nodes.append(node)
        nodes.extend(after.get(node.output[0], []))
    nodes.extend(encoder.nodes)
    del graph.node[:]
    graph.node.extend(nodes)
    sg.extend(encoder)
    existing = {t.name for t in graph.initializer}
    graph.initializer.extend(t for n, t in sg.initializers.items() if n not in existing)
    _prune(result.model)
    live = {o for n in graph.node for o in n.output} | {v.name for v in graph.input}
    result.encodings["activation_encodings"] = [
        e for e in result.encodings["activation_encodings"] if e["name"] in live
    ]
    return result


def build_qjl_encode_model(
    config: TurboQuantConfig, tokens: int, heads: int = 8
) -> onnx.ModelProto:
    """Isolated append encoder, including actual Native reconstruction of its MSE stage."""
    if not config.qjl or tokens < 1 or heads < 1:
        raise ValueError("Need a QJL profile and positive token/head counts.")
    p = "tq_key_0_"
    src = p + "present_tokens_last"
    sg = encode_subgraph(
        config,
        config.key,
        src,
        p + "packed_out_mse_only",
        p + "scale_out",
        (heads, 1),
        tokens,
        p + "enc_",
    )
    sg.const(
        "tq_native_key3_centroids_fp16",
        np.tile(load_codebook(3, config.block_size), 2)
        .astype(np.float16)
        .reshape(1, 1, 1, 16),
    )
    _encoder(sg, config, 0, heads, tokens, 17)
    outputs = [
        helper.make_tensor_value_info(
            p + name + "_out", dtype, [heads, 1, tokens, width]
        )
        for name, dtype, width in (
            ("packed", TensorProto.UINT8, config.block_size // 2),
            ("scale", TensorProto.FLOAT, 1),
            ("qjlscale", TensorProto.FLOAT, 1),
        )
    ]
    graph = helper.make_graph(
        sg.nodes,
        f"qjl_encode_t{tokens}",
        [
            helper.make_tensor_value_info(
                src, TensorProto.FLOAT, [heads, 1, tokens, config.block_size]
            )
        ],
        outputs,
        list(sg.initializers.values()),
    )
    graph.value_info.append(
        helper.make_tensor_value_info(
            p + "qjl_base_native_fp16",
            TensorProto.FLOAT16,
            [heads, 1, tokens, config.block_size],
        )
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid(NATIVE_DOMAIN, 1),
        ],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    return model
