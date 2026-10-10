# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Default QHPI decoder for k4_v4_scaled exports; the format-2 cache ABI is unchanged."""

from __future__ import annotations

import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from qai_hub_models.models.templates.llm.turboquant.config import TurboQuantConfig
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import SurgeryResult
from qai_hub_models.models.templates.llm.turboquant.reference import load_codebook
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import _prune

NATIVE_DOMAIN = "turboquant"
NATIVE_OP = "Decode4"


def native_op(config: TurboQuantConfig, kind: str) -> str:
    bits = 4 if config.qjl and kind == "key" else getattr(config, kind).bits
    return f"Decode{bits}"


def native_table(config: TurboQuantConfig, kind: str) -> str:
    if config.qjl and kind == "key":
        return "tq_native_key3_centroids_fp16"
    bits = getattr(config, kind).bits
    return (
        "tq_native_centroids_fp16" if bits == 4 else f"tq_native_b{bits}_centroids_fp16"
    )


def use_native_decoder(
    result: SurgeryResult, config: TurboQuantConfig
) -> SurgeryResult:
    """Replace each rotated KV tile decoder, keeping QK/softmax/AV unchanged.

    Explicit FP16 casts preserve ONNX's surrounding float32 type contract;
    QAIRT's float16 fallback eliminates redundant conversion at deployment.
    This is a decoder kernel, not fused packed attention.
    """
    if (
        not config.precomputed_norm
        or config.norm_dtype != "float16"
        or config.block_size != 128
        or config.key.bits not in (2, 3, 4, 5, 6)
        or config.value.bits not in (2, 3, 4, 5, 6)
        or not result.attention_tiles
        or any(
            t["strategy"] != "rotated_precomputed_scale" for t in result.attention_tiles
        )
        or any(t.get("decoder") for t in result.attention_tiles)
    ):
        raise ValueError(
            "Native decoder requires rotated tiled attention and scaled 2..6-bit KV, D=128"
        )
    result = copy.deepcopy(result)
    graph = result.model.graph
    tables = set()
    for kind in ("value", "key"):
        table_name = native_table(config, kind)
        if table_name in tables:
            continue
        tables.add(table_name)
        values = load_codebook(getattr(config, kind).bits, 128)
        if config.qjl and kind == "key":
            values = np.tile(values, 2)
        graph.initializer.append(
            numpy_helper.from_array(
                values.astype(np.float16).reshape(1, 1, 1, -1),
                table_name,
            )
        )
    replacements = {}
    for layer in result.attention_tiles:
        layer["decoder"] = "native_unpack_lut_v1"
        for tile in layer["tiles"]:
            for kind in ("key", "value"):
                prefix = f"tq_{kind}_{layer['layer']}_tile{tile['start']}_"
                restored = prefix + "restored"
                output16 = prefix + "native_fp16"
                scale16 = prefix + "native_scale_fp16"
                replacements[restored] = [
                    helper.make_node(
                        "Cast",
                        [prefix + "norm"],
                        [scale16],
                        name=scale16,
                        to=TensorProto.FLOAT16,
                    ),
                    helper.make_node(
                        native_op(config, kind),
                        [
                            prefix + "packed",
                            scale16,
                            native_table(config, kind),
                        ],
                        [output16],
                        domain=NATIVE_DOMAIN,
                        name=output16,
                    ),
                    helper.make_node(
                        "Cast",
                        [output16],
                        [restored],
                        name=restored,
                        to=TensorProto.FLOAT,
                    ),
                ]
                graph.value_info.append(
                    helper.make_tensor_value_info(
                        output16,
                        TensorProto.FLOAT16,
                        [layer["kv_heads"], 1, tile["stop"] - tile["start"], 128],
                    )
                )
    nodes = []
    replaced = set()
    for node in graph.node:
        if node.output[0] in replacements:
            nodes.extend(replacements[node.output[0]])
            replaced.add(node.output[0])
        else:
            nodes.append(node)
    if replaced != set(replacements):
        raise ValueError("Missing tile decoder outputs during native replacement")
    del graph.node[:]
    graph.node.extend(nodes)
    if not any(op.domain == NATIVE_DOMAIN for op in result.model.opset_import):
        result.model.opset_import.append(helper.make_opsetid(NATIVE_DOMAIN, 1))
    _prune(result.model)
    live = {v.name for v in graph.input} | {o for n in graph.node for o in n.output}
    result.encodings["activation_encodings"] = [
        e for e in result.encodings["activation_encodings"] if e["name"] in live
    ]
    return result


def with_reference_decoder(model: onnx.ModelProto) -> onnx.ModelProto:
    """Attach a CPU-test-only ONNX function; never use this model for conversion.

    The deployed op remains opaque to the converter. This function expresses
    MSB-first unpack and FP16 LUT arithmetic independently of the DSP kernel.
    """
    result = copy.deepcopy(model)
    nodes = []

    def const(name: str, array: np.ndarray) -> None:
        nodes.append(
            helper.make_node(
                "Constant", [], [name], value=numpy_helper.from_array(array)
            )
        )

    const("sixteen", np.array(16, dtype=np.int32))
    const("axis4", np.array([4], dtype=np.int64))
    const("expand", np.array([1, 1, 1, 2], dtype=np.int64))
    const("flat_shape", np.array([-1], dtype=np.int64))
    nodes.extend(
        [
            helper.make_node("Cast", ["packed"], ["bytes_i32"], to=TensorProto.INT32),
            helper.make_node("Div", ["bytes_i32", "sixteen"], ["hi"]),
            helper.make_node("Mod", ["bytes_i32", "sixteen"], ["lo"]),
            helper.make_node("Unsqueeze", ["hi", "axis4"], ["hi5"]),
            helper.make_node("Unsqueeze", ["lo", "axis4"], ["lo5"]),
            helper.make_node("Concat", ["hi5", "lo5"], ["interleaved"], axis=4),
            helper.make_node("Shape", ["packed"], ["packed_shape"]),
            helper.make_node("Mul", ["packed_shape", "expand"], ["out_shape"]),
            helper.make_node("Reshape", ["interleaved", "out_shape"], ["indices"]),
            helper.make_node("Reshape", ["centroids", "flat_shape"], ["lut"]),
            helper.make_node("Gather", ["lut", "indices"], ["selected"], axis=0),
            helper.make_node("Mul", ["selected", "scale"], ["decoded"]),
        ]
    )
    result.functions.append(
        helper.make_function(
            NATIVE_DOMAIN,
            NATIVE_OP,
            ["packed", "scale", "centroids"],
            ["decoded"],
            nodes,
            [helper.make_opsetid("", 17)],
        )
    )
    for bits in (2, 3, 5, 6):
        op = f"Decode{bits}"
        if not any(
            n.domain == NATIVE_DOMAIN and n.op_type == op for n in result.graph.node
        ):
            continue
        nodes = []
        # Per-coordinate byte/shift constants express the stream independently
        # of the deployed HVX and encoder regrouping implementations.
        position = np.arange(128, dtype=np.int64) * bits
        const("offset", position // 8)
        const("next_offset", np.minimum(position // 8 + 1, 16 * bits - 1))
        const("shift", (2 ** (16 - bits - position % 8)).astype(np.int32))
        const("mask_modulus", np.array(1 << bits, np.int32))
        const("byte_base", np.array(256, np.int32))
        const("flat_shape", np.array([-1], np.int64))
        nodes.extend(
            [
                helper.make_node("Cast", ["packed"], ["bytes"], to=TensorProto.INT32),
                helper.make_node("Gather", ["bytes", "offset"], ["first"], axis=3),
                helper.make_node("Gather", ["bytes", "next_offset"], ["next"], axis=3),
                helper.make_node("Mul", ["first", "byte_base"], ["upper"]),
                helper.make_node("Add", ["upper", "next"], ["word"]),
                helper.make_node("Div", ["word", "shift"], ["shifted"]),
                helper.make_node("Mod", ["shifted", "mask_modulus"], ["indices"]),
                helper.make_node("Reshape", ["centroids", "flat_shape"], ["lut"]),
                helper.make_node("Gather", ["lut", "indices"], ["selected"], axis=0),
                helper.make_node("Mul", ["selected", "scale"], ["decoded"]),
            ]
        )
        result.functions.append(
            helper.make_function(
                NATIVE_DOMAIN,
                op,
                ["packed", "scale", "centroids"],
                ["decoded"],
                nodes,
                [helper.make_opsetid("", 17)],
            )
        )
    return result
