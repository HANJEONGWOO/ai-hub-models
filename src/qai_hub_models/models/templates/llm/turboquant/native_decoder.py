# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Opt-in QHPI decoder replacement; the format-2 cache ABI is unchanged."""

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
        or config.key.bits != 4
        or config.value.bits != 4
        or not result.attention_tiles
        or any(
            t["strategy"] != "rotated_precomputed_scale" for t in result.attention_tiles
        )
        or any(t.get("decoder") for t in result.attention_tiles)
    ):
        raise ValueError(
            "Native Decode4 requires rotated tiled attention and 4-bit format-2 KV, D=128"
        )
    result = copy.deepcopy(result)
    graph = result.model.graph
    table_name = "tq_native_centroids_fp16"
    graph.initializer.append(
        numpy_helper.from_array(
            load_codebook(4, 128).astype(np.float16).reshape(1, 1, 1, 16), table_name
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
                        NATIVE_OP,
                        [prefix + "packed", scale16, table_name],
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
    return result
