# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Default QHPI decoder for k4_v4_scaled exports; the format-2 cache ABI is unchanged."""

from __future__ import annotations

import copy
import re

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from qai_hub_models.models.templates.llm.turboquant.config import TurboQuantConfig
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import SurgeryResult
from qai_hub_models.models.templates.llm.turboquant.reference import spec_codebook
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import _prune

NATIVE_DOMAIN = "turboquant"
NATIVE_OP = "Decode4"
STRUCTURED_OP = "Decode8Structured"
STRUCTURED_COEFFICIENTS = "tq_native_structured_k8_coefficients_fp16"
UNIFORM_OP = "Decode8Uniform"
UNIFORM_COEFFICIENTS = "tq_native_uniform_k8_coefficients_fp16"


def native_op(config: TurboQuantConfig, kind: str) -> str:
    if getattr(config, kind).quantizer == "uniform_k8_gaussian_v1":
        return UNIFORM_OP
    bits = 4 if config.qjl and kind == "key" else getattr(config, kind).bits
    return f"Decode{bits}"


def native_table(config: TurboQuantConfig, kind: str) -> str:
    if getattr(config, kind).quantizer == "uniform_k8_gaussian_v1":
        return UNIFORM_COEFFICIENTS
    if getattr(config, kind).codebook_variant:
        return "tq_native_converged_k8_centroids_fp16"
    if getattr(config, kind).quantizer != "lloyd_max":
        return "tq_native_structured_k8_centroids_fp16"
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
        or any(s.bits not in range(2, 9) for s in config.codecs)
        or not result.attention_tiles
        or any(
            t["strategy"] != "rotated_precomputed_scale" for t in result.attention_tiles
        )
        or any(t.get("decoder") for t in result.attention_tiles)
    ):
        raise ValueError(
            "Native decoder requires rotated tiled attention and scaled 2..8-bit KV, D=128"
        )
    result = copy.deepcopy(result)
    graph = result.model.graph
    tables = set()
    for layer_id, kind in (
        (t["layer"], k) for t in result.attention_tiles for k in ("value", "key")
    ):
        layer_config = config.for_layer(layer_id)
        table_name = native_table(layer_config, kind)
        if table_name in tables:
            continue
        tables.add(table_name)
        values = spec_codebook(getattr(layer_config, kind), 128)
        if table_name == UNIFORM_COEFFICIENTS:
            from qai_hub_models.models.templates.llm.turboquant.uniform_k8 import (
                coefficients,
            )

            values = coefficients()
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
        layer_config = config.for_layer(layer["layer"])
        layer["decoder"] = "native_unpack_lut_v1"
        if layer_config.key.quantizer == "uniform_k8_gaussian_v1":
            layer["key_decoder_op"] = UNIFORM_OP
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
                        native_op(layer_config, kind),
                        [
                            prefix + "packed",
                            scale16,
                            native_table(layer_config, kind),
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
    For strict arithmetic rounding tests use ONNX ReferenceEvaluator: ORT CPU
    FP16 promotion can remove intermediate rounding, even at ORT_DISABLE_ALL.
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
    for bits in (2, 3, 5, 6, 7, 8):
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
    for arithmetic_op in (STRUCTURED_OP, UNIFORM_OP):
        if not any(
            n.domain == NATIVE_DOMAIN and n.op_type == arithmetic_op
            for n in result.graph.node
        ):
            continue
        nodes = []
        for name, value in (
            ("sign_edge", 128),
            ("negative_edge", 127),
            ("region_stride", 128 if arithmetic_op == UNIFORM_OP else 64),
            ("two", 2),
            ("one", 1),
        ):
            const(name, np.array(value, np.int32))
        const("flat_shape", np.array([-1], np.int64))
        nodes.extend(
            [
                helper.make_node("Cast", ["packed"], ["codes"], to=TensorProto.INT32),
                helper.make_node(
                    "GreaterOrEqual", ["codes", "sign_edge"], ["positive"]
                ),
                helper.make_node("Sub", ["codes", "sign_edge"], ["pos_magnitude"]),
                helper.make_node("Sub", ["negative_edge", "codes"], ["neg_magnitude"]),
                helper.make_node(
                    "Where",
                    ["positive", "pos_magnitude", "neg_magnitude"],
                    ["magnitude"],
                ),
                helper.make_node("Div", ["magnitude", "region_stride"], ["region"]),
                helper.make_node("Mod", ["magnitude", "region_stride"], ["local"]),
                helper.make_node("Mul", ["region", "two"], ["offset_index"]),
                helper.make_node("Add", ["offset_index", "one"], ["step_index"]),
                helper.make_node("Reshape", ["coefficients", "flat_shape"], ["params"]),
                helper.make_node("Gather", ["params", "offset_index"], ["offset16"]),
                helper.make_node("Gather", ["params", "step_index"], ["step16"]),
                helper.make_node(
                    "Cast", ["offset16"], ["offset"], to=TensorProto.FLOAT
                ),
                helper.make_node("Cast", ["step16"], ["step"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["local"], ["local_f"], to=TensorProto.FLOAT),
                helper.make_node("Mul", ["local_f", "step"], ["product_f"]),
                helper.make_node(
                    "Cast", ["product_f"], ["product16"], to=TensorProto.FLOAT16
                ),
                helper.make_node(
                    "Cast", ["product16"], ["product"], to=TensorProto.FLOAT
                ),
                helper.make_node("Add", ["product", "offset"], ["centroid_f"]),
                helper.make_node(
                    "Cast", ["centroid_f"], ["centroid"], to=TensorProto.FLOAT16
                ),
                helper.make_node("Neg", ["centroid"], ["negative"]),
                helper.make_node(
                    "Where", ["positive", "centroid", "negative"], ["signed"]
                ),
                helper.make_node("Mul", ["signed", "scale"], ["decoded"]),
            ]
        )
        result.functions.append(
            helper.make_function(
                NATIVE_DOMAIN,
                arithmetic_op,
                ["packed", "scale", "coefficients"],
                ["decoded"],
                nodes,
                [helper.make_opsetid("", 17)],
            )
        )
    return result


def use_structured_arithmetic_decoder(
    result: SurgeryResult, config: TurboQuantConfig
) -> SurgeryResult:
    """Opt-in export-only kernel selection; quantizer/cache identity is unchanged.

    Call after Native and (when enabled) current-KV surgery. Every other node,
    encoding and I/O stays byte-for-byte unchanged; no encoder recomputation.
    """
    from qai_hub_models.models.templates.llm.turboquant.structured_k8 import (
        codebook,
        coefficients,
    )

    if (
        not config.structured_k8
        or not result.attention_tiles
        or any(
            x.get("decoder") != "native_unpack_lut_v1" for x in result.attention_tiles
        )
    ):
        raise ValueError(
            "Structured arithmetic requires an opt-in Structured K8 Native graph"
        )
    result = copy.deepcopy(result)
    graph = result.model.graph
    table_name = "tq_native_structured_k8_centroids_fp16"
    if not any(
        config.codec_for("key", x["layer"]).quantizer != "lloyd_max"
        for x in result.attention_tiles
    ):
        return result
    tables = [t for t in graph.initializer if t.name == table_name]
    if len(tables) != 1 or not np.array_equal(
        numpy_helper.to_array(tables[0]).ravel(), codebook().astype(np.float16)
    ):
        raise ValueError("Missing/mismatched Structured K8 LUT")
    changed = set()
    for node in graph.node:
        match = re.fullmatch(r"tq_key_(\d+)_(?:tile\d+|current)_native_fp16", node.name)
        if not match or config.codec_for("key", int(match[1])).quantizer == "lloyd_max":
            continue
        if (
            node.domain != NATIVE_DOMAIN
            or node.op_type != "Decode8"
            or node.input[2] != table_name
        ):
            raise ValueError("Expected original Native Structured K8 LUT decoder")
        node.op_type = STRUCTURED_OP
        node.input[2] = STRUCTURED_COEFFICIENTS
        changed.add(node.name)
    expected = {
        f"tq_key_{x['layer']}_tile{tile['start']}_native_fp16"
        for x in result.attention_tiles
        for tile in x["tiles"]
        if config.codec_for("key", x["layer"]).quantizer != "lloyd_max"
    } | {
        f"tq_key_{x['layer']}_current_native_fp16"
        for x in result.current_kv_attention
        if x["kind"] == "key"
        and config.codec_for("key", x["layer"]).quantizer != "lloyd_max"
    }
    if not changed or changed != expected:
        raise ValueError("Incomplete current/past Structured K8 decoder replacement")
    graph.initializer.append(
        numpy_helper.from_array(
            coefficients().astype(np.float16).reshape(1, 1, 1, 4),
            STRUCTURED_COEFFICIENTS,
        )
    )
    # Remove just the obsolete LUT. Do not prune/reorder unrelated graph data.
    if any(table_name in n.input for n in graph.node):
        raise ValueError("Unconverted Structured K8 LUT consumer")
    kept = [t for t in graph.initializer if t.name != table_name]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    for info in result.attention_tiles:
        if config.codec_for("key", info["layer"]).quantizer != "lloyd_max":
            info["key_decoder_op"] = STRUCTURED_OP
    for info in result.current_kv_attention:
        if (
            info["kind"] == "key"
            and config.codec_for("key", info["layer"]).quantizer != "lloyd_max"
        ):
            info["decoder_op"] = STRUCTURED_OP
    return result
