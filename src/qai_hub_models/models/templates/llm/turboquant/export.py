# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""ONNX lowering of 4-bit PolarQuant encode/decode (format version 1).

The subgraphs use only ops the QNN HTP backend implements for FP16/INT32 tensors
(QAIRT 2.48 op-def supplement): HTP has no bitwise ops and no UINT_8 arithmetic,
so indices are counted with ``Greater`` + INT32 ``ReduceSum``, packed as
``hi * 16 + lo`` in INT32 and only then cast to UINT8. Decode unpacks nibbles
with exact float arithmetic and selects symmetric centroid pairs with small
elementwise subgraphs. This avoids scalar ``Gather`` and the 15-fold expansion
of the original threshold ladder. Tensors stay rank <= 4, with head_dim as
the innermost axis of comparisons so HTP can vectorise them.

Layout: float tensors are ``[lead0, lead1, tokens, head_dim]``; packed tensors
``[lead0, lead1, tokens, head_dim // 2]`` uint8; norms ``[lead0, lead1, tokens, 1]``.
Standalone graphs default to ``lead = (1, heads)``; ``head_major=True`` models
the delta-KV layout ``(kv_heads, 1, tokens, head_dim)``. Internally both use
one batch, with reshapes at the boundary preserving the external layout.

Constants are named by content (``tq_rotation_s42_d128``, ...), so subgraphs for
many layers can be merged into one graph without duplicating initializers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from qai_hub_models.models.templates.llm.turboquant.config import (
    KVCodecSpec,
    TurboQuantConfig,
)
from qai_hub_models.models.templates.llm.turboquant.reference import (
    load_boundaries,
    load_codebook,
    make_rotation,
)

OPSET = 17
IR_VERSION = 8


@dataclass
class Subgraph:
    nodes: list[onnx.NodeProto] = field(default_factory=list)
    initializers: dict[str, TensorProto] = field(default_factory=dict)

    def const(self, name: str, value: np.ndarray) -> str:
        if name not in self.initializers:
            self.initializers[name] = numpy_helper.from_array(np.asarray(value), name)
        return name

    def shape(self, dims: list[int]) -> str:
        return self.const(
            "tq_shape_" + "_".join(map(str, dims)), np.array(dims, dtype=np.int64)
        )

    def node(
        self, op: str, inputs: list[str], outputs: list[str], **attrs: Any
    ) -> None:
        self.nodes.append(
            helper.make_node(op, inputs, outputs, name=outputs[0], **attrs)
        )

    def extend(self, other: Subgraph) -> None:
        self.nodes += other.nodes
        for name, tensor in other.initializers.items():
            self.initializers.setdefault(name, tensor)


def _check_supported(spec: KVCodecSpec, config: TurboQuantConfig) -> None:
    if not spec.is_polar or spec.bits != 4:
        raise NotImplementedError(
            "ONNX lowering exists only for 4-bit PolarQuant in format version 1."
        )
    if config.block_size % 2:
        raise ValueError("4-bit packing needs an even block size.")


def _scalars(sg: Subgraph) -> tuple[str, str, str]:
    return (
        sg.const("tq_zero_f", np.array(0.0, dtype=np.float32)),
        sg.const("tq_one_f", np.array(1.0, dtype=np.float32)),
        sg.const("tq_axis3", np.array([3], dtype=np.int64)),
    )


def _rotation_name(
    sg: Subgraph, config: TurboQuantConfig, spec: KVCodecSpec, transpose: bool
) -> str:
    d = config.block_size
    matrix = make_rotation(config.rotation, spec.seed, d).matrix()
    suffix = "_t" if transpose else ""
    value = matrix.T if transpose else matrix
    return sg.const(
        f"tq_rotation{suffix}_{config.rotation.value}_s{spec.seed}_d{d}",
        value.astype(np.float32),
    )


def byte_centroid_lut(bits: int, block_size: int) -> np.ndarray:
    """``[256, 2]`` table: centroids of the high and low nibble of each byte value."""
    centroids = load_codebook(bits, block_size)
    values = np.arange(256)
    return np.stack((centroids[values >> 4], centroids[values & 0xF]), axis=1)


def encode_subgraph(
    config: TurboQuantConfig,
    spec: KVCodecSpec,
    src: str,
    packed_out: str,
    norm_out: str,
    lead: tuple[int, int],
    num_tokens: int,
    prefix: str,
    opset: int = OPSET,
) -> Subgraph:
    """``src -> (packed_out, norm_out)``, matching :class:`PolarQuantReference`.

    ``opset`` is the default-domain opset of the graph the subgraph goes into:
    ReduceMax takes its axes as an attribute up to opset 17 and as an input from 18.
    """
    _check_supported(spec, config)
    d = config.block_size
    sg = Subgraph()
    zero, one, axis3 = _scalars(sg)
    p = prefix
    storage_lead = lead
    heads = lead[0] * lead[1]
    if lead[0] != 1:
        # HTP treats the outermost dimension as batches. The Hub KV ABI puts
        # heads there, making the same codec several times slower. These
        # reshapes preserve element order and the external cache ABI.
        lead = (1, heads)
        sg.node(
            "Reshape",
            [src, sg.shape([1, heads, num_tokens, d])],
            [f"{p}canonical_src"],
        )
        src = f"{p}canonical_src"
    threshold = sg.const("tq_prescale_threshold", np.array(256.0, dtype=np.float32))
    down = sg.const("tq_prescale_down", np.array(2.0**-8, dtype=np.float32))
    up = sg.const("tq_prescale_up", np.array(2.0**8, dtype=np.float32))

    # Overflow-safe x / ||x|| for FP16: divide by max|x| before squaring.
    sg.node("Abs", [src], [f"{p}abs"])
    if opset >= 18:
        sg.node("ReduceMax", [f"{p}abs", axis3], [f"{p}max_abs"], keepdims=1)
    else:
        sg.node("ReduceMax", [f"{p}abs"], [f"{p}max_abs"], axes=[3], keepdims=1)
    # SM8850 HTP FP16 Div is inaccurate for divisors above 2**14 (NaN past ~3.6e4),
    # so rows with max|x| > 256 are first scaled by an exact power of two.
    sg.node("Greater", [f"{p}max_abs", threshold], [f"{p}big"])
    sg.node("Where", [f"{p}big", down, one], [f"{p}pre"])
    sg.node("Where", [f"{p}big", up, one], [f"{p}post"])
    sg.node("Mul", [src, f"{p}pre"], [f"{p}x_pre"])
    sg.node("Mul", [f"{p}max_abs", f"{p}pre"], [f"{p}max_pre"])
    sg.node("Greater", [f"{p}max_pre", zero], [f"{p}has_mag"])
    sg.node("Where", [f"{p}has_mag", f"{p}max_pre", one], [f"{p}scale"])
    sg.node("Div", [f"{p}x_pre", f"{p}scale"], [f"{p}scaled"])
    sg.node("Mul", [f"{p}scaled", f"{p}scaled"], [f"{p}sq"])
    sg.node("ReduceSum", [f"{p}sq", axis3], [f"{p}sum_sq"], keepdims=1)
    sg.node("Sqrt", [f"{p}sum_sq"], [f"{p}len"])
    sg.node("Greater", [f"{p}len", zero], [f"{p}has_len"])
    sg.node("Where", [f"{p}has_len", f"{p}len", one], [f"{p}safe_len"])
    sg.node("Div", [f"{p}scaled", f"{p}safe_len"], [f"{p}unit"])
    sg.node("Mul", [f"{p}max_pre", f"{p}len"], [f"{p}norm_pre"])
    norm_target = norm_out if storage_lead == lead else f"{p}canonical_norm"
    sg.node("Mul", [f"{p}norm_pre", f"{p}post"], [norm_target])
    if storage_lead != lead:
        sg.node(
            "Reshape",
            [norm_target, sg.shape([*storage_lead, num_tokens, 1])],
            [norm_out],
        )

    # Keep head_dim innermost: HTP vectorises that axis. Putting the 15
    # thresholds there instead makes both the broadcast and reduction slow.
    boundaries = sg.const(
        f"tq_boundaries_vector_b{spec.bits}_d{d}",
        load_boundaries(spec.bits, d).astype(np.float32).reshape(1, 1, -1, 1),
    )
    int64 = np.int64
    start0 = sg.const("tq_start0", np.array([0], dtype=int64))
    start1 = sg.const("tq_start1", np.array([1], dtype=int64))
    end_d = sg.const(f"tq_end{d}", np.array([d], dtype=int64))
    axis2 = sg.const("tq_axis2", np.array([2], dtype=int64))
    step2 = sg.const("tq_step2", np.array([2], dtype=int64))
    sixteen = sg.const("tq_sixteen_i32", np.array(16, dtype=np.int32))
    # Row vectors: y = R x  <=>  y_row = x_row @ R^T.
    sg.node(
        "MatMul", [f"{p}unit", _rotation_name(sg, config, spec, True)], [f"{p}rotated"]
    )
    sg.node(
        "Reshape",
        [f"{p}rotated", sg.shape([heads, num_tokens, 1, d])],
        [f"{p}rotated_ht1d"],
    )
    sg.node("Greater", [f"{p}rotated_ht1d", boundaries], [f"{p}above"])
    sg.node("Cast", [f"{p}above"], [f"{p}above_i32"], to=TensorProto.INT32)
    sg.node("ReduceSum", [f"{p}above_i32", axis2], [f"{p}index"], keepdims=0)
    sg.node("Slice", [f"{p}index", start0, end_d, axis2, step2], [f"{p}index_hi"])
    sg.node("Slice", [f"{p}index", start1, end_d, axis2, step2], [f"{p}index_lo"])
    sg.node("Mul", [f"{p}index_hi", sixteen], [f"{p}index_hi_shifted"])
    sg.node("Add", [f"{p}index_hi_shifted", f"{p}index_lo"], [f"{p}byte_i32"])
    sg.node("Cast", [f"{p}byte_i32"], [f"{p}byte_u8"], to=TensorProto.UINT8)
    sg.node(
        "Reshape",
        [f"{p}byte_u8", sg.shape([*storage_lead, num_tokens, d // 2])],
        [packed_out],
    )
    return sg


def _select_centroid(
    sg: Subgraph,
    idx: str,
    centroids: np.ndarray,
    bits: int,
    block_size: int,
    prefix: str,
) -> str:
    """Exact 16-entry lookup on integer indices, using symmetric affine pairs.

    The frozen codebook is symmetric. Fold indices to magnitudes 0..7, then
    select one of four lines, each interpolating two adjacent centroids.
    This needs no 15-way broadcast, Gather, or narrow matrix multiplication;
    all elementwise operations keep head_dim innermost. Where leaves depend
    on the input (constant-only leaves fail HTP detailed profiling).
    """
    if bits != 4 or not np.array_equal(centroids, -centroids[::-1]):
        raise ValueError("Affine-pair lookup requires a symmetric 4-bit codebook.")
    p = prefix

    def scalar(name: str, value: float) -> str:
        return sg.const(name, np.array(value, dtype=np.float32))

    middle = scalar("tq_index_middle", 7.5)
    half = scalar("tq_half_f", 0.5)
    zero = scalar("tq_zero_f", 0.0)
    sg.node("Sub", [idx, middle], [f"{p}signed_index"])
    sg.node("Abs", [f"{p}signed_index"], [f"{p}abs_index"])
    sg.node("Sub", [f"{p}abs_index", half], [f"{p}magnitude_index"])
    for pair in range(4):
        offset = 2 * pair
        slope = centroids[8 + offset + 1] - centroids[8 + offset]
        base = centroids[8 + offset] - offset * slope
        tag = f"tq_centroid_b{bits}_d{block_size}_pair{pair}"
        sg.node(
            "Mul",
            [f"{p}magnitude_index", scalar(tag + "_slope", slope)],
            [f"{p}pair{pair}_rise"],
        )
        sg.node(
            "Add",
            [f"{p}pair{pair}_rise", scalar(tag + "_base", base)],
            [f"{p}pair{pair}"],
        )
    for label, threshold, hi, lo in (
        ("lower", 1.5, "pair1", "pair0"),
        ("upper", 5.5, "pair3", "pair2"),
        ("magnitude", 3.5, "upper", "lower"),
    ):
        sg.node(
            "Greater",
            [f"{p}magnitude_index", scalar(f"tq_pair_threshold_{label}", threshold)],
            [f"{p}{label}_test"],
        )
        sg.node("Where", [f"{p}{label}_test", f"{p}{hi}", f"{p}{lo}"], [f"{p}{label}"])
    sg.node("Sub", [zero, f"{p}magnitude"], [f"{p}negative"])
    sg.node("Greater", [idx, middle], [f"{p}positive"])
    sg.node("Where", [f"{p}positive", f"{p}magnitude", f"{p}negative"], [f"{p}y_hat"])
    return f"{p}y_hat"


def decode_subgraph(
    config: TurboQuantConfig,
    spec: KVCodecSpec,
    packed_in: str,
    norm_in: str,
    dst: str,
    lead: tuple[int, int],
    num_tokens: int,
    prefix: str,
) -> Subgraph:
    """``(packed_in, norm_in) -> dst`` with the config's norm correction."""
    _check_supported(spec, config)
    d = config.block_size
    sg = Subgraph()
    zero, one, axis3 = _scalars(sg)
    p = prefix
    heads = lead[0] * lead[1]
    storage_lead = lead
    # Nibbles by exact float16 arithmetic (bytes are < 256, so b/16 and the
    # remainder are exact); the two nibbles of a byte are MSB first. The
    # arithmetic runs on the packed layout (innermost dim 64): HTP vectorises
    # over the innermost dim, so an innermost dim of 1 would be ~3x slower.
    inv16 = sg.const("tq_inv16_f", np.array(1.0 / 16, dtype=np.float32))
    sixteen_f = sg.const("tq_sixteen_f", np.array(16.0, dtype=np.float32))
    byte_target = f"{p}byte_f" if lead[0] == 1 else f"{p}storage_byte_f"
    sg.node("Cast", [packed_in], [byte_target], to=TensorProto.FLOAT)
    if lead[0] != 1:
        lead = (1, heads)
        # Cast before reshaping: HTP cannot transpose raw UINT8 tensors.
        sg.node(
            "Reshape",
            [byte_target, sg.shape([1, heads, num_tokens, d // 2])],
            [f"{p}byte_f"],
        )
        sg.node(
            "Reshape",
            [norm_in, sg.shape([1, heads, num_tokens, 1])],
            [f"{p}canonical_norm"],
        )
        norm_in = f"{p}canonical_norm"
    sg.node("Mul", [f"{p}byte_f", inv16], [f"{p}byte_div16"])
    sg.node("Floor", [f"{p}byte_div16"], [f"{p}nibble_hi"])
    sg.node("Mul", [f"{p}nibble_hi", sixteen_f], [f"{p}nibble_hi16"])
    sg.node("Sub", [f"{p}byte_f", f"{p}nibble_hi16"], [f"{p}nibble_lo"])
    pair_shape = sg.shape([heads, num_tokens, d // 2, 1])
    sg.node("Reshape", [f"{p}nibble_hi", pair_shape], [f"{p}nibble_hi_1"])
    sg.node("Reshape", [f"{p}nibble_lo", pair_shape], [f"{p}nibble_lo_1"])
    sg.node(
        "Concat", [f"{p}nibble_hi_1", f"{p}nibble_lo_1"], [f"{p}nibble_pairs"], axis=3
    )
    sg.node(
        "Reshape",
        [f"{p}nibble_pairs", sg.shape([lead[0], lead[1], num_tokens, d])],
        [f"{p}index_f"],
    )
    centroids = load_codebook(spec.bits, d).astype(np.float32)
    y_hat = _select_centroid(sg, f"{p}index_f", centroids, spec.bits, d, p)
    y_unit = y_hat
    if config.norm_correction:
        sg.node("Mul", [y_hat, y_hat], [f"{p}sq"])
        sg.node("ReduceSum", [f"{p}sq", axis3], [f"{p}sum_sq"], keepdims=1)
        sg.node("Sqrt", [f"{p}sum_sq"], [f"{p}len"])
        sg.node("Greater", [f"{p}len", zero], [f"{p}has_len"])
        sg.node("Where", [f"{p}has_len", f"{p}len", one], [f"{p}safe_len"])
        sg.node("Div", [y_hat, f"{p}safe_len"], [f"{p}y_unit"])
        y_unit = f"{p}y_unit"
    # Row vectors: x = R^T y  <=>  x_row = y_row @ R.
    sg.node("MatMul", [y_unit, _rotation_name(sg, config, spec, False)], [f"{p}x_unit"])
    target = dst if storage_lead == lead else f"{p}canonical_restored"
    sg.node("Mul", [f"{p}x_unit", norm_in], [target])
    if storage_lead != lead:
        sg.node("Reshape", [target, sg.shape([*storage_lead, num_tokens, d])], [dst])
    return sg


def build_encode_model(
    config: TurboQuantConfig,
    spec: KVCodecSpec,
    num_heads: int,
    num_tokens: int,
    graph_name: str = "tq_encode",
    *,
    head_major: bool = False,
) -> onnx.ModelProto:
    """Standalone ``x -> (packed, norm)`` graph."""
    lead = (num_heads, 1) if head_major else (1, num_heads)
    sg = encode_subgraph(config, spec, "x", "packed", "norm", lead, num_tokens, "enc_")
    d = config.block_size
    return _finish(
        sg,
        graph_name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [*lead, num_tokens, d])],
        [
            helper.make_tensor_value_info(
                "packed", TensorProto.UINT8, [*lead, num_tokens, d // 2]
            ),
            helper.make_tensor_value_info(
                "norm", TensorProto.FLOAT, [*lead, num_tokens, 1]
            ),
        ],
    )


def build_decode_model(
    config: TurboQuantConfig,
    spec: KVCodecSpec,
    num_heads: int,
    num_tokens: int,
    graph_name: str = "tq_decode",
    *,
    head_major: bool = False,
) -> onnx.ModelProto:
    """Standalone ``(packed, norm) -> x_hat`` graph."""
    lead = (num_heads, 1) if head_major else (1, num_heads)
    sg = decode_subgraph(
        config, spec, "packed", "norm", "x_hat", lead, num_tokens, "dec_"
    )
    d = config.block_size
    return _finish(
        sg,
        graph_name,
        [
            helper.make_tensor_value_info(
                "packed", TensorProto.UINT8, [*lead, num_tokens, d // 2]
            ),
            helper.make_tensor_value_info(
                "norm", TensorProto.FLOAT, [*lead, num_tokens, 1]
            ),
        ],
        [
            helper.make_tensor_value_info(
                "x_hat", TensorProto.FLOAT, [*lead, num_tokens, d]
            )
        ],
    )


def _finish(
    sg: Subgraph,
    graph_name: str,
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
) -> onnx.ModelProto:
    graph = helper.make_graph(
        sg.nodes, graph_name, inputs, outputs, list(sg.initializers.values())
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="qai_hub_models.turboquant",
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    return model
