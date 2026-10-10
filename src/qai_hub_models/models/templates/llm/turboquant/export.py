# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""ONNX lowering of PolarQuant encode/decode (4-bit legacy; scaled 2..6-bit).

The subgraphs use only ops the QNN HTP backend implements for FP16/INT32 tensors
(QAIRT 2.48 op-def supplement): HTP has no bitwise ops and no UINT_8 arithmetic,
so indices use an exact Lloyd-Max comparison tree (one level per bit), without
an expanded threshold axis. Four-bit uses ``hi * 16 + lo`` in INT32; asymmetric
profiles pack continuous MSB-first streams with power-of-two fragment arithmetic.
Both sum bytes in INT32 before casting to UINT8. Decode unpacks codes
with exact float arithmetic and selects symmetric centroid pairs with small
elementwise subgraphs. This avoids scalar ``Gather`` and the 15-fold expansion
of the original threshold ladder. Tensors stay rank <= 4, with head_dim as
the innermost axis of comparisons so HTP can vectorise them.

Layout: float tensors are ``[lead0, lead1, tokens, head_dim]``; packed tensors
``[lead0, lead1, tokens, head_dim * storage_bits // 8]`` uint8;
norms ``[lead0, lead1, tokens, 1]``.
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
    if not spec.is_polar or (spec.bits != 4 and not config.precomputed_norm):
        raise NotImplementedError(
            "ONNX lowering requires 4-bit or scaled 2..6-bit PolarQuant."
        )
    if config.block_size % 8:
        raise ValueError("Packed codecs require a block size divisible by eight.")


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


def _scalar_index_tree(
    sg: Subgraph, src: str, bits: int, block_size: int, prefix: str
) -> str:
    """Exact ``searchsorted(boundaries, src, side='left')``, same shape as src.

    Select one threshold at each level using earlier decisions. No Gather,
    centroid-distance search, or all-boundary comparison/reduction is needed.
    Leaf selection uses bit*hi + (1-bit)*lo: static/static Where fails HTP
    detailed execution, while lo + bit*(hi-lo) can shift FP16 boundaries.
    """
    if bits not in (2, 3, 4, 5, 6):
        raise ValueError("Scalar tree requires a frozen 2..6-bit codebook.")
    boundaries = load_boundaries(bits, block_size).astype(np.float32)

    def constant(name: str, value: float) -> str:
        return sg.const(
            f"tq_scalar_b{bits}_d{block_size}_{name}",
            np.array(value, dtype=np.float32),
        )

    def node(op: str, inputs: list[str], name: str, **attrs: Any) -> str:
        out = prefix + name
        sg.node(op, inputs, [out], **attrs)
        return out

    decisions: list[str] = []
    decision_floats: list[tuple[str, str]] = []
    terms: list[str] = []
    for level in range(bits):
        stride = 1 << (bits - 1 - level)
        choices = [
            constant(f"threshold_{i}", float(boundaries[i]))
            for i in range(stride - 1, len(boundaries), 2 * stride)
        ]
        for previous in range(level - 1, -1, -1):
            selected = []
            for i in range(0, len(choices), 2):
                label = f"level{level}_select{previous}_{i // 2}"
                if previous == level - 1:
                    bit_f, inv_f = decision_floats[previous]
                    hi = node("Mul", [bit_f, choices[i + 1]], label + "_hi")
                    lo = node("Mul", [inv_f, choices[i]], label + "_lo")
                    selected.append(node("Add", [hi, lo], label))
                else:
                    selected.append(
                        node(
                            "Where",
                            [decisions[previous], choices[i + 1], choices[i]],
                            label,
                        )
                    )
            choices = selected
        decision = node("Greater", [src, choices[0]], f"level{level}_above")
        decisions.append(decision)
        if level < bits - 1:
            bit_f = node(
                "Cast", [decision], f"level{level}_float", to=TensorProto.FLOAT
            )
            inv_f = node("Sub", [constant("one", 1), bit_f], f"level{level}_inverse")
            decision_floats.append((bit_f, inv_f))
        bit = node("Cast", [decision], f"level{level}_bit", to=TensorProto.INT32)
        weight = sg.const(
            f"tq_scalar_weight_{stride}", np.array(stride, dtype=np.int32)
        )
        terms.append(node("Mul", [bit, weight], f"level{level}_weighted"))
    index = terms[0]
    for level, term in enumerate(terms[1:], 1):
        index = node("Add", [index, term], f"sum{level}")
    return index


def storage_bits(config: TurboQuantConfig, spec: KVCodecSpec) -> int:
    """QJL's K3 base still occupies a nibble together with its sign bit."""
    return 4 if config.qjl and spec == config.key else spec.bits


def _repack(
    sg: Subgraph,
    src: str,
    in_bits: int,
    out_bits: int,
    heads: int,
    tokens: int,
    d: int,
    lead: tuple[int, int],
    prefix: str,
    *,
    integer_output: bool = False,
) -> str:
    """MSB-first stream regrouping using exact small power-of-two arithmetic.

    Groups cover eight scalar codes (bits bytes). No intermediate exceeds 255,
    so FP16 arithmetic is exact. All tensors have rank <= 4. The 4-bit legacy
    path deliberately retains its existing faster nibble implementation.
    """
    p = prefix
    group_bits = in_bits * out_bits
    in_count, out_count = group_bits // in_bits, group_bits // out_bits
    groups = d // 8
    sg.node("Cast", [src], [p + "float"], to=TensorProto.FLOAT)
    sg.node(
        "Reshape",
        [p + "float", sg.shape([heads, tokens, groups * in_count])],
        [p + "flat"],
    )

    def number(n: float) -> str:
        return sg.const(f"tq_repack_f_{n:g}", np.array(n, np.float32))

    inputs = []
    for i in range(in_count):
        name = p + f"input{i}"
        sg.node(
            "Slice",
            [
                p + "flat",
                sg.shape([i]),
                sg.shape([groups * in_count]),
                sg.shape([2]),
                sg.shape([in_count]),
            ],
            [name],
        )
        inputs.append(name)
    columns = []
    for o in range(out_count):
        terms = []
        for i in range(in_count):
            lo, hi = (
                max(i * in_bits, o * out_bits),
                min((i + 1) * in_bits, (o + 1) * out_bits),
            )
            if lo >= hi:
                continue
            tag = p + f"o{o}_i{i}_"
            term = inputs[i]
            right = (i + 1) * in_bits - hi
            if right:
                sg.node("Mul", [term, number(2.0**-right)], [tag + "divide"])
                sg.node("Floor", [tag + "divide"], [tag + "shift"])
                term = tag + "shift"
            if lo > i * in_bits:
                sg.node("Mul", [term, number(2.0 ** -(hi - lo))], [tag + "upper_frac"])
                sg.node("Floor", [tag + "upper_frac"], [tag + "upper"])
                sg.node(
                    "Mul",
                    [tag + "upper", number(2 ** (hi - lo))],
                    [tag + "upper_shift"],
                )
                sg.node("Sub", [term, tag + "upper_shift"], [tag + "masked"])
                term = tag + "masked"
            left = (o + 1) * out_bits - hi
            if left:
                sg.node("Mul", [term, number(2**left)], [tag + "left"])
                term = tag + "left"
            if integer_output:
                sg.node("Cast", [term], [tag + "i32"], to=TensorProto.INT32)
                term = tag + "i32"
            terms.append(term)
        total = terms[0]
        for j, term in enumerate(terms[1:], 1):
            output = p + f"o{o}_sum{j}"
            sg.node("Add", [total, term], [output])
            total = output
        col = p + f"column{o}"
        sg.node("Reshape", [total, sg.shape([heads, tokens, groups, 1])], [col])
        columns.append(col)
    sg.node("Concat", columns, [p + "columns"], axis=3)
    sg.node(
        "Reshape",
        [p + "columns", sg.shape([*lead, tokens, groups * out_count])],
        [p + "output"],
    )
    return p + "output"


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
    norm_target = (
        f"{p}raw_norm"
        if config.precomputed_norm
        else norm_out
        if storage_lead == lead
        else f"{p}canonical_norm"
    )
    sg.node("Mul", [f"{p}norm_pre", f"{p}post"], [norm_target])
    if storage_lead != lead and not config.precomputed_norm:
        sg.node(
            "Reshape",
            [norm_target, sg.shape([*storage_lead, num_tokens, 1])],
            [norm_out],
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
    # Preserve head_dim as the vectorized innermost axis, without a boundary axis.
    index = _scalar_index_tree(sg, f"{p}rotated", spec.bits, d, p + "scalar_")
    sg.node(
        "Reshape",
        [index, sg.shape([heads, num_tokens, d])],
        [f"{p}index"],
    )
    if storage_bits(config, spec) == 4:
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
    else:
        packed = _repack(
            sg,
            f"{p}index",
            spec.bits,
            8,
            heads,
            num_tokens,
            d,
            storage_lead,
            p + "pack_",
            integer_output=True,
        )
        # Integer fragment sums prevent converter folding of adjacent casts
        # into an unsupported FP16 -> raw UINT8 conversion.
        sg.node("Cast", [packed], [packed_out], to=TensorProto.UINT8)
    if config.precomputed_norm:
        effective = norm_target
        if config.norm_correction:
            sg.node("Cast", [f"{p}index"], [f"{p}scale_index_f"], to=TensorProto.FLOAT)
            sg.node(
                "Reshape",
                [f"{p}scale_index_f", sg.shape([*lead, num_tokens, d])],
                [f"{p}scale_index"],
            )
            y_hat = _select_centroid(
                sg,
                f"{p}scale_index",
                load_codebook(spec.bits, d).astype(np.float32),
                spec.bits,
                d,
                p + "scale_",
            )
            sg.node("Mul", [y_hat, y_hat], [f"{p}scale_sq"])
            sg.node("ReduceSum", [f"{p}scale_sq", axis3], [f"{p}scale_sum"], keepdims=1)
            sg.node("Sqrt", [f"{p}scale_sum"], [f"{p}scale_len"])
            sg.node("Greater", [f"{p}scale_len", zero], [f"{p}scale_valid"])
            sg.node(
                "Where", [f"{p}scale_valid", f"{p}scale_len", one], [f"{p}scale_safe"]
            )
            effective = f"{p}effective_scale"
            sg.node("Div", [norm_target, f"{p}scale_safe"], [effective])
        sg.node(
            "Reshape", [effective, sg.shape([*storage_lead, num_tokens, 1])], [norm_out]
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
    """Symmetric affine-pair lookup for frozen 2..6-bit centroid tables.

    The frozen codebook is symmetric. Fold indices to magnitudes 0..7, then
    select one of four lines, each interpolating two adjacent centroids.
    This needs no 15-way broadcast, Gather, or narrow matrix multiplication;
    all elementwise operations keep head_dim innermost. Where leaves depend
    on the input (constant-only leaves fail HTP detailed profiling).
    """
    if bits not in (2, 3, 4, 5, 6) or not np.array_equal(centroids, -centroids[::-1]):
        raise ValueError("Affine-pair lookup requires a symmetric 2..6-bit codebook.")
    p = prefix

    def scalar(name: str, value: float) -> str:
        return sg.const(name, np.array(value, dtype=np.float32))

    count = 1 << bits
    middle = scalar(
        "tq_index_middle" if bits == 4 else f"tq_index_middle_b{bits}", count / 2 - 0.5
    )
    half = scalar("tq_half_f", 0.5)
    zero = scalar("tq_zero_f", 0.0)
    sg.node("Sub", [idx, middle], [f"{p}signed_index"])
    sg.node("Abs", [f"{p}signed_index"], [f"{p}abs_index"])
    sg.node("Sub", [f"{p}abs_index", half], [f"{p}magnitude_index"])
    for pair in range(count // 4):
        offset = 2 * pair
        slope = centroids[count // 2 + offset + 1] - centroids[count // 2 + offset]
        base = centroids[count // 2 + offset] - offset * slope
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
    branches = (
        (
            ("lower", 1.5, "pair1", "pair0"),
            ("upper", 5.5, "pair3", "pair2"),
            ("magnitude", 3.5, "upper", "lower"),
        )
        if bits == 4
        else (("magnitude", 1.5, "pair1", "pair0"),)
    )
    if bits not in (3, 4):
        branches = []

        def branch(start: int, stop: int, label: str) -> str:
            if stop - start == 1:
                return f"pair{start}"
            mid = (start + stop) // 2
            lo = branch(start, mid, label + "_lo")
            hi = branch(mid, stop, label + "_hi")
            branches.append((label, 2 * mid - 0.5, hi, lo))
            return label

        root = branch(0, count // 4, "magnitude")
        if bits == 2:
            sg.node("Identity", [p + root], [p + "magnitude"])
    for label, threshold, hi, lo in branches:
        sg.node(
            "Greater",
            [
                f"{p}magnitude_index",
                scalar(
                    f"tq_pair_threshold_{label}"
                    if bits == 4
                    else f"tq_pair_threshold_b{bits}_{label}",
                    threshold,
                ),
            ],
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
    *,
    rotated: bool = False,
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
            [
                byte_target,
                sg.shape([1, heads, num_tokens, d * storage_bits(config, spec) // 8]),
            ],
            [f"{p}byte_f"],
        )
        sg.node(
            "Reshape",
            [norm_in, sg.shape([1, heads, num_tokens, 1])],
            [f"{p}canonical_norm"],
        )
        norm_in = f"{p}canonical_norm"
    if storage_bits(config, spec) == 4:
        sg.node("Mul", [f"{p}byte_f", inv16], [f"{p}byte_div16"])
        sg.node("Floor", [f"{p}byte_div16"], [f"{p}nibble_hi"])
        sg.node("Mul", [f"{p}nibble_hi", sixteen_f], [f"{p}nibble_hi16"])
        sg.node("Sub", [f"{p}byte_f", f"{p}nibble_hi16"], [f"{p}nibble_lo"])
        pair_shape = sg.shape([heads, num_tokens, d // 2, 1])
        sg.node("Reshape", [f"{p}nibble_hi", pair_shape], [f"{p}nibble_hi_1"])
        sg.node("Reshape", [f"{p}nibble_lo", pair_shape], [f"{p}nibble_lo_1"])
        sg.node(
            "Concat",
            [f"{p}nibble_hi_1", f"{p}nibble_lo_1"],
            [f"{p}nibble_pairs"],
            axis=3,
        )
        sg.node(
            "Reshape",
            [f"{p}nibble_pairs", sg.shape([*lead, num_tokens, d])],
            [f"{p}index_f"],
        )
    else:
        unpacked = _repack(
            sg, f"{p}byte_f", 8, spec.bits, heads, num_tokens, d, lead, p + "unpack_"
        )
        sg.node("Identity", [unpacked], [f"{p}index_f"])
    centroids = load_codebook(spec.bits, d).astype(np.float32)
    index_f = f"{p}index_f"
    if config.qjl and spec == config.key:
        eight = sg.const("tq_eight_f", np.array(8, dtype=np.float32))
        sg.node("GreaterOrEqual", [index_f, eight], [f"{p}qjl_positive"])
        sg.node("Sub", [index_f, eight], [f"{p}index_minus8"])
        sg.node(
            "Where",
            [f"{p}qjl_positive", f"{p}index_minus8", index_f],
            [f"{p}mse_index"],
        )
        index_f = f"{p}mse_index"
    y_hat = _select_centroid(sg, index_f, centroids, spec.bits, d, p)
    y_unit = y_hat
    if config.norm_correction and not config.precomputed_norm:
        sg.node("Mul", [y_hat, y_hat], [f"{p}sq"])
        sg.node("ReduceSum", [f"{p}sq", axis3], [f"{p}sum_sq"], keepdims=1)
        sg.node("Sqrt", [f"{p}sum_sq"], [f"{p}len"])
        sg.node("Greater", [f"{p}len", zero], [f"{p}has_len"])
        sg.node("Where", [f"{p}has_len", f"{p}len", one], [f"{p}safe_len"])
        sg.node("Div", [y_hat, f"{p}safe_len"], [f"{p}y_unit"])
        y_unit = f"{p}y_unit"
    # Row vectors: x = R^T y  <=>  x_row = y_row @ R.
    if not rotated:
        sg.node(
            "MatMul", [y_unit, _rotation_name(sg, config, spec, False)], [f"{p}x_unit"]
        )
        y_unit = f"{p}x_unit"
    target = dst if storage_lead == lead else f"{p}canonical_restored"
    sg.node("Mul", [y_unit, norm_in], [target])
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
    if config.qjl and spec == config.key:
        raise ValueError("Use build_qjl_encode_model for the combined K3+1 encoder.")
    lead = (num_heads, 1) if head_major else (1, num_heads)
    sg = encode_subgraph(config, spec, "x", "packed", "norm", lead, num_tokens, "enc_")
    d = config.block_size
    return _finish(
        sg,
        graph_name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [*lead, num_tokens, d])],
        [
            helper.make_tensor_value_info(
                "packed",
                TensorProto.UINT8,
                [*lead, num_tokens, d * storage_bits(config, spec) // 8],
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
    if config.qjl and spec == config.key:
        raise ValueError(
            "QJL K needs its residual scale; use QJLKeyReference or tiled attention."
        )
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
                "packed",
                TensorProto.UINT8,
                [*lead, num_tokens, d * storage_bits(config, spec) // 8],
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
