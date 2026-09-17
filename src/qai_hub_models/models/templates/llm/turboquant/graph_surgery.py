# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Apply a TurboQuant profile to an exported delta-KV LLM part (ONNX + AIMET encodings).

Each KV tensor of the part is changed according to its :class:`KVCodecSpec`
(``P = context_length - seq_len`` past tokens, ``A = seq_len``):

Storage (``kind``)
    - ``POLAR``: ``past_{kind}_{L}_in`` is replaced by ``tq_{kind}_{L}_packed_in``
      ``uint8 (kv_heads, 1, P, bytes)`` and ``tq_{kind}_{L}_norm_in`` ``float
      (kv_heads, 1, P, 1)``; a decode subgraph feeds the original consumers.
      ``past_{kind}_{L}_out`` becomes internal and an encode subgraph produces
      ``tq_{kind}_{L}_{packed,norm}_out`` (token axis -2).
    - ``INT16``: names and shapes stay; the cache I/O, the taps and the whole
      write path carry one explicit 16-bit grid, so the stored value is the
      producer's 16-bit activation with no conversion.

Codec input (both kinds)
    The value before the KV-specific int8 encodings. The first computing op
    behind ``past_*_out`` (the tap: the R3 MatMul for K, v_proj for V in Qwen3)
    is re-gridded from 8 to 16 bits over its calibrated range, so weights and
    the op stay integer. The value-preserving chain from the tap to
    ``past_*_out`` is rebuilt for the cache (behind an encoding-less float guard
    for POLAR, on the shared 16-bit grid for INT16); tensors it shares with the
    attention path are duplicated so attention keeps the exported int8 path for
    new tokens. On the read path the per-head slices stay on the cache's
    representation and QAIRT converts to int8 at the attention Concat consumed
    by the 16x8 MatMul, whose encoding is kept.

Weights and every other activation encoding are reused unchanged. Codec
subgraphs have static shapes, so apply this per ``(seq_len, context_length)``.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import onnx
from onnx import TensorProto, helper

from qai_hub_models.models.templates.llm.turboquant.config import (
    KVCodecSpec,
    TurboQuantConfig,
)
from qai_hub_models.models.templates.llm.turboquant.export import (
    OPSET,
    Subgraph,
    decode_subgraph,
    encode_subgraph,
)
from qai_hub_models.models.templates.llm.turboquant.packing import packed_nbytes

KV_INPUT = re.compile(r"past_(key|value)_(\d+)_in")

# Ops that carry KV values unchanged; value is the number of data inputs
# (None = all inputs). Other inputs (shapes, slice bounds) are not walked.
_VALUE_PRESERVING: dict[str, int | None] = {
    "Concat": None,
    "Transpose": 1,
    "Reshape": 1,
    "Slice": 1,
    "Squeeze": 1,
    "Unsqueeze": 1,
    "Identity": 1,
    "Cast": 1,
}
REGRID_BITS = 16


@dataclass(frozen=True)
class CodecTensorIO:
    """Graph I/O for one codec-stored KV tensor in one exported graph."""

    kind: str
    layer: int
    packed_in: str
    norm_in: str
    packed_out: str
    norm_out: str
    num_kv_heads: int
    past_tokens: int
    new_tokens: int
    packed_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class EncodingEdit:
    """One activation-encoding change made for a KV tensor."""

    tensor: str
    action: str  # "drop" or "regrid"
    reason: str
    before: dict[str, Any]
    after: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class KVPath:
    """Where one KV tensor's values come from and where the kept int8 boundary is."""

    kind: str
    layer: int
    taps: tuple[str, ...]  # first computing-op outputs behind past_*_out
    tap_ops: tuple[str, ...]
    kept_consumers: tuple[str, ...]  # read-side tensors whose encodings stay

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SurgeryResult:
    model: onnx.ModelProto
    encodings: dict[str, Any]
    codec_io: list[CodecTensorIO] = field(default_factory=list)
    edits: list[EncodingEdit] = field(default_factory=list)
    paths: list[KVPath] = field(default_factory=list)
    attention_tiles: list[dict[str, Any]] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        return {
            "codec_io": [io.to_dict() for io in self.codec_io],
            "encoding_edits": [e.to_dict() for e in self.edits],
            "kv_paths": [p.to_dict() for p in self.paths],
            "attention_tiles": self.attention_tiles,
        }


class _GraphIndex:
    def __init__(self, graph: onnx.GraphProto) -> None:
        self.producer: dict[str, onnx.NodeProto] = {}
        self.consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
        self.initializers = {t.name for t in graph.initializer}
        for node in graph.node:
            for out in node.output:
                self.producer[out] = node
            for name in node.input:
                self.consumers[name].append(node)


def _dims(value: onnx.ValueInfoProto) -> list[Any]:
    return [
        d.dim_param if d.dim_param else d.dim_value
        for d in value.type.tensor_type.shape.dim
    ]


def _data_inputs(node: onnx.NodeProto) -> list[str]:
    count = _VALUE_PRESERVING[node.op_type]
    return list(node.input) if count is None else list(node.input[:count])


def regrid_encoding(enc: dict[str, Any], bits: int = REGRID_BITS) -> dict[str, Any]:
    """Same calibrated minimum, ``bits``-bit grid: scale / 2^(bits-bw), offset * 2^(bits-bw)."""
    factor = 2 ** (bits - int(enc["bw"]))
    new = copy.deepcopy(enc)
    new["bw"] = bits
    new["scale"] = [s / factor for s in enc["scale"]]
    new["offset"] = [o * factor for o in enc["offset"]]
    return new


def _grid_bounds(enc: dict[str, Any]) -> tuple[float, float]:
    if len(enc["scale"]) != 1:
        raise ValueError(
            f"{enc['name']}: only per-tensor activation encodings are supported."
        )
    scale, offset = float(enc["scale"][0]), float(enc["offset"][0])
    levels = (1 << int(enc["bw"])) - 1
    return offset * scale, (levels + offset) * scale


def union_grid(
    encodings: list[dict[str, Any]], bits: int = REGRID_BITS
) -> dict[str, Any]:
    """One ``bits``-bit affine grid that covers every input encoding's range.

    Symmetric per-head grids of one cache tensor share the offset, so the union
    is the widest head's grid and the other heads are stored on it exactly.
    """
    grids = [regrid_encoding(e, bits) for e in encodings]
    new = copy.deepcopy(grids[0])
    if len({g["offset"][0] for g in grids}) == 1:
        # Same zero point (symmetric grids): the widest scale covers every range exactly.
        new["scale"] = [max(g["scale"][0] for g in grids)]
        return new
    lo = min(_grid_bounds(e)[0] for e in encodings)
    hi = max(_grid_bounds(e)[1] for e in encodings)
    scale = (hi - lo) / ((1 << bits) - 1)
    new["scale"] = [scale]
    new["offset"] = [float(round(lo / scale))]
    return new


class _CacheChain:
    """Rebuild the value-preserving chain from the taps to ``past_*_out`` for the cache.

    Tensors shared with other consumers (the per-head Transpose that also feeds
    the attention Concat) are duplicated so the attention path keeps its exported
    int8 encodings. The present tensor's own producer is rewired in place.

    ``mode="float"`` (codec input): the branch starts at an encoding-less guard
    and stays float16. ``mode="int16"`` (uncompressed cache): every tensor of
    the branch, the taps and the cache I/O get one explicit 16-bit grid (the
    union of the heads' re-gridded tap ranges), so nothing is converted.
    """

    def __init__(
        self, index: _GraphIndex, act: dict[str, dict[str, Any]], mode: str
    ) -> None:
        if mode not in ("float", "int16"):
            raise ValueError(f"Unknown cache chain mode {mode!r}.")
        self.index = index
        self.act = act
        self.mode = mode
        self.grid: dict[str, Any] | None = None
        self.edits: list[EncodingEdit] = []
        self.taps: list[str] = []
        self.tap_ops: list[str] = []
        self.new_nodes: list[onnx.NodeProto] = []
        self._copies: dict[str, str] = {}

    def build(self, present: str) -> None:
        if self.index.consumers.get(present):
            raise ValueError(f"{present} is consumed inside the graph; cannot rewire.")
        if self.mode == "int16":
            taps = self._collect_taps(present)
            missing = [t for t in taps if t not in self.act]
            if missing:
                raise ValueError(
                    f"No encoding on KV tap(s) {missing}; cannot store exactly."
                )
            self.grid = union_grid([regrid_encoding(self.act[t]) for t in taps])
            self.grid.pop("name", None)
        self._copy(present, chain_consumer=None)

    def _collect_taps(self, name: str) -> list[str]:
        producer = self.index.producer.get(name)
        if producer is None or producer.op_type not in _VALUE_PRESERVING:
            return [name]
        taps: list[str] = []
        for i in _data_inputs(producer):
            if i not in self.index.initializers:
                taps += [t for t in self._collect_taps(i) if t not in taps]
        return taps

    def _with_grid(self, name: str) -> dict[str, Any]:
        assert self.grid is not None
        return {"name": name, **self.grid}

    def _copy(self, name: str, chain_consumer: onnx.NodeProto | None) -> str:
        producer = self.index.producer.get(name)
        if producer is None or producer.op_type not in _VALUE_PRESERVING:
            return self._tap(name, producer)
        if name in self._copies:
            return self._copies[name]
        data = _data_inputs(producer)
        new_data = [
            i if i in self.index.initializers else self._copy(i, producer) for i in data
        ]
        shared = any(
            c is not chain_consumer for c in self.index.consumers.get(name, [])
        )
        if not shared:
            for i in range(len(data)):
                producer.input[i] = new_data[i]
            if self.mode == "int16":
                self.edits.append(
                    EncodingEdit(
                        name,
                        "set",
                        f"cache write path {producer.op_type}; stored on the shared 16-bit grid",
                        self.act.get(name, {}),
                        self._with_grid(name),
                    )
                )
            elif name in self.act:
                self.edits.append(
                    EncodingEdit(
                        name,
                        "drop",
                        f"cache write path {producer.op_type}; stays float16",
                        self.act[name],
                    )
                )
            self._copies[name] = name
            return name
        copy_name = f"{name}_tq_cache"
        node = copy.deepcopy(producer)
        node.name = f"{producer.name}_tq_cache" if producer.name else ""
        for i in range(len(data)):
            node.input[i] = new_data[i]
        del node.output[:]
        node.output.append(copy_name)
        self.new_nodes.append(node)
        after: dict[str, Any] = {"copy": copy_name}
        if self.mode == "int16":
            after["encoding"] = self._with_grid(copy_name)
        self.edits.append(
            EncodingEdit(
                name,
                "duplicate",
                f"shared {producer.op_type} output; attention keeps this int8 "
                f"encoding, the cache reads {copy_name}",
                self.act.get(name, {}),
                after,
            )
        )
        self._copies[name] = copy_name
        return copy_name

    def _tap(self, name: str, producer: onnx.NodeProto | None) -> str:
        """Re-grid the tap to 16 bits; in float mode also start the guard.

        ``Max(x, x)`` is exact and carries no encoding, so the float-fallback
        quantizer keeps the branch float16 from there. Plain copies of the
        layout ops are not enough: when such a copy is a no-op (token graphs),
        the converter folds it away and the cache lands on the int8 attention path.
        In int16 mode every tensor carries the explicit shared grid instead.
        """
        guard = f"{name}_tq_cache" if self.mode == "float" else name
        if name in self.taps:
            return guard
        self.taps.append(name)
        self.tap_ops.append(producer.op_type if producer is not None else "graph_input")
        enc = self.act.get(name)
        if self.mode == "int16":
            new = self._with_grid(name)
            if enc != new:
                self.edits.append(
                    EncodingEdit(
                        name,
                        "regrid",
                        "KV tap held at int8 by the KV/16x8 rules; 16-bit grid shared "
                        "by every head of this cache tensor",
                        enc or {},
                        new,
                    )
                )
            return name
        if enc is not None and int(enc["bw"]) == 8:
            self.edits.append(
                EncodingEdit(
                    name,
                    "regrid",
                    "KV tap held at int8 by the KV/16x8 rules; same range at 16 bits",
                    enc,
                    regrid_encoding(enc),
                )
            )
        self.new_nodes.append(
            helper.make_node("Max", [name, name], [guard], name=f"{guard}_guard")
        )
        self.edits.append(
            EncodingEdit(
                guard,
                "guard",
                "exact float no-op Max(x, x) without encoding; starts the float16 "
                "cache branch so the converter cannot fold it into the int8 path",
                {},
                {"source": name},
            )
        )
        return guard


def _read_path(
    index: _GraphIndex,
    past_in: str,
    act: dict[str, dict[str, Any]],
    grid: dict[str, Any] | None,
) -> tuple[list[EncodingEdit], list[str]]:
    """Edit the cache input side; report the attention-side boundary.

    Float mode drops the cache input's encoding and leaves the per-head Slice
    encodings alone: fed by a float tensor QAIRT keeps them float16 and converts
    to int8 at the first value-preserving op that also takes new-token data,
    the attention Concat consumed by the 16x8 MatMul. int16 mode sets the input
    and every tensor derived only from it to the shared grid, so the same
    Concat is the only int8 conversion. The Concat's encoding is kept and reported.
    """
    edits: list[EncodingEdit] = []
    kept: set[str] = set()

    def edit(name: str, what: str) -> None:
        if grid is None:
            if name in act:
                edits.append(EncodingEdit(name, "drop", what, act[name]))
        else:
            edits.append(
                EncodingEdit(
                    name,
                    "set",
                    f"{what}; shared 16-bit grid",
                    act.get(name, {}),
                    {"name": name, **grid},
                )
            )

    edit(past_in, "KV cache input")
    derived = {past_in}
    stack = [past_in]
    while stack:
        name = stack.pop()
        for consumer in index.consumers.get(name, []):
            if consumer.op_type not in _VALUE_PRESERVING:
                continue
            data = [i for i in _data_inputs(consumer) if i not in index.initializers]
            if name not in data:
                continue
            if not all(i in derived for i in data):
                kept.update(o for o in consumer.output if o in act)
                continue
            for out in consumer.output:
                if out not in derived:
                    derived.add(out)
                    stack.append(out)
                    if grid is not None:
                        edit(
                            out,
                            f"{consumer.op_type} output derived from the cache input",
                        )
    return edits, sorted(kept)


def _apply_edits(
    encodings: dict[str, Any], edits: list[EncodingEdit]
) -> dict[str, Any]:
    new = copy.deepcopy(encodings)
    dropped = {e.tensor for e in edits if e.action == "drop"}
    replaced = {e.tensor: e.after for e in edits if e.action in ("regrid", "set")}
    added = [
        e.after["encoding"]
        for e in edits
        if e.action == "duplicate" and e.after and "encoding" in e.after
    ]
    acts = []
    for enc in encodings["activation_encodings"]:
        name = enc["name"]
        if name in dropped:
            continue
        acts.append(copy.deepcopy(replaced.pop(name, enc)))
    acts += [copy.deepcopy(v) for v in replaced.values()] + added
    new["activation_encodings"] = acts
    return new


def _insert_codec(
    decode: Subgraph,
    encode: Subgraph,
    config: TurboQuantConfig,
    spec: KVCodecSpec,
    io: CodecTensorIO,
    opset: int,
) -> tuple[str, list[onnx.ValueInfoProto], list[onnx.ValueInfoProto]]:
    """Build both subgraphs for ``io``; return the restored past name and new I/O."""
    kind, layer, heads = io.kind, io.layer, io.num_kv_heads
    prefix = f"tq_{kind}_{layer}_"
    present = f"past_{kind}_{layer}_out"
    restored = prefix + "restored"
    decode.extend(
        decode_subgraph(
            config,
            spec,
            io.packed_in,
            io.norm_in,
            restored,
            (heads, 1),
            io.past_tokens,
            prefix + "dec_",
        )
    )
    if kind == "key":
        # Stored with the token axis at -2; hub keys are (heads, 1, head_dim, tokens).
        decode.node(
            "Transpose", [restored], [prefix + "restored_hub"], perm=[0, 1, 3, 2]
        )
        restored = prefix + "restored_hub"
    src = present
    if kind == "key":
        encode.node(
            "Transpose", [present], [prefix + "present_tokens_last"], perm=[0, 1, 3, 2]
        )
        src = prefix + "present_tokens_last"
    encode.extend(
        encode_subgraph(
            config,
            spec,
            src,
            io.packed_out,
            io.norm_out,
            (heads, 1),
            io.new_tokens,
            prefix + "enc_",
            opset=opset,
        )
    )
    nbytes = io.packed_bytes
    new_inputs = [
        helper.make_tensor_value_info(
            io.packed_in, TensorProto.UINT8, [heads, 1, io.past_tokens, nbytes]
        ),
        helper.make_tensor_value_info(
            io.norm_in, TensorProto.FLOAT, [heads, 1, io.past_tokens, 1]
        ),
    ]
    new_outputs = [
        helper.make_tensor_value_info(
            io.packed_out, TensorProto.UINT8, [heads, 1, io.new_tokens, nbytes]
        ),
        helper.make_tensor_value_info(
            io.norm_out, TensorProto.FLOAT, [heads, 1, io.new_tokens, 1]
        ),
    ]
    return restored, new_inputs, new_outputs


def apply_kv_profile(
    model: onnx.ModelProto,
    encodings: dict[str, Any],
    config: TurboQuantConfig,
    seq_len: int,
    context_length: int,
) -> SurgeryResult:
    """Return the part with ``config`` applied; inputs are not mutated.

    ``model`` may be loaded without external data; initializers are left untouched.
    """
    if not config.modifies_graph:
        raise ValueError(f"Profile '{config.profile}' keeps the exported KV path.")
    if not 0 < seq_len < context_length:
        raise ValueError("Need 0 < seq_len < context_length.")
    model = copy.deepcopy(model)
    graph = model.graph
    opset = max(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
        default=OPSET,
    )
    index = _GraphIndex(graph)
    act = {e["name"]: e for e in encodings["activation_encodings"]}
    past = context_length - seq_len
    d = config.block_size
    outputs = {o.name: o for o in graph.output}

    decode = Subgraph()
    encode = Subgraph()
    removed_inputs: set[str] = set()
    new_inputs: list[onnx.ValueInfoProto] = []
    new_outputs: list[onnx.ValueInfoProto] = []
    cache_nodes: dict[str, list[onnx.NodeProto]] = {}  # present name -> copies
    renames: dict[str, str] = {}
    result = SurgeryResult(model, encodings)

    for value in list(graph.input):
        match = KV_INPUT.fullmatch(value.name)
        if match is None:
            continue
        kind, layer = match.group(1), int(match.group(2))
        spec = config.key if kind == "key" else config.value
        if not spec.modifies_graph:
            continue
        dims = _dims(value)
        heads = int(dims[0])
        head_dim = int(dims[2] if kind == "key" else dims[3])
        if head_dim != d or int(dims[1]) != 1:
            raise ValueError(f"{value.name} has unsupported shape {dims}.")
        present = f"past_{kind}_{layer}_out"
        if present not in outputs:
            raise ValueError(f"{value.name} has no matching graph output {present}.")

        chain = _CacheChain(index, act, "int16" if spec.is_int16 else "float")
        chain.build(present)
        read_edits, kept = _read_path(index, value.name, act, chain.grid)
        result.edits += read_edits + chain.edits
        cache_nodes[present] = chain.new_nodes
        result.paths.append(
            KVPath(kind, layer, tuple(chain.taps), tuple(chain.tap_ops), tuple(kept))
        )
        if spec.is_int16:
            continue

        scalar = "scale" if config.precomputed_norm else "norm"
        io = CodecTensorIO(
            kind=kind,
            layer=layer,
            packed_in=f"tq_{kind}_{layer}_packed_in",
            norm_in=f"tq_{kind}_{layer}_{scalar}_in",
            packed_out=f"tq_{kind}_{layer}_packed_out",
            norm_out=f"tq_{kind}_{layer}_{scalar}_out",
            num_kv_heads=heads,
            past_tokens=past,
            new_tokens=seq_len,
            packed_bytes=packed_nbytes(d, spec.bits),
        )
        result.codec_io.append(io)
        restored, ins, outs = _insert_codec(decode, encode, config, spec, io, opset)
        renames[value.name] = restored
        removed_inputs.add(value.name)
        new_inputs += ins
        new_outputs += outs

    if not result.paths:
        raise ValueError("No past_{key,value}_<L>_in inputs matched the profile.")

    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in renames:
                node.input[i] = renames[name]
    replaced_outputs = {f"past_{io.kind}_{io.layer}_out" for io in result.codec_io}
    kept_inputs = [v for v in graph.input if v.name not in removed_inputs]
    kept_outputs = [v for v in graph.output if v.name not in replaced_outputs]

    del graph.input[:]
    graph.input.extend(kept_inputs + new_inputs)
    del graph.output[:]
    graph.output.extend(kept_outputs + new_outputs)
    original_nodes = list(graph.node)
    del graph.node[:]
    # Cache copies read only taps, so placing them just before the present's
    # producer (which follows every tap) keeps the graph topologically sorted.
    rebuilt: list[onnx.NodeProto] = []
    for node in original_nodes:
        for out in node.output:
            rebuilt.extend(cache_nodes.pop(out, []))
        rebuilt.append(node)
    graph.node.extend(decode.nodes + rebuilt + encode.nodes)
    existing = {init.name for init in graph.initializer}
    for sg in (decode, encode):
        for name, tensor in sg.initializers.items():
            if name not in existing:
                graph.initializer.append(tensor)
                existing.add(name)

    result.encodings = _apply_edits(encodings, result.edits)
    return result
