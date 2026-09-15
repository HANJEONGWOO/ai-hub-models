# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Insert TurboQuant KV codecs into an exported delta-KV LLM part (ONNX + AIMET encodings).

For every KV tensor whose codec is POLAR the part's I/O changes as follows
(``P = context_length - seq_len`` past tokens, ``A = seq_len`` new tokens):

- ``past_{kind}_{L}_in`` is removed and replaced by ``tq_{kind}_{L}_packed_in``
  ``uint8 (kv_heads, 1, P, bytes)`` and ``tq_{kind}_{L}_norm_in`` ``float (kv_heads, 1, P, 1)``.
  A decode subgraph restores the float tensor in the original layout and feeds
  the original consumers.
- ``past_{kind}_{L}_out`` stays an internal tensor; an encode subgraph turns it
  into ``tq_{kind}_{L}_packed_out`` / ``tq_{kind}_{L}_norm_out`` (token axis -2).

Weights and every existing activation encoding are reused unchanged. The codec
subgraphs carry no encodings, so a float-fallback quantizer keeps them in float
and inserts the int <-> float conversions at their boundaries. Only the removed
``past_*_in`` encodings are dropped. The codec reads the part's existing present
tensor, which in shipped w4a16 graphs is an int8 activation.

Codec subgraphs have static shapes, so apply this per ``(seq_len, context_length)``.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

import onnx
from onnx import TensorProto, helper

from qai_hub_models.models.templates.llm.turboquant.config import TurboQuantConfig
from qai_hub_models.models.templates.llm.turboquant.export import (
    Subgraph,
    decode_subgraph,
    encode_subgraph,
)
from qai_hub_models.models.templates.llm.turboquant.packing import packed_nbytes

KV_INPUT = re.compile(r"past_(key|value)_(\d+)_in")


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


def _dims(value: onnx.ValueInfoProto) -> list[Any]:
    return [
        d.dim_param if d.dim_param else d.dim_value
        for d in value.type.tensor_type.shape.dim
    ]


def apply_kv_codec(
    model: onnx.ModelProto,
    encodings: dict[str, Any],
    config: TurboQuantConfig,
    seq_len: int,
    context_length: int,
) -> tuple[onnx.ModelProto, dict[str, Any], list[CodecTensorIO]]:
    """Return ``(model, encodings, codec_io)`` with codecs inserted; inputs are not mutated.

    ``model`` may be loaded without external data; initializers are left untouched.
    """
    if not config.enabled:
        raise ValueError(
            f"Profile '{config.profile}' stores no KV tensor with a codec."
        )
    if not 0 < seq_len < context_length:
        raise ValueError("Need 0 < seq_len < context_length.")
    model = copy.deepcopy(model)
    graph = model.graph
    past = context_length - seq_len
    d = config.block_size
    outputs = {o.name: o for o in graph.output}

    decode = Subgraph()
    encode = Subgraph()
    removed_inputs: set[str] = set()
    new_inputs: list[onnx.ValueInfoProto] = []
    new_outputs: list[onnx.ValueInfoProto] = []
    renames: dict[str, str] = {}
    codec_io: list[CodecTensorIO] = []

    for value in list(graph.input):
        match = KV_INPUT.fullmatch(value.name)
        if match is None:
            continue
        kind, layer = match.group(1), int(match.group(2))
        spec = config.key if kind == "key" else config.value
        if not spec.is_polar:
            continue
        dims = _dims(value)
        heads = int(dims[0])
        head_dim = int(dims[2] if kind == "key" else dims[3])
        if head_dim != d or int(dims[1]) != 1:
            raise ValueError(f"{value.name} has unsupported shape {dims}.")
        present = f"past_{kind}_{layer}_out"
        if present not in outputs:
            raise ValueError(f"{value.name} has no matching graph output {present}.")
        nbytes = packed_nbytes(d, spec.bits)
        io = CodecTensorIO(
            kind=kind,
            layer=layer,
            packed_in=f"tq_{kind}_{layer}_packed_in",
            norm_in=f"tq_{kind}_{layer}_norm_in",
            packed_out=f"tq_{kind}_{layer}_packed_out",
            norm_out=f"tq_{kind}_{layer}_norm_out",
            num_kv_heads=heads,
            past_tokens=past,
            new_tokens=seq_len,
            packed_bytes=nbytes,
        )
        codec_io.append(io)
        prefix = f"tq_{kind}_{layer}_"

        restored = f"tq_{kind}_{layer}_restored"
        decode.extend(
            decode_subgraph(
                config,
                spec,
                io.packed_in,
                io.norm_in,
                restored,
                (heads, 1),
                past,
                prefix + "dec_",
            )
        )
        if kind == "key":
            # Stored with the token axis at -2; hub keys are (heads, 1, head_dim, tokens).
            decode.node(
                "Transpose", [restored], [prefix + "restored_hub"], perm=[0, 1, 3, 2]
            )
            restored = prefix + "restored_hub"
        renames[value.name] = restored
        removed_inputs.add(value.name)
        new_inputs += [
            helper.make_tensor_value_info(
                io.packed_in, TensorProto.UINT8, [heads, 1, past, nbytes]
            ),
            helper.make_tensor_value_info(
                io.norm_in, TensorProto.FLOAT, [heads, 1, past, 1]
            ),
        ]

        src = present
        if kind == "key":
            encode.node(
                "Transpose",
                [present],
                [prefix + "present_tokens_last"],
                perm=[0, 1, 3, 2],
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
                seq_len,
                prefix + "enc_",
            )
        )
        new_outputs += [
            helper.make_tensor_value_info(
                io.packed_out, TensorProto.UINT8, [heads, 1, seq_len, nbytes]
            ),
            helper.make_tensor_value_info(
                io.norm_out, TensorProto.FLOAT, [heads, 1, seq_len, 1]
            ),
        ]

    if not codec_io:
        raise ValueError("No past_{key,value}_<L>_in inputs matched the profile.")

    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in renames:
                node.input[i] = renames[name]
    replaced_outputs = {f"past_{io.kind}_{io.layer}_out" for io in codec_io}
    kept_inputs = [v for v in graph.input if v.name not in removed_inputs]
    kept_outputs = [v for v in graph.output if v.name not in replaced_outputs]

    del graph.input[:]
    graph.input.extend(kept_inputs + new_inputs)
    del graph.output[:]
    graph.output.extend(kept_outputs + new_outputs)
    original_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(decode.nodes + original_nodes + encode.nodes)
    existing = {init.name for init in graph.initializer}
    for sg in (decode, encode):
        for name, tensor in sg.initializers.items():
            if name not in existing:
                graph.initializer.append(tensor)
                existing.add(name)

    new_encodings = copy.deepcopy(encodings)
    new_encodings["activation_encodings"] = [
        e for e in encodings["activation_encodings"] if e["name"] not in removed_inputs
    ]
    return model, new_encodings, codec_io
