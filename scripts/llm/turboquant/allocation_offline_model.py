# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Offline W4A16 graph replay for allocation NLL, NOT a deployment fallback.

Uses the exported checkpoint's weights and frozen per-tensor/per-channel
encodings, not a stock HF float model. Attention is replayed layer-wise over
the full teacher-forced document with AR128 right-aligned tile256 reductions.
FP32 reductions/fake affine quantization approximate HTP; no bit-exact claim.
The full downstream model is recomputed, so this is not local Attention MSE.
"""

from __future__ import annotations

import heapq
import json
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

from qai_hub_models.models.templates.llm.turboquant.reference import (
    DenseQRRotation,
    load_boundaries,
    load_codebook,
)
from qai_hub_models.models.templates.llm.turboquant.structured_k8 import (
    coefficients,
    region_boundary,
)


def rounded(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(torch.float32)


@lru_cache
def rotation(seed: int, device: str) -> torch.Tensor:
    return torch.tensor(
        DenseQRRotation(seed, 128)
        .matrix()
        .astype(np.float32)
        .astype(np.float16)
        .astype(np.float32),
        device=device,
    )


@lru_cache
def tables(bits: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = load_codebook(bits, 128)
    encoded = []
    for index in range(1 << bits):
        mag = abs(index - ((1 << bits) / 2 - 0.5)) - 0.5
        pair = 2 * (int(mag) // 2)
        start = (1 << (bits - 1)) + pair
        slope = c[start + 1] - c[start]
        base = c[start] - pair * slope
        value = np.float16(np.float16(mag * np.float16(slope)) + np.float16(base))
        encoded.append(float(value) if index >= 1 << (bits - 1) else -float(value))
    return (
        torch.tensor(c.astype(np.float16).astype(np.float32), device=device),
        torch.tensor(encoded, device=device),
        torch.tensor(
            load_boundaries(bits, 128).astype(np.float16).astype(np.float32),
            device=device,
        ),
    )


def codec(
    x: torch.Tensor,
    bits: int,
    seed: int,
    structured: bool = False,
    *,
    structured_params: torch.Tensor | None = None,
) -> torch.Tensor:
    x = rounded(x)
    maximum = x.abs().amax(-1, keepdim=True)
    pre, post = (
        torch.where(maximum > 256, 1 / 256, 1.0),
        torch.where(maximum > 256, 256.0, 1.0),
    )
    maxpre = rounded(maximum * pre)
    scaled = rounded(rounded(x * pre) / torch.where(maxpre > 0, maxpre, 1.0))
    length = rounded(
        torch.sqrt(rounded(rounded(scaled * scaled).sum(-1, keepdim=True)))
    )
    unit = rounded(scaled / torch.where(length > 0, length, 1.0))
    norm = rounded(rounded(maxpre * length) * post)
    y = rounded(unit @ rotation(seed, str(x.device)).T)
    if structured:
        params = (
            torch.tensor(
                coefficients().astype(np.float16).astype(np.float32), device=x.device
            )
            if structured_params is None
            else structured_params
        )
        magnitude = y.abs()
        region = (magnitude > float(np.float16(region_boundary()))).long()
        offset, step = params[2 * region], params[2 * region + 1]
        reciprocal = rounded(1 / step)
        local = torch.floor(
            rounded(rounded(rounded(magnitude - offset) * reciprocal) + 0.5)
        ).clamp(0, 63)
        positive = rounded(rounded(local * step) + offset)
        selected = torch.where(y >= 0, positive, -positive)
        encoder = selected
    else:
        lut, enc, boundaries = tables(bits, str(x.device))
        index = torch.bucketize(y.contiguous(), boundaries, right=False)
        selected, encoder = lut[index], enc[index]
    correction = rounded(
        torch.sqrt(rounded(rounded(encoder * encoder).sum(-1, keepdim=True)))
    )
    scale = rounded(norm / torch.where(correction > 0, correction, 1.0))
    return rounded(selected * scale)


class Grid:
    def __init__(self, value: dict, device: str, rank: int = 0, axis: int = 0) -> None:
        self.bw, self.dtype = value["bw"], value["dtype"]
        shape = [1] * rank if len(value.get("scale", [1])) > 1 else []
        if shape:
            shape[axis] = -1
        self.scale = torch.tensor(value.get("scale", [1]), device=device).reshape(shape)
        self.offset = torch.tensor(value.get("offset", [0]), device=device).reshape(
            shape
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.dtype == "FLOAT":
            return rounded(x) if self.bw == 16 else x
        return (
            torch.round(x / self.scale - self.offset).clamp(0, (1 << self.bw) - 1)
            + self.offset
        ) * self.scale


class Attention:
    def __init__(
        self,
        layer: int,
        input_names: list[str],
        output_names: list[str],
        grids: list[list[dict]],
        device: str,
    ) -> None:
        self.layer, self.inputs, self.outputs, self.device = (
            layer,
            input_names,
            output_names,
            device,
        )
        self.grids = []
        for stage in range(3):
            entries = [row[stage] for row in grids]
            assert all(e["bw"] == 16 and e["dtype"] == "INT" for e in entries)
            self.grids.append(
                (
                    torch.tensor([e["scale"][0] for e in entries], device=device)[
                        :, None, None
                    ],
                    torch.tensor([e["offset"][0] for e in entries], device=device)[
                        :, None, None
                    ],
                )
            )

    def affine(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        scale, offset = self.grids[stage]
        return (torch.round(x / scale - offset).clamp(0, 65535) + offset) * scale

    def __call__(
        self, values: list[torch.Tensor], layers: set[int]
    ) -> list[torch.Tensor]:
        q = torch.cat(values[:16], dim=0).squeeze(1)
        k, v = values[16].squeeze(1), values[17].squeeze(1)
        k = codec(
            k, 8 if self.layer in layers else 4, 42, self.layer in layers
        ).repeat_interleave(2, 0)
        v = codec(v, 2, 542).repeat_interleave(2, 0)
        outputs = []
        # This matches HTP's 896 past + 128 current layout for every AR128 call,
        # including empty left padding and sequential FP16 partial AV sums.
        for start in range(0, q.shape[1], 128):
            end = min(start + 128, q.shape[1])
            scores = self.affine(
                rounded(q[:, start:end] @ k[:, :end].transpose(-1, -2)), 0
            )
            mask = torch.where(
                torch.arange(end, device=q.device)[None, :]
                <= torch.arange(start, end, device=q.device)[:, None],
                0.0,
                -100.0,
            )
            masked = self.affine(scores + mask, 1)
            # Affine probability quantization includes the (masked) zero padding.
            if end < 1024:
                padding = torch.full(
                    (16, end - start, 1024 - end), -100.0, device=q.device
                )
                padding = self.affine(padding, 1)
                masked = torch.cat((padding, masked), -1)
            probabilities = rounded(self.affine(torch.softmax(masked, -1), 2))
            total = None
            for edge in (256, 512, 768, 1024):
                first, last = (
                    max(0, edge - 256 - (1024 - end)),
                    max(0, edge - (1024 - end)),
                )
                part = (
                    rounded(
                        probabilities[:, :, 1024 - end + first : 1024 - end + last]
                        @ v[:, first:last]
                    )
                    if first < last
                    else torch.zeros(
                        (*q[:, start:end].shape[:-1], 128), device=q.device
                    )
                )
                total = part if total is None else rounded(total + part)
            outputs.append(rounded(total @ rotation(542, self.device)))
        result = torch.cat(outputs, 1).unsqueeze(1)
        return list(result.split(1, 0))


class Part:
    def __init__(self, path: Path, device: str) -> None:
        self.device = device
        model = onnx.load(path, load_external_data=False)
        data = json.loads(path.with_suffix(".encodings").read_text())
        activations = {v["name"]: v for v in data["activation_encodings"]}
        parameters = {v["name"]: v for v in data["param_encodings"]}
        producer = {out: node for node in model.graph.node for out in node.output}
        self.macros = {}
        layers = sorted(
            {
                int(m[1])
                for n in model.graph.node
                if (m := re.fullmatch(r"tq_key_(\d+)_enc_rotated", n.output[0]))
            }
        )
        for layer in layers:
            inputs, outputs, grids = [], [], []
            for head in range(8):
                for group in range(2):
                    prefix = f"tq_attn_{layer}_tile0_head{head}_q{group}_"
                    inputs.append(f"tq_attn_{layer}_head{head}_q{group}_rotated")
                    prob = producer[prefix + "prob"].input[0]
                    masked = producer[prob].input[0]
                    grids.append(
                        [
                            activations[prefix + "score"],
                            activations[masked],
                            activations[prob],
                        ]
                    )
                    total = f"tq_attn_{layer}_tile768_head{head}_q{group}_sum"
                    candidates = [
                        n
                        for n in model.graph.node
                        if n.op_type == "MatMul" and n.input[0] == total
                    ]
                    assert len(candidates) == 1
                    outputs.append(candidates[0].output[0])
            inputs += [f"tq_key_{layer}_present_tokens_last", f"past_value_{layer}_out"]
            macro = Attention(layer, inputs, outputs, grids, device)
            self.macros[layer] = macro
            node = onnx.helper.make_node(
                "OfflineAttention",
                inputs,
                outputs,
                name=f"offline_attention_{layer}",
                layer=layer,
            )
            for output in outputs:
                producer[output] = node
        self.outputs = [
            o.name
            for o in model.graph.output
            if not o.name.startswith(("tq_", "past_"))
        ]
        assert len(self.outputs) == 1
        # Dependency closure and stable topological sort, ignoring unused KV I/O.
        required, pending, nodes = set(), list(self.outputs), {}
        while pending:
            name = pending.pop()
            if name in required:
                continue
            required.add(name)
            if name in producer:
                node = producer[name]
                nodes[node.output[0]] = node
                pending.extend(node.input)
        self.constants, self.cpu_constants = {}, {}
        parameter_consumers = {}
        for node in model.graph.node:
            for name in node.input:
                parameter_consumers.setdefault(name, []).append(node)
        for tensor in model.graph.initializer:
            if tensor.name not in required:
                continue
            array = numpy_helper.to_array(tensor, base_dir=str(path.parent)).copy()
            self.cpu_constants[tensor.name] = array if array.size < 10000 else None
            value = torch.from_numpy(array).to(
                device=device if array.dtype.kind == "f" else "cpu"
            )
            if tensor.name in parameters:
                consumers = parameter_consumers[tensor.name]
                axis = (
                    value.ndim - 1
                    if any(
                        n.op_type == "MatMul" and n.input[1] == tensor.name
                        for n in consumers
                    )
                    else 0
                )
                assert len(parameters[tensor.name].get("scale", [1])) in (
                    1,
                    value.shape[axis],
                ), tensor.name
                value = Grid(parameters[tensor.name], device, value.ndim, axis)(value)
            self.constants[tensor.name] = value
        for key, node in list(nodes.items()):
            if node.op_type == "Constant":
                attribute = node.attribute[0]
                if attribute.name == "value":
                    array = numpy_helper.to_array(attribute.t).copy()
                elif attribute.name in (
                    "value_int",
                    "value_ints",
                    "value_float",
                    "value_floats",
                ):
                    array = np.asarray(
                        onnx.helper.get_attribute_value(attribute),
                        dtype=np.int64 if "int" in attribute.name else np.float32,
                    )
                else:
                    raise NotImplementedError(attribute.name)
                self.cpu_constants[node.output[0]] = array
                self.constants[node.output[0]] = torch.from_numpy(array).to(
                    device=device if array.dtype.kind == "f" else "cpu"
                )
                del nodes[key]
        owner = {out: key for key, node in nodes.items() for out in node.output}
        dependencies = {
            key: {owner[i] for i in node.input if i in owner}
            for key, node in nodes.items()
        }
        consumers = {}
        for key, deps in dependencies.items():
            for dependency in deps:
                consumers.setdefault(dependency, []).append(key)
        ready = [key for key, deps in dependencies.items() if not deps]
        heapq.heapify(ready)
        self.nodes = []
        while ready:
            key = heapq.heappop(ready)
            node = nodes[key]
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if node.op_type in (
                "Slice",
                "Squeeze",
                "Unsqueeze",
                "ReduceMean",
                "Reshape",
            ):
                attrs["constant_inputs"] = [
                    self.cpu_constants[i] for i in node.input[1:]
                ]
            self.nodes.append(
                (node.op_type, list(node.input), list(node.output), attrs)
            )
            for consumer in consumers.get(key, []):
                dependencies[consumer].remove(key)
                if not dependencies[consumer]:
                    heapq.heappush(ready, consumer)
        assert len(self.nodes) == len(nodes), "Cycle in offline graph"
        self.grids = {
            name: Grid(value, device)
            for name, value in activations.items()
            if name in required
        }
        self.uses = Counter(i for _, inputs, _, _ in self.nodes for i in inputs)
        for name in self.outputs:
            self.uses[name] += 1
        self.inputs = {i.name for i in model.graph.input if i.name in required}
        self.identity = {
            "onnx": str(path),
            "nodes": len(self.nodes),
            "parameters_with_encodings": len(parameters),
            "retained_activation_grids": len(self.grids),
            "layers": layers,
        }

    @torch.inference_mode()
    def __call__(
        self, incoming: dict[str, torch.Tensor], k8_layers: set[int]
    ) -> dict[str, torch.Tensor]:
        values = dict(self.constants)
        uses = self.uses.copy()
        for name in self.inputs:
            value = incoming[name]
            values[name] = self.grids[name](value) if name in self.grids else value
        for op, inputs, outputs, attrs in self.nodes:
            x = [values[name] for name in inputs]
            if op == "OfflineAttention":
                results = self.macros[attrs["layer"]](x, k8_layers)
            else:
                results = [self.operation(op, x, attrs)]
            for name, value in zip(outputs, results, strict=True):
                if value.is_floating_point():
                    if name in self.grids:
                        value = self.grids[name](value)
                    elif op in ("Conv", "MatMul") or name.startswith("tq_"):
                        value = rounded(value)
                values[name] = value
            for name in inputs:
                uses[name] -= 1
                if not uses[name] and name not in self.constants:
                    del values[name]
        return {name: values[name] for name in self.outputs}

    @staticmethod
    def operation(op: str, x: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
        if op == "Add":
            return x[0] + x[1]
        if op == "Sub":
            return x[0] - x[1]
        if op == "Mul":
            return x[0] * x[1]
        if op == "Max":
            return torch.maximum(x[0], x[1])
        if op == "Min":
            return torch.minimum(x[0], x[1])
        if op == "Cast":
            return x[0].to(
                {1: torch.float32, 10: torch.float16, 6: torch.int32, 7: torch.int64}[
                    a["to"]
                ]
            )
        if op == "Div":
            return x[0] / x[1]
        if op == "Pow":
            return torch.pow(x[0], x[1])
        if op == "Sqrt":
            return torch.sqrt(x[0])
        if op == "Reciprocal":
            return torch.reciprocal(x[0])
        if op == "Sigmoid":
            return torch.sigmoid(x[0])
        if op == "Identity":
            return x[0]
        if op == "Transpose":
            return x[0].permute(a["perm"])
        if op == "Concat":
            return torch.cat(x, dim=a["axis"])
        if op == "MatMul":
            return x[0] @ x[1]
        if op == "Conv":
            assert list(a.get("pads", [0] * 4)) == [0] * 4 and x[1].shape[-2:] == (1, 1)
            return F.conv2d(
                x[0],
                x[1],
                x[2] if len(x) > 2 else None,
                stride=a.get("strides", [1, 1]),
                groups=a.get("group", 1),
            )
        if op == "Gather":
            assert a.get("axis", 0) == 0
            return x[0][x[1].long()]
        if op == "ReduceMean":
            return x[0].mean(
                tuple(a["constant_inputs"][0].ravel().tolist()),
                keepdim=bool(a.get("keepdims", 1)),
            )
        if op == "Reshape":
            return x[0].reshape(a["constant_inputs"][0].tolist())
        if op in ("Squeeze", "Unsqueeze"):
            axes = a["constant_inputs"][0].ravel().tolist()
            value = x[0]
            for axis in sorted(axes, reverse=op == "Squeeze"):
                value = (
                    value.squeeze(axis) if op == "Squeeze" else value.unsqueeze(axis)
                )
            return value
        if op == "Slice":
            constants = a["constant_inputs"]
            starts, ends = [v.ravel().tolist() for v in constants[:2]]
            axes = (
                constants[2].ravel().tolist()
                if len(constants) > 2
                else range(len(starts))
            )
            steps = (
                constants[3].ravel().tolist()
                if len(constants) > 3
                else [1] * len(starts)
            )
            selected = [slice(None)] * x[0].ndim
            for start, end, axis, step in zip(starts, ends, axes, steps, strict=True):
                selected[axis] = slice(start, end, step)
            return x[0][tuple(selected)]
        raise NotImplementedError(op)


class Model:
    def __init__(self, bundle: Path, assets: Path, device: str = "cuda") -> None:
        self.device = device
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.parts = []
        for i in range(1, 5):
            path = bundle / f"prompt_ar128_cl1024_{i}_of_4.onnx"
            if i == 1 and not path.exists():
                path = next(
                    (
                        bundle.parent.parent
                        / "qwen3_1_7b_w4a16_split/Qwen3_1_7B_Part1_Of_4_w4a16.aimet"
                    ).glob("*.onnx")
                )
            part = Part(path, device)
            self.parts.append(part)
            print("loaded", part.identity, flush=True)
        manifest = json.loads((assets / "assets.json").read_text())
        rope = np.fromfile(assets / manifest["rope"], dtype="<f4").reshape(2, 1024, 64)
        self.rope = torch.tensor(rope, device=device)

    @torch.inference_mode()
    def logits(self, tokens: np.ndarray, layers: list[int]) -> torch.Tensor:
        assert len(tokens) == 1024
        values = {
            "input_ids": torch.tensor(tokens.astype(np.int64), device=self.device)[
                None
            ],
            "position_ids_cos": self.rope[0][None, None],
            "position_ids_sin": self.rope[1][None, None],
        }
        for part in self.parts:
            values.update(part(values, set(layers)))
        value = values[self.parts[-1].outputs[0]]
        return value.reshape(1024, -1)

    def score(self, tokens: np.ndarray, layers: list[int]) -> tuple[dict, torch.Tensor]:
        start = time.monotonic()
        logits = self.logits(tokens, layers)[:-1]
        target = torch.tensor(tokens[1:].astype(np.int64), device=self.device)
        loss = F.cross_entropy(logits, target, reduction="none")
        if not torch.isfinite(loss).all():
            raise ValueError("Non-finite offline model NLL")
        return {
            "nll_sum": float(loss.double().sum()),
            "scored_tokens": 1023,
            "mean_nll": float(loss.double().mean()),
            "elapsed_s": time.monotonic() - start,
        }, logits
