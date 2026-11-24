# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Optional exact replay acceleration of the existing offline surrogate.

This is not an HTP implementation or a new numerical model. Both key codecs
are evaluated and a device-side layer flag selects their output. The original
FP32 GEMM/reduction and FP16 boundary semantics remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from allocation_offline_model import Attention, Model, codec, rotation, rounded
from benchmark_asymmetric_once import ARTIFACTS, digest, read, save_new
from benchmark_nll_allocation import STATIC

from qai_hub_models.models.templates.llm.turboquant.structured_k8 import coefficients

PREPARED = ARTIFACTS / "structured_k8_nll_allocation_20261013/offline/prepared_w4a16.pt"


class DynamicAttention(Attention):
    """Same tiled attention, with a runtime choice between two key codecs."""

    def __call__(
        self, values: list[torch.Tensor], layers: set[int]
    ) -> list[torch.Tensor]:
        q = torch.cat(values[:16], dim=0).squeeze(1)
        key, value = values[16].squeeze(1), values[17].squeeze(1)
        k = torch.where(
            self.flags[self.layer],
            codec(key, 8, 42, True, structured_params=self.params),
            codec(key, 4, 42),
        ).repeat_interleave(2, 0)
        v = codec(value, 2, 542).repeat_interleave(2, 0)
        outputs = []
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
            if end < 1024:
                padding = torch.full(
                    (16, end - start, 1024 - end), -100.0, device=q.device
                )
                masked = torch.cat((self.affine(padding, 1), masked), -1)
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
        return list(torch.cat(outputs, 1).unsqueeze(1).split(1, 0))


def load() -> Model:
    identity = read(PREPARED.parent / "identity.json")
    assert digest(PREPARED) == identity["prepared_model_sha256"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    # Only a hash-checked, trusted artifact created by our previous experiment.
    return torch.load(PREPARED, weights_only=False, map_location="cuda")


class GraphReplay:
    def __init__(self, model: Model, tokens: np.ndarray) -> None:
        self.model = model
        self.length = len(tokens)
        if self.length not in (256, 1024):
            raise ValueError("Only frozen coarse/full lengths 256/1024 are supported")
        self.tokens = torch.tensor(tokens.astype(np.int64), device=model.device)[None]
        self.flags = torch.zeros(28, dtype=torch.bool, device=model.device)
        self.params = torch.tensor(
            coefficients().astype(np.float16).astype(np.float32), device=model.device
        )
        self.original_macros = [
            {layer: getattr(m, "reference", m) for layer, m in part.macros.items()}
            for part in model.parts
        ]
        for part in model.parts:
            part.macros = {}
        for part, macros in zip(model.parts, self.original_macros, strict=True):
            for layer, macro in macros.items():
                dynamic = DynamicAttention.__new__(DynamicAttention)
                dynamic.__dict__ = copy.copy(macro.__dict__)
                dynamic.flags = self.flags
                dynamic.params = self.params
                dynamic.reference = macro
                part.macros[layer] = dynamic
        self.dynamic_macros = [part.macros for part in model.parts]
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):
                self.forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.output = self.forward()
        torch.cuda.synchronize()

    @torch.inference_mode()
    def forward(self) -> torch.Tensor:
        values = {
            "input_ids": self.tokens,
            "position_ids_cos": self.model.rope[0, : self.length][None, None],
            "position_ids_sin": self.model.rope[1, : self.length][None, None],
        }
        for part in self.model.parts:
            values.update(part(values, set()))
        return values[self.model.parts[-1].outputs[0]].reshape(self.length, -1)[:-1]

    @torch.inference_mode()
    def logits(self, tokens: np.ndarray, layers: list[int]) -> torch.Tensor:
        if len(tokens) != self.length or len(set(layers)) != len(layers):
            raise ValueError("Invalid replay tokens or duplicate layer")
        flag = np.zeros(28, dtype=np.bool_)
        flag[layers] = True
        self.tokens.copy_(torch.from_numpy(tokens.astype(np.int64))[None])
        self.flags.copy_(torch.from_numpy(flag))
        self.graph.replay()
        return self.output

    def score(self, tokens: np.ndarray, layers: list[int]) -> dict:
        start = time.monotonic()
        logits = self.logits(tokens, layers)
        target = torch.tensor(tokens[1:].astype(np.int64), device=self.model.device)
        losses = F.cross_entropy(logits, target, reduction="none")
        if not torch.isfinite(losses).all():
            raise ValueError("Non-finite replay NLL")
        return {
            "nll_sum": float(losses.double().sum()),
            "scored_tokens": len(tokens) - 1,
            "mean_nll": float(losses.double().mean()),
            "elapsed_s": time.monotonic() - start,
        }

    def eager(self, tokens: np.ndarray, layers: list[int]) -> tuple[dict, torch.Tensor]:
        try:
            for part, macros in zip(
                self.model.parts, self.original_macros, strict=True
            ):
                part.macros = macros
            if len(tokens) == 1024:
                return self.model.score(tokens, layers)
            start = time.monotonic()
            with torch.inference_mode():
                values = {
                    "input_ids": torch.tensor(
                        tokens.astype(np.int64), device=self.model.device
                    )[None],
                    "position_ids_cos": self.model.rope[0, : len(tokens)][None, None],
                    "position_ids_sin": self.model.rope[1, : len(tokens)][None, None],
                }
                for part in self.model.parts:
                    values.update(part(values, set(layers)))
                logits = values[self.model.parts[-1].outputs[0]].reshape(
                    len(tokens), -1
                )[:-1]
                loss = F.cross_entropy(
                    logits,
                    torch.tensor(tokens[1:].astype(np.int64), device=self.model.device),
                    reduction="none",
                )
                return {
                    "nll_sum": float(loss.double().sum()),
                    "mean_nll": float(loss.double().mean()),
                    "scored_tokens": len(tokens) - 1,
                    "elapsed_s": time.monotonic() - start,
                }, logits
        finally:
            for part, macros in zip(self.model.parts, self.dynamic_macros, strict=True):
                part.macros = macros


def preflight(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    source = ARTIFACTS / "structured_k8_nll_allocation_20261013/assets"
    tokens = [np.fromfile(source / f"calibration_{i}.bin", dtype="<i4") for i in (0, 1)]
    model = load()
    print("Loaded frozen W4A16 surrogate", flush=True)
    replays = {n: GraphReplay(model, tokens[0][:n]) for n in (256, 1024)}
    print("Captured 256/1024 dynamic-mask CUDA graphs", flush=True)
    rows = []
    for name, layers in STATIC.items():
        for i, ids in enumerate(tokens):
            for n, replay in replays.items():
                eager, expected = replay.eager(ids[:n], layers)
                actual = replay.logits(ids[:n], layers)
                exact = torch.equal(expected, actual)
                max_abs = float((expected - actual).abs().max())
                del expected
                measured = [replay.score(ids[:n], layers) for _ in range(3)]
                rows.append(
                    {
                        "mask": name,
                        "document": i,
                        "length": n,
                        "token_sha256": digest(source / f"calibration_{i}.bin"),
                        "bit_exact_logits": exact,
                        "max_abs_logit_error": max_abs,
                        "eager": eager,
                        "replay": measured,
                    }
                )
                print(
                    name,
                    i,
                    n,
                    "exact",
                    exact,
                    "seconds",
                    measured[0]["elapsed_s"],
                    flush=True,
                )
    save_new(
        out,
        {
            "passed": all(r["bit_exact_logits"] for r in rows),
            "cases": rows,
            "prepared_model_sha256": digest(PREPARED),
            "replay_source_sha256": digest(Path(__file__)),
            "surrogate_source_sha256": digest(
                Path(__file__).with_name("allocation_offline_model.py")
            ),
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "limits": "Exactness against the existing offline surrogate only, NOT HTP.",
        },
    )
    assert all(r["bit_exact_logits"] for r in rows), "Do not use nonexact acceleration"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-report", type=Path, required=True)
    preflight(parser.parse_args().preflight_report)
