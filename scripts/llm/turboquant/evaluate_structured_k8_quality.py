# SPDX-License-Identifier: BSD-3-Clause
"""Offline K/QK errors on frozen captures; never used to select coefficients."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from benchmark_asymmetric_once import read, save_new
from benchmark_structured_k8_once import CAPTURES, GROUPS, validate
from optimize_structured_k8 import gaussian_mse
from select_mixed_layers import codec, matrix, rounded

from qai_hub_models.models.templates.llm.turboquant.reference import load_codebook
from qai_hub_models.models.templates.llm.turboquant.structured_k8 import (
    codebook,
    indices,
)


def structured_codec(x: torch.Tensor) -> torch.Tensor:
    magnitude = x.abs().amax(-1, keepdim=True)
    large = magnitude > 256
    pre, post = torch.where(large, 1 / 256, 1.0), torch.where(large, 256.0, 1.0)
    maxpre = rounded(magnitude * pre)
    scaled = rounded(rounded(x * pre) / torch.where(maxpre > 0, maxpre, 1.0))
    length = rounded(
        torch.sqrt(rounded(rounded(scaled * scaled).sum(-1, keepdim=True)))
    )
    unit = rounded(scaled / torch.where(length > 0, length, 1.0))
    norm = rounded(rounded(maxpre * length) * post)
    y = rounded(unit @ matrix(42).T)
    index = torch.tensor(indices(y.numpy(), fp16=True).astype(np.int64))
    selected = torch.tensor(codebook().astype(np.float32))[index]
    correction = rounded(
        torch.sqrt(rounded(rounded(selected * selected).sum(-1, keepdim=True)))
    )
    scale = rounded(norm / torch.where(correction > 0, correction, 1.0))
    return rounded(selected * scale)


def errors(reference: torch.Tensor, result: torch.Tensor) -> dict:
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite quality output")
    ref = reference.double()
    return {
        "sse": float((result.double() - ref).square().sum()),
        "reference_energy": float(ref.square().sum()),
        "count": ref.numel(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    validate(root)
    torch.set_num_threads(4)
    position = torch.arange(31, 1024, 32)
    mask = (
        (torch.arange(1024)[None, :] <= position[:, None])
        .unsqueeze(0)
        .expand(16, -1, -1)
    )
    rows = []
    for document in range(2):
        for layer in range(28):
            with np.load(
                CAPTURES / "samples" / f"validation_{document}_layer{layer:02d}.npz"
            ) as data:
                key = rounded(torch.tensor(data["k"].astype(np.float32)))
                query = rounded(torch.tensor(data["q"][:, position].astype(np.float32)))
            qrot = rounded(query @ matrix(42).T)
            krot = rounded(key @ matrix(42).T)
            ref_scores = rounded(query @ key.repeat_interleave(2, 0).transpose(-1, -2))[
                mask
            ]
            rotated_scores = rounded(
                qrot @ krot.repeat_interleave(2, 0).transpose(-1, -2)
            )[mask]
            for group in GROUPS:
                restored = (
                    structured_codec(key)
                    if layer % 2 and group == GROUPS[1]
                    else codec(key, 8 if layer % 2 else 4, 42)
                )
                scores = rounded(
                    qrot @ restored.repeat_interleave(2, 0).transpose(-1, -2)
                )[mask]
                item = {
                    "group": group,
                    "document": document,
                    "layer": layer,
                    "bits": 8 if layer % 2 else 4,
                    "k_reconstruction": errors(key, rounded(restored @ matrix(42))),
                    "k_rotated": errors(krot, restored),
                    "qk_score": errors(ref_scores, scores),
                    "qk_score_rotated_reference": errors(rotated_scores, scores),
                }
                rows.append(item)
            print("quality", document, layer, flush=True)
    aggregate = {}
    for group in GROUPS:
        aggregate[group] = {}
        for scope in ("all_layers", "k8_layers"):
            totals = defaultdict(lambda: defaultdict(float))
            for row in rows:
                if row["group"] != group or (scope == "k8_layers" and row["bits"] != 8):
                    continue
                for metric in (
                    "k_reconstruction",
                    "k_rotated",
                    "qk_score",
                    "qk_score_rotated_reference",
                ):
                    for key, value in row[metric].items():
                        totals[metric][key] += value
            aggregate[group][scope] = {
                metric: {
                    **values,
                    "mse": values["sse"] / values["count"],
                    "relative_mse": values["sse"] / values["reference_energy"],
                }
                for metric, values in totals.items()
            }
    # This is an analytic nearest-centroid scalar comparison, separate from
    # actual arithmetic indexing, norm/scale correction and real captured KV.
    gaussian = {
        "lloyd_max": gaussian_mse(load_codebook(8, 128)[128:] * np.sqrt(128))[0] / 128,
        "structured_rounded_nearest": gaussian_mse(codebook()[128:] * np.sqrt(128))[0]
        / 128,
        "structured_ideal": read(root / "codebook/optimization.json")[
            "ideal_gaussian_mse"
        ],
    }
    save_new(
        root / "offline_quality.json",
        {
            "aggregate": aggregate,
            "per_layer_document": rows,
            "gaussian_scalar_mse": gaussian,
            "reference": "Same W4A16 FP16-KV captures, not full FP16 weights. Q already includes attention scaling; 32 fixed queries/document, causal pre-mask/pre-softmax scores, GQA repeat=2.",
            "limits": "Offline FP16-boundary emulator with FP32 reductions, not HTP bit-exact or end-to-end NLL. Existing validation documents; no fit/selection on these data. All-layer averages include unchanged K4 layers; K8-only reported separately.",
        },
    )


if __name__ == "__main__":
    main()
