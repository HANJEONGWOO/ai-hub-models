# SPDX-License-Identifier: BSD-3-Clause
"""Freeze hardware-aligned plans using captured W4A16 FP16-KV attention.

CPU is offline selection only, never a deployment fallback. FP16 operation
boundaries are emulated with FP32 reductions, not claimed bit-exact to HTP.
No rotation/codebook training, no PPL-driven plan selection.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from benchmark_asymmetric_once import digest, read, save_new

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.reference import (
    DenseQRRotation,
    load_boundaries,
    load_codebook,
)

OPTIONS = ((4, 2), (4, 4), (8, 2), (8, 4))


def rounded(x: torch.Tensor) -> torch.Tensor:
    return x.half().float()


@lru_cache
def matrix(seed: int) -> torch.Tensor:
    return rounded(torch.tensor(DenseQRRotation(seed, 128).matrix().astype(np.float32)))


@lru_cache
def tables(bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = load_codebook(bits, 128).astype(np.float32)
    enc = []
    for index in range(1 << bits):
        mag = abs(index - ((1 << bits) / 2 - 0.5)) - 0.5
        offset = 2 * (int(mag) // 2)
        start = (1 << (bits - 1)) + offset
        slope = c[start + 1] - c[start]
        base = c[start] - offset * slope
        value = np.float16(np.float16(mag * np.float16(slope)) + np.float16(base))
        enc.append(float(value) if index >= 1 << (bits - 1) else -float(value))
    return (
        rounded(torch.tensor(c)),
        torch.tensor(enc),
        rounded(torch.tensor(load_boundaries(bits, 128).astype(np.float32))),
    )


def codec(x: torch.Tensor, bits: int, seed: int) -> torch.Tensor:
    centroids, enc, boundaries = tables(bits)
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
    y = rounded(unit @ matrix(seed).T)
    indices = torch.bucketize(y.contiguous(), boundaries, right=False)
    selected = enc[indices]
    correction = rounded(
        torch.sqrt(rounded(rounded(selected * selected).sum(-1, keepdim=True)))
    )
    scale = rounded(norm / torch.where(correction > 0, correction, 1.0))
    result = rounded(centroids[indices] * scale)
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite reconstructed KV")
    return result


def affine(x: torch.Tensor, grid: list[dict]) -> torch.Tensor:
    if any(g["dtype"] != "INT" or g["bw"] != 16 for g in grid):
        raise ValueError("Unexpected attention calibration")
    scale = torch.tensor([g["scale"][0] for g in grid])[:, None, None]
    offset = torch.tensor([g["offset"][0] for g in grid])[:, None, None]
    return (torch.round(x / scale - offset).clamp(0, 65535) + offset) * scale


def attention(
    sample: dict, grids: list[dict], k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    # Same 32 query positions and right-aligned AR128 tile convention as the
    # source capture experiment. q already includes the model's attention scale.
    positions = torch.arange(31, 1024, 32)
    q = rounded(sample["q"][:, positions] @ matrix(42).T)
    k, v = k.repeat_interleave(2, 0), v.repeat_interleave(2, 0)
    outputs = []
    for chunk in range(8):
        chosen = positions // 128 == chunk
        pos, end = positions[chosen], (chunk + 1) * 128
        scores = affine(
            rounded(q[:, chosen] @ k[:, :end].transpose(-1, -2)),
            [g["score"] for g in grids],
        )
        mask = torch.where(torch.arange(end)[None, :] <= pos[:, None], 0.0, -100.0)
        masked = affine(scores + mask, [g["masked"] for g in grids])
        prob = rounded(
            affine(torch.softmax(masked, dim=-1), [g["prob"] for g in grids])
        )
        total = None
        for edge in (256, 512, 768, 1024):
            start, stop = max(0, edge - 256 - (1024 - end)), max(0, edge - (1024 - end))
            part = (
                rounded(prob[:, :, start:stop] @ v[:, start:stop])
                if start < stop
                else torch.zeros((*prob.shape[:-1], 128))
            )
            total = part if total is None else rounded(total + part)
        outputs.append(rounded(total @ matrix(542)))
    return torch.cat(outputs, 1)


def choose(
    costs: list[list[float]], k8_count: int, v4_count: int
) -> tuple[tuple[int, int], ...]:
    """Exact quota-constrained assignment for additive layer MSE, lexical ties."""
    states = {(0, 0): (0.0, ())}
    for row in costs:
        next_states = {}
        for (nk, nv), (cost, path) in states.items():
            for pair, error in zip(OPTIONS, row, strict=True):
                key = nk + (pair[0] == 8), nv + (pair[1] == 4)
                if key[0] > k8_count or key[1] > v4_count:
                    continue
                candidate = cost + error, (*path, pair)
                if key not in next_states or candidate < next_states[key]:
                    next_states[key] = candidate
        states = next_states
    return states[k8_count, v4_count][1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    source = args.captures
    manifest = read(source / "samples/manifest.json")
    for file, key in (
        ("data_manifest.json", "data_manifest_sha256"),
        ("capture_identity.json", "capture_identity_sha256"),
        ("samples/encodings.json", "encodings_sha256"),
    ):
        if digest(source / file) != manifest[key]:
            raise ValueError(f"Changed source capture metadata: {file}")
    encodings = read(source / "samples/encodings.json")
    selected = {
        n: h
        for n, h in manifest["sha256"].items()
        if n.startswith(("train_", "validation_"))
    }
    for name, checksum in selected.items():
        if digest(source / "samples" / name) != checksum:
            raise ValueError(f"Changed source sample: {name}")
    save_new(
        args.out / "selection_protocol.json",
        {
            "source": str(source),
            "samples_sha256": selected,
            "data_manifest_sha256": manifest["data_manifest_sha256"],
            "selection": "4 existing train documents, 32 queries/window, 28 layers; additive absolute Attention output MSE; exact joint K8/V4 quota DP; lexical ties",
            "validation": "2 existing validation documents, never used to modify plans",
            "ppl": "legacy 4 WikiText windows, not used for layer allocation",
            "options": OPTIONS,
            "seed": {"key": 42, "value": 542},
            "performance_sessions_per_group_condition": 1,
            "limitations": "local teacher-forced captured attention; not end-to-end propagation; FP32 reductions with FP16 boundary rounding; no global quality optimum claim",
        },
    )
    errors = {}
    for split, count in (("train", 4), ("validation", 2)):
        costs = []
        for layer in range(28):
            sums = np.zeros(len(OPTIONS))
            for window in range(count):
                with np.load(
                    source / "samples" / f"{split}_{window}_layer{layer:02d}.npz"
                ) as data:
                    sample = {
                        k: torch.tensor(data[k].astype(np.float32)) for k in data.files
                    }
                keys = {b: codec(sample["k"], b, 42) for b in (4, 8)}
                values = {b: codec(sample["v"], b, 542) for b in (2, 4)}
                target = sample["o"][:, 31::32]
                for i, (kb, vb) in enumerate(OPTIONS):
                    result = attention(
                        sample, encodings[str(layer)], keys[kb], values[vb]
                    )
                    sums[i] += float((result - target).double().square().mean())
            costs.append((sums / count).tolist())
            print(split, layer, costs[-1], flush=True)
        errors[split] = costs
        if split == "train":
            for budget, nk, nv in ((5, 7, 14), (6, 14, 0)):
                base = get_profile(f"mixed_k{budget}_v{8 - budget}_uniform")
                config = replace(
                    base,
                    profile=f"mixed_k{budget}_v{8 - budget}_mse",
                    layer_bits=choose(costs, nk, nv),
                    layer_policy="attention_mse",
                )
                save_new(
                    args.out / f"{config.profile}.json",
                    {
                        "profile": config.profile,
                        "policy": config.layer_policy,
                        "bits": config.layer_bits,
                        "config_hash": config.config_hash(),
                        "config": config.to_dict(),
                    },
                )
    summary = {}
    for budget in (5, 6):
        for policy in ("uniform", "mse"):
            name = f"mixed_k{budget}_v{8 - budget}_{policy}"
            config = get_profile(
                name,
                layer_plan=read(args.out / f"{name}.json") if policy == "mse" else None,
            )
            summary[name] = {
                split: sum(
                    row[OPTIONS.index(pair)]
                    for row, pair in zip(costs, config.layer_bits, strict=True)
                )
                / 28
                for split, costs in errors.items()
            }
    save_new(
        args.out / "selection_results.json",
        {"options": OPTIONS, "layer_mse": errors, "attention_mse": summary},
    )


if __name__ == "__main__":
    main()
