# SPDX-License-Identifier: BSD-3-Clause
"""Actual FP16 arithmetic scalar MSE and frozen W4A16 K/QK diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from benchmark_asymmetric_once import ARTIFACTS, digest, read, save_new
from evaluate_structured_k8_quality import errors, structured_codec
from optimize_uniform_k8 import integrated_mse
from select_mixed_layers import codec, matrix, rounded

from qai_hub_models.models.templates.llm.turboquant import structured_k8, uniform_k8


def uniform_codec(x: torch.Tensor) -> torch.Tensor:
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
    index = torch.tensor(uniform_k8.indices(y.numpy(), fp16=True).astype(np.int64))
    selected = torch.tensor(uniform_k8.codebook().astype(np.float32))[index]
    correction = rounded(
        torch.sqrt(rounded(rounded(selected * selected).sum(-1, keepdim=True)))
    )
    scale = rounded(norm / torch.where(correction > 0, correction, 1.0))
    return rounded(selected * scale)


def scalar_fp16(module: object) -> dict:
    # Exhaust all nonnegative finite binary16 inputs, integrate original FP64
    # Gaussian over the input-rounding cells, then apply executed index/LUT.
    h = np.arange(0x7C00, dtype=np.uint16).view(np.float16).astype(float)
    edges = np.r_[0, (h[:-1] + h[1:]) / 2, np.inf] * np.sqrt(128)
    indices = module.indices(h, fp16=True)
    centroids = module.codebook()[indices] * np.sqrt(128)
    error = integrated_mse(centroids, edges) / 128
    # -0 chooses the positive sign, unlike reflected nonzero negative codes.
    first0 = -np.expm1(-(edges[1] ** 2) / 2) / np.sqrt(2 * np.pi)
    error += 4 * centroids[0] * first0 / 128
    ideal_indices = module.indices(h)
    return {
        "gaussian_mse_fp16_input_index_centroid": error,
        "finite_nonnegative_fp16_inputs": len(h),
        "index_different_from_float64_rule": int(
            np.count_nonzero(indices != ideal_indices)
        ),
        "note": "Integrated over FP16 input rounding cells, separate rounded reciprocal/arithmetic and decoder centroid. Norm/scales are vector-level and evaluated separately.",
    }


def quality(root: Path) -> None:
    if (root / "offline_quality.json").exists():
        raise FileExistsError(root / "offline_quality.json")
    torch.set_num_threads(4)
    position = torch.arange(31, 1024, 32)
    mask = (
        (torch.arange(1024)[None, :] <= position[:, None])
        .unsqueeze(0)
        .expand(16, -1, -1)
    )
    rows, hashes = [], {}
    groups = {"uniform": uniform_codec, "structured": structured_codec}
    for document in range(2):
        for layer in range(28):
            path = (
                ARTIFACTS
                / "k_rotation_quality_20261006/samples"
                / f"validation_{document}_layer{layer:02d}.npz"
            )
            hashes[str(path)] = digest(path)
            with np.load(path) as data:
                key = rounded(torch.tensor(data["k"].astype(np.float32)))
                query = rounded(torch.tensor(data["q"][:, position].astype(np.float32)))
            qrot, krot = rounded(query @ matrix(42).T), rounded(key @ matrix(42).T)
            ref = rounded(query @ key.repeat_interleave(2, 0).transpose(-1, -2))[mask]
            refrot = rounded(qrot @ krot.repeat_interleave(2, 0).transpose(-1, -2))[
                mask
            ]
            for name, function in groups.items():
                restored = function(key) if layer % 2 else codec(key, 4, 42)
                scores = rounded(
                    qrot @ restored.repeat_interleave(2, 0).transpose(-1, -2)
                )[mask]
                rows.append(
                    {
                        "group": name,
                        "document": document,
                        "layer": layer,
                        "bits": 8 if layer % 2 else 4,
                        "k_reconstruction": errors(key, rounded(restored @ matrix(42))),
                        "k_rotated": errors(krot, restored),
                        "qk_score": errors(ref, scores),
                        "qk_score_rotated_reference": errors(refrot, scores),
                    }
                )
            print("quality", document, layer, flush=True)
    aggregate = {}
    for name in groups:
        aggregate[name] = {}
        for scope in ("all_layers", "k8_layers"):
            totals = defaultdict(lambda: defaultdict(float))
            for row in rows:
                if row["group"] != name or (scope == "k8_layers" and row["bits"] != 8):
                    continue
                for metric in (
                    "k_reconstruction",
                    "k_rotated",
                    "qk_score",
                    "qk_score_rotated_reference",
                ):
                    for key, value in row[metric].items():
                        totals[metric][key] += value
            aggregate[name][scope] = {
                m: {
                    **v,
                    "mse": v["sse"] / v["count"],
                    "relative_mse": v["sse"] / v["reference_energy"],
                }
                for m, v in totals.items()
            }
    gaussian = {}
    synthetic = rounded(
        torch.tensor(
            np.random.default_rng(20261020).normal(size=(16384, 128)).astype(np.float32)
        )
    )
    for name, module in (("uniform", uniform_k8), ("structured", structured_k8)):
        c = module.codebook()[128:] * np.sqrt(128)
        edges = np.r_[0, (c[:-1] + c[1:]) / 2, np.inf]
        stats = errors(synthetic, rounded(groups[name](synthetic) @ matrix(42)))
        gaussian[name] = {
            **scalar_fp16(module),
            "nearest_centroid_mse": integrated_mse(c, edges) / 128,
            "synthetic_final_kv": {**stats, "mse": stats["sse"] / stats["count"]},
            "synthetic_seed": 20261020,
        }
    save_new(
        root / "offline_quality.json",
        {
            "aggregate": aggregate,
            "per_layer_document": rows,
            "gaussian": gaussian,
            "capture_sha256": hashes,
            "optimization": read(root / "codebook/optimization.json"),
            "reference": "Same W4A16 FP16-KV captures, fixed 32 causal query positions per document, GQA=2; no data fitting",
            "limits": "FP16 operation-boundary emulator uses FP32 reductions; not HTP bit-exact. Scalar, vector and full-model NLL objectives kept separate.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    quality(p.parse_args().root.resolve())
