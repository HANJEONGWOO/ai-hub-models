# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Gaussian convergence curves and held-fixed captured KV/QK reconstruction errors."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from benchmark_asymmetric_once import ARTIFACTS, digest, read, save_new
from benchmark_lm8_convergence import GROUPS
from evaluate_structured_k8_quality import errors, structured_codec
from optimize_lloyd_max_k8 import mse
from scipy.integrate import quad
from select_mixed_layers import codec, matrix, rounded

from qai_hub_models.models.templates.llm.turboquant.lloyd_max_k8 import codebook
from qai_hub_models.models.templates.llm.turboquant.reference import load_codebook
from qai_hub_models.models.templates.llm.turboquant.structured_k8 import (
    codebook as structured_codebook,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def encoder_lut(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.float32)
    values = []
    for i in range(256):
        mag = abs(i - 127.5) - 0.5
        offset = 2 * (int(mag) // 2)
        start = 128 + offset
        slope = c[start + 1] - c[start]
        base = c[start] - offset * slope
        value = np.float16(np.float16(mag * np.float16(slope)) + np.float16(base))
        values.append(float(value) if i >= 128 else -float(value))
    return np.array(values, np.float32)


def converged_codec(x: torch.Tensor) -> torch.Tensor:
    c = codebook()
    lut = rounded(torch.tensor(c.astype(np.float32)))
    enc = torch.tensor(encoder_lut(c))
    boundaries = rounded(torch.tensor(((c[:-1] + c[1:]) / 2).astype(np.float32)))
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
    indices = torch.bucketize(y.contiguous(), boundaries, right=False)
    selected = enc[indices]
    correction = rounded(
        torch.sqrt(rounded(rounded(selected * selected).sum(-1, keepdim=True)))
    )
    scale = rounded(norm / torch.where(correction > 0, correction, 1.0))
    return rounded(lut[indices] * scale)


def quality(root: Path) -> None:
    torch.set_num_threads(4)
    captures = ARTIFACTS / "k_rotation_quality_20261006/samples"
    position = torch.arange(31, 1024, 32)
    mask = (
        (torch.arange(1024)[None, :] <= position[:, None])
        .unsqueeze(0)
        .expand(16, -1, -1)
    )
    rows, frozen = [], {}
    for document in range(2):
        for layer in range(28):
            path = captures / f"validation_{document}_layer{layer:02d}.npz"
            frozen[str(path)] = digest(path)
            with np.load(path) as data:
                key = rounded(torch.tensor(data["k"].astype(np.float32)))
                query = rounded(torch.tensor(data["q"][:, position].astype(np.float32)))
            qrot, krot = rounded(query @ matrix(42).T), rounded(key @ matrix(42).T)
            ref = rounded(query @ key.repeat_interleave(2, 0).transpose(-1, -2))[mask]
            rot_ref = rounded(qrot @ krot.repeat_interleave(2, 0).transpose(-1, -2))[
                mask
            ]
            for group in GROUPS:
                restored = (
                    (
                        converged_codec(key)
                        if group == "lm8_converged"
                        else structured_codec(key)
                    )
                    if layer % 2 and group != "lm8_100"
                    else codec(key, 8 if layer % 2 else 4, 42)
                )
                scores = rounded(
                    qrot @ restored.repeat_interleave(2, 0).transpose(-1, -2)
                )[mask]
                rows.append(
                    {
                        "group": group,
                        "document": document,
                        "layer": layer,
                        "bits": 8 if layer % 2 else 4,
                        "k_reconstruction": errors(key, rounded(restored @ matrix(42))),
                        "k_rotated": errors(krot, restored),
                        "qk_score": errors(ref, scores),
                        "qk_score_rotated_reference": errors(rot_ref, scores),
                    }
                )
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
                    **v,
                    "mse": v["sse"] / v["count"],
                    "relative_mse": v["sse"] / v["reference_energy"],
                }
                for metric, v in totals.items()
            }
    tables = {
        "lm8_100": load_codebook(8, 128),
        "lm8_converged": codebook(),
        "structured": structured_codebook(),
    }
    gaussian = {}
    synthetic = rounded(
        torch.tensor(
            np.random.default_rng(20261020).normal(size=(16384, 128)).astype(np.float32)
        )
    )
    for group, c in tables.items():
        reconstructed = (
            codec(synthetic, 8, 42)
            if group == "lm8_100"
            else converged_codec(synthetic)
            if group == "lm8_converged"
            else structured_codec(synthetic)
        )
        enc = c.astype(np.float32) if group == "structured" else encoder_lut(c)
        direct = c.astype(np.float16).astype(np.float32)
        stats = errors(synthetic, rounded(reconstructed @ matrix(42)))
        gaussian[group] = {
            "ideal_nearest_scalar_mse": mse(c[128:] * np.sqrt(128)) / 128,
            "fp16_nearest_scalar_mse": mse(
                c[128:].astype(np.float16).astype(float) * np.sqrt(128)
            )
            / 128,
            "encoder_interpolation_vs_native_lut_different_entries": int(
                np.count_nonzero(enc != direct)
            ),
            "encoder_interpolation_vs_native_lut_max_abs_error": float(
                np.max(np.abs(enc - direct))
            ),
            "synthetic_final_kv": {**stats, "mse": stats["sse"] / stats["count"]},
            "note": "Nearest scalar MSE is not Structured arithmetic-index MSE. Synthetic final KV includes actual codec rule, normalization, Dense, encoder interpolation/norm, effective FP16 scale, Native FP16 multiply and inverse rotation.",
        }
        for suffix, array in (
            ("lut_fp16", c.astype(np.float16)),
            ("encoder_interpolation_fp16", enc.astype(np.float16)),
        ):
            with (root / "codebook" / f"{group}_{suffix}.npy").open("xb") as stream:
                np.save(stream, array)
    save_new(
        root / "offline_quality.json",
        {
            "aggregate": aggregate,
            "per_layer_document": rows,
            "gaussian": gaussian,
            "capture_sha256": frozen,
            "reference": "Same W4A16 FP16-KV captures, not full FP16 model. Fixed 32 query positions/document; Q includes attention scale; causal pre-softmax scores; GQA=2.",
            "limits": "FP16 operation-boundary emulator with FP32 reductions, not HTP bit-exact. Historical captures for diagnostics only; no coefficient or PPL selection.",
        },
    )


def curves(root: Path) -> None:
    result = read(root / "codebook/optimization.json")
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for trial in result["trials"]:
        history = pd.read_csv(root / "codebook" / (trial["name"] + ".csv"))
        sample = history.iloc[
            np.unique(
                np.r_[
                    np.arange(min(200, len(history))),
                    np.linspace(0, len(history) - 1, 1600).astype(int),
                ]
            )
        ]
        axes[0].plot(sample["iteration"], sample["mse"], label=trial["name"])
        axes[1].plot(
            sample["iteration"].iloc[1:], sample["max_centroid_delta_sigma"].iloc[1:]
        )
    axes[0].axhline(
        result["legacy_gaussian_mse"],
        color="gray",
        linestyle="--",
        label="frozen LM8-100",
    )
    axes[1].axhline(result["centroid_tolerance_sigma"], color="gray", linestyle="--")
    axes[0].set_ylabel("Gaussian scalar MSE, N(0,1/128)")
    axes[1].set_ylabel("Max centroid movement (sigma)")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlim(0, max(t["iterations"] for t in result["trials"]))
        ax.set_yscale("log")
        ax.set_xlabel("Additional Lloyd iterations")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / ("lloyd_convergence_nonnegative." + suffix), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if (args.root / "offline_quality.json").exists():
        raise FileExistsError(
            "Use a fresh directory; completed quality results are immutable"
        )
    c = codebook()[128:] * np.sqrt(128)
    b = np.r_[0, (c[:-1] + c[1:]) / 2, np.inf]
    numerical = (
        2
        * sum(
            quad(
                lambda x, value=value: (x - value) ** 2
                * np.exp(-x * x / 2)
                / np.sqrt(2 * np.pi),
                lo,
                hi,
                epsabs=1e-14,
            )[0]
            for value, lo, hi in zip(c, b[:-1], b[1:], strict=True)
        )
        / 128
    )
    from optimize_lloyd_max_k8 import moments

    mass, first, _ = moments(c)
    save_new(
        args.root / "codebook/independent_integral_check.json",
        {
            "quadrature_mse": numerical,
            "analytic_mse": mse(c) / 128,
            "relative_difference": abs(numerical - mse(c) / 128) / numerical,
            "fixed_point_max_delta_sigma": float(np.max(np.abs(first / mass - c))),
            "passed": abs(numerical - mse(c) / 128) / numerical < 1e-8,
        },
    )
    curves(args.root)
    quality(args.root)


if __name__ == "__main__":
    main()
