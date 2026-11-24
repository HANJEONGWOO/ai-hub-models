# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""FP64 Gaussian Lloyd iteration with immutable multi-start convergence logs.

Optimize scalar MSE only, without real KV, NLL, or Structured fitting. Work in
standard-normal coordinates to keep the stopping criterion scale independent;
export to N(0, 1/128). Symmetry is enforced, not inferred from CDF subtraction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

from qai_hub_models.models.templates.llm.turboquant.reference import load_codebook

MAX_ITERATIONS = 200_000
CENTROID_TOL = 1e-10  # max movement in standard deviations
MSE_REL_TOL = 1e-11
PATIENCE = 10


def moments(positive: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.r_[0.0, (positive[:-1] + positive[1:]) / 2, np.inf]
    phi = np.exp(-0.5 * edges**2) / np.sqrt(2 * np.pi)
    # Survival differences avoid upper-tail 1-CDF cancellation.
    mass = ndtr(-edges[:-1]) - ndtr(-edges[1:])
    first = phi[:-1] - phi[1:]
    edge_phi = np.zeros_like(edges)
    np.multiply(edges, phi, out=edge_phi, where=np.isfinite(edges))
    second = mass + edge_phi[:-1] - edge_phi[1:]
    if np.any(mass <= 0):
        raise ValueError("Empty/non-finite Gaussian cell")
    return mass, first, second


def mse(positive: np.ndarray) -> float:
    mass, first, second = moments(positive)
    return float(2 * np.sum(second - 2 * positive * first + positive**2 * mass))


def starts() -> dict[str, np.ndarray]:
    p = (np.arange(128, dtype=np.float64) + 0.5) / 256 + 0.5
    return {
        "legacy_100": load_codebook(8, 128)[128:] * np.sqrt(128),
        "gaussian_quantiles": ndtri(p),
        "uniform_4sigma": (np.arange(128, dtype=np.float64) + 0.5) * 4 / 128,
        "high_rate_companding": np.sqrt(3) * ndtri(p),
    }


def iterate(
    initial: np.ndarray, maximum: int = MAX_ITERATIONS
) -> tuple[np.ndarray, list[dict], bool]:
    c = np.array(initial, dtype=np.float64, copy=True)
    history = [
        {
            "iteration": 0,
            "mse": mse(c) / 128,
            "max_centroid_delta_sigma": 0.0,
            "relative_mse_change": 0.0,
            "stable_steps": 0,
        }
    ]
    stable = 0
    for i in range(1, maximum + 1):
        mass, first, _ = moments(c)
        updated = first / mass
        movement = float(np.max(np.abs(updated - c)))
        error = mse(updated) / 128
        previous = history[-1]["mse"]
        relative = abs(error - previous) / previous
        if error > previous + 2e-16:
            raise ArithmeticError(
                "Lloyd objective increased beyond FP64 integration tolerance"
            )
        stable = (
            stable + 1 if movement <= CENTROID_TOL and relative <= MSE_REL_TOL else 0
        )
        history.append(
            {
                "iteration": i,
                "mse": error,
                "max_centroid_delta_sigma": movement,
                "relative_mse_change": relative,
                "stable_steps": stable,
            }
        )
        c = updated
        if stable >= PATIENCE:
            return c, history, True
    return c, history, False


def array_hash(array: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(array.astype(dtype).tobytes()).hexdigest()


def optimize(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    trials = []
    for name, initial in starts().items():
        c, history, converged = iterate(initial)
        with (out / f"{name}.csv").open("x") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        trials.append(
            {
                "name": name,
                "converged": converged,
                "iterations": len(history) - 1,
                "initial_mse": history[0]["mse"],
                "final_mse": history[-1]["mse"],
                "final_delta_sigma": history[-1]["max_centroid_delta_sigma"],
                "positive_centroids": (c / np.sqrt(128)).tolist(),
                "history_sha256": hashlib.sha256(
                    (out / f"{name}.csv").read_bytes()
                ).hexdigest(),
            }
        )
        print(
            name,
            trials[-1]["iterations"],
            converged,
            trials[-1]["final_mse"],
            flush=True,
        )
    eligible = [x for x in trials if x["converged"]]
    if not eligible:
        raise RuntimeError("No converged trial; refuse to publish converged profile")
    best = min(eligible, key=lambda x: (x["final_mse"], x["name"]))
    positive = np.array(best["positive_centroids"], dtype=np.float64)
    codebook = np.r_[-positive[::-1], positive]
    boundaries = (codebook[:-1] + codebook[1:]) / 2
    result = {
        "schema": "lloyd-max-k8-converged-v1",
        "distribution": "N(0, 1/128)",
        "precision": "float64",
        "objective": "exact integrated scalar Gaussian MSE",
        "max_iterations": MAX_ITERATIONS,
        "centroid_tolerance_sigma": CENTROID_TOL,
        "relative_mse_tolerance": MSE_REL_TOL,
        "patience": PATIENCE,
        "initializations": "four deterministic starts; no random sampling or data fitting",
        "global_optimum_claim": False,
        "trials": trials,
        "selected": best["name"],
        "centroid_hex": [float(x).hex() for x in codebook],
        "centroid_f64_sha256": array_hash(codebook, "<f8"),
        "centroid_f16_sha256": array_hash(codebook, "<f2"),
        "boundary_f32_sha256": array_hash(boundaries, "<f4"),
        "boundary_f16_sha256": array_hash(boundaries, "<f2"),
        "legacy_gaussian_mse": mse(load_codebook(8, 128)[128:] * np.sqrt(128)) / 128,
        "converged_gaussian_mse": best["final_mse"],
        "fp16_centroid_nearest_gaussian_mse": mse(
            codebook[128:].astype(np.float16).astype(float) * np.sqrt(128)
        )
        / 128,
        "rounding": "Preserve existing encoder FP32 constants/HTP FP16 interpolation, FP16 effective scale and Native FP16 LUT product; not a claim of exact encoder centroid interpolation.",
    }
    with (out / "optimization.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    for name, array in (
        ("centroids_fp64", codebook),
        ("centroids_fp16", codebook.astype(np.float16)),
        ("boundaries_fp32", boundaries.astype(np.float32)),
        ("boundaries_fp16", boundaries.astype(np.float16)),
    ):
        with (out / f"{name}.npy").open("xb") as stream:
            np.save(stream, array)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    optimize(args.out)


if __name__ == "__main__":
    main()
