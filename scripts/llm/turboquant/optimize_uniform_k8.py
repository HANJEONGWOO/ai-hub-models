# SPDX-License-Identifier: BSD-3-Clause
"""One Gaussian-MSE-optimized symmetric 256-level mid-rise quantizer.

No real KV/NLL fitting. Optimize the single spacing in FP64, then freeze its
nearest FP16 representation. Offset is always spacing/2, never a second fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import ndtr


def integrated_mse(positive: np.ndarray, edges: np.ndarray) -> float:
    phi = np.exp(-0.5 * edges**2) / np.sqrt(2 * np.pi)
    mass = ndtr(-edges[:-1]) - ndtr(-edges[1:])
    first = phi[:-1] - phi[1:]
    edge_phi = np.zeros_like(edges)
    np.multiply(edges, phi, out=edge_phi, where=np.isfinite(edges))
    second = mass + edge_phi[:-1] - edge_phi[1:]
    return float(2 * np.sum(second - 2 * positive * first + positive**2 * mass))


def objective(step_sigma: float) -> float:
    c = (np.arange(128, dtype=np.float64) + 0.5) * step_sigma
    edges = np.r_[np.arange(128) * step_sigma, np.inf]
    return integrated_mse(c, edges) / 128


def optimize(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    history, trials = [], []
    # Fixed brackets, tolerances and diagnostic grid; none use model outcomes.
    for trial, bounds in enumerate(((0.005, 0.1), (0.015, 0.05), (0.02, 0.04))):

        def loss(x: float, trial: int = trial) -> float:
            value = objective(x)
            history.append(
                {
                    "trial": trial,
                    "evaluation": len(history),
                    "step_sigma": x,
                    "mse": value,
                }
            )
            return value

        fit = minimize_scalar(
            loss,
            method="bounded",
            bounds=bounds,
            options={"xatol": 1e-13, "maxiter": 1000},
        )
        trials.append(
            {
                "bounds_sigma": bounds,
                "step_sigma": float(fit.x),
                "mse": float(fit.fun),
                "converged": bool(fit.success),
                "evaluations": int(fit.nfev),
                "message": str(fit.message),
            }
        )
    assert all(t["converged"] for t in trials)
    best = min(trials, key=lambda t: t["mse"])
    ideal = best["step_sigma"] / np.sqrt(128)
    delta = np.float16(ideal)
    offset = np.float16(delta / np.float16(2))
    local = np.arange(128, dtype=np.float16)
    pos = (np.float16(local * delta) + offset).astype(np.float16)
    table = np.r_[-pos[::-1], pos].astype("<f8")
    assert np.all(np.diff(table) > 0) and np.array_equal(table, -table[::-1])
    sigma_step = float(delta) * np.sqrt(128)
    rule_mse = (
        integrated_mse(
            pos.astype(float) * np.sqrt(128), np.r_[np.arange(128) * sigma_step, np.inf]
        )
        / 128
    )
    nearest = (
        integrated_mse(
            pos.astype(float) * np.sqrt(128),
            np.r_[
                0,
                (pos[:-1].astype(float) + pos[1:].astype(float)) * np.sqrt(128) / 2,
                np.inf,
            ],
        )
        / 128
    )
    edges = np.r_[np.arange(128) * best["step_sigma"], np.inf]
    positive = (np.arange(128) + 0.5) * best["step_sigma"]
    numerical = (
        sum(
            quad(
                lambda x, c=c: (x - c) ** 2 * np.exp(-x * x / 2) / np.sqrt(2 * np.pi),
                lo,
                hi,
                epsabs=1e-14,
                epsrel=1e-10,
            )[0]
            for c, lo, hi in zip(positive, edges[:-1], edges[1:], strict=True)
        )
        * 2
        / 128
    )
    assert abs(numerical / best["mse"] - 1) < 1e-8
    grid = np.geomspace(0.002, 0.2, 2049)
    assert best["mse"] <= min(objective(x) for x in grid)
    result = {
        "schema": "uniform-k8-gaussian-v1",
        "distribution": "N(0,1/128)",
        "levels": 256,
        "optimizer": "FP64 bounded scalar minimization, three fixed brackets; nearest FP16 step; no real-data fitting",
        "global_optimum_claim": False,
        "trials": trials,
        "ideal_delta": ideal,
        "delta": float(delta),
        "offset": float(offset),
        "coefficient_hex": [float(offset).hex(), float(delta).hex()],
        "centroid_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
        "centroid_fp16_sha256": hashlib.sha256(
            table.astype("<f2").tobytes()
        ).hexdigest(),
        "ideal_gaussian_mse": best["mse"],
        "independent_integral_mse": numerical,
        "rounded_centroid_rule_gaussian_mse": rule_mse,
        "rounded_centroid_nearest_gaussian_mse": nearest,
        "index": "j=clip(floor(abs(y)/delta),0,127); y>=0?128+j:127-j",
        "rounding": "FP16(FP16(j*delta)+offset), sign, FP16 effective-scale product. Index FP16 reciprocal/multiply and input rounding evaluated separately.",
    }
    with (out / "optimization.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    with (out / "convergence.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    for name, values in (
        ("centroids_fp16", table.astype("<f2")),
        ("centroids_fp64", table),
        ("coefficients_fp16", np.array([offset, delta], dtype="<f2")),
    ):
        with (out / (name + ".npy")).open("xb") as f:
            np.save(f, values)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    print(json.dumps(optimize(p.parse_args().out), indent=2))
