# SPDX-License-Identifier: BSD-3-Clause
"""Offline two-region K8 fit to N(0, 1/128), without fitting LM centroids.

Optimize four shared constants using exact Gaussian scalar MSE integrals.
No real KV, attention loss, PPL or layer selection enters this objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr


def gaussian_mse(positive: np.ndarray) -> tuple[float, np.ndarray]:
    edges = np.r_[0.0, (positive[:-1] + positive[1:]) / 2, np.inf]
    phi = np.exp(-(edges**2) / 2) / np.sqrt(2 * np.pi)
    mass = ndtr(-edges[:-1]) - ndtr(-edges[1:])
    first = phi[:-1] - phi[1:]
    edge_phi = np.zeros_like(edges)
    np.multiply(edges, phi, out=edge_phi, where=np.isfinite(edges))
    second = mass + edge_phi[:-1] - edge_phi[1:]
    error = 2 * np.sum(second - 2 * positive * first + positive**2 * mass)
    return float(error), 4 * (positive * mass - first)


def optimize() -> dict:
    # p = initial offset, central spacing, positive inter-region gap,
    #     tail-minus-central spacing. Bounds guarantee strict ordering.
    k = np.arange(64, dtype=np.float64)
    design = np.r_[
        np.c_[np.ones(64), k, np.zeros(64), np.zeros(64)],
        np.c_[np.ones(64), 63 + k, np.ones(64), k],
    ]

    def objective(p: np.ndarray) -> tuple[float, np.ndarray]:
        mse, gradient = gaussian_mse(design @ p)
        return mse, design.T @ gradient

    trials = []
    for edge in (1.0, 1.5, 2.0):
        for tail in (4.0, 5.0, 6.0):
            central, outer = edge / 64, (tail - edge) / 64
            initial = [central / 2, central, (central + outer) / 2, outer - central]
            fit = minimize(
                objective,
                initial,
                jac=True,
                method="L-BFGS-B",
                bounds=[(1e-6, None)] * 4,
                options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 4000},
            )
            trials.append(
                {
                    "initial": initial,
                    "parameters": fit.x.tolist(),
                    "mse_unit": float(fit.fun),
                    "success": bool(fit.success),
                    "message": str(fit.message),
                    "iterations": int(fit.nit),
                }
            )
    best = min(trials, key=lambda x: x["mse_unit"])
    a, h, gap, extra = best["parameters"]
    ideal = np.array([a, h, a + 63 * h + gap, h + extra]) / np.sqrt(128)
    coefficients = ideal.astype(np.float16)
    # The deployed encoder computes one FP16 product followed by one FP16 add.
    positive = np.concatenate(
        [
            (k.astype(np.float16) * coefficients[i + 1] + coefficients[i]).astype(
                np.float16
            )
            for i in (0, 2)
        ]
    )
    centroids = np.r_[-positive[::-1], positive].astype("<f8")
    assert np.all(np.diff(centroids) > 0)
    return {
        "schema": "structured-k8-two-region-v1",
        "distribution": "N(0, 1/128)",
        "objective": "Gaussian scalar reconstruction MSE",
        "optimizer": "L-BFGS-B with analytic integrated MSE and envelope gradient",
        "initializations": "nine fixed (region edge, tail extent) pairs; no RNG",
        "global_optimum_claim": False,
        "trials": trials,
        "ideal_coefficients": ideal.tolist(),
        "coefficients": coefficients.astype(float).tolist(),
        "coefficient_hex": [float(v).hex() for v in coefficients],
        "region_boundary": float((float(positive[63]) + float(positive[64])) / 2),
        "centroids": centroids.tolist(),
        "centroid_sha256": hashlib.sha256(centroids.tobytes()).hexdigest(),
        "ideal_gaussian_mse": best["mse_unit"] / 128,
        "rounded_centroid_nearest_gaussian_mse": gaussian_mse(
            positive.astype(float) * np.sqrt(128)
        )[0]
        / 128,
        "notes": "Rounded-centroid nearest MSE is not arithmetic-index FP16 MSE; evaluate the deployed index rule and norm correction separately. Existing LM is frozen 100 iterations, not assumed converged.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = optimize()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(result, stream, indent=2)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
