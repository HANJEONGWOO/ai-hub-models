# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Cross-check the in-repo PolarQuant oracle against pinned turboquant_plus.

Runs the reference package at ``constants.REFERENCE_COMMIT`` next to
``qai_hub_models.models.templates.llm.turboquant`` and compares codebooks,
rotations, indices, reconstructions and packed bytes. Optionally writes the
golden fixture consumed by the unit tests, so CI never needs the reference repo.

Usage (``src`` must shadow the installed wheel):

    PYTHONPATH=src python scripts/llm/turboquant/verify_reference.py \
        --reference-repo ~/git/turboquant_plus \
        --report /tmp/claude/turboquant/verify_reference.json \
        --write-golden
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from qai_hub_models.models.templates.llm.turboquant.config import (
    KEY_SEED,
    VALUE_SEED,
    CodecKind,
    KVCodecSpec,
    Rotation,
)
from qai_hub_models.models.templates.llm.turboquant.constants import REFERENCE_COMMIT
from qai_hub_models.models.templates.llm.turboquant.packing import pack_indices
from qai_hub_models.models.templates.llm.turboquant.reference import (
    DenseQRRotation,
    FWHTRotation,
    PolarQuantReference,
    load_codebook,
)

BLOCK = 128
REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = (
    REPO_ROOT / "src/qai_hub_models/test/test_models/turboquant_golden_v1.json"
)


def extract_reference(repo: Path, workdir: Path) -> SimpleNamespace:
    """Unpack the pinned ``turboquant`` package and load its modules by name."""
    archive = workdir / "turboquant.tar"
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            f"--output={archive}",
            REFERENCE_COMMIT,
            "turboquant",
        ],
        check=True,
    )
    with tarfile.open(archive) as tar:
        tar.extractall(workdir, filter="data")
    sys.path.insert(0, str(workdir))
    names = ("codebook", "polar_quant", "rotation", "utils")
    return SimpleNamespace(
        **{name: importlib.import_module(f"turboquant.{name}") for name in names}
    )


def test_vectors(rng: np.random.Generator) -> np.ndarray:
    """Gaussian rows plus zero, tiny, huge, one-hot and constant edge cases."""
    gaussian = rng.standard_normal((2048, BLOCK)) * rng.uniform(0.01, 50, (2048, 1))
    edge = np.zeros((6, BLOCK))
    edge[1] = 1e-30
    edge[2] = 1e4
    edge[3, 5] = 3.0
    edge[4] = -1.0
    edge[5, ::2] = 0.25
    return np.concatenate([gaussian, edge])


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def run_checks(ref: SimpleNamespace, x: np.ndarray) -> dict[str, Any]:
    codebook, polar_quant, rotation, utils = (
        ref.codebook,
        ref.polar_quant,
        ref.rotation,
        ref.utils,
    )
    report: dict[str, Any] = {"reference_commit": REFERENCE_COMMIT, "checks": {}}
    checks = report["checks"]

    for bits in (3, 4):
        ref_c = codebook.optimal_centroids(bits, BLOCK)
        checks[f"codebook_{bits}bit_exact"] = bool(
            np.array_equal(ref_c, load_codebook(bits, BLOCK))
        )

    for seed in (KEY_SEED, VALUE_SEED):
        s1, s2, _ = rotation.random_rotation_fast(BLOCK, np.random.default_rng(seed))
        ours = FWHTRotation(seed, BLOCK)
        fwd = rotation.apply_fast_rotation_batch(x, s1, s2, BLOCK)
        inv = np.stack(
            [rotation.apply_fast_rotation_transpose(v, s1, s2, BLOCK) for v in x[:64]]
        )
        checks[f"fwht_seed{seed}_forward_max_abs"] = max_abs(ours.forward(x), fwd)
        checks[f"fwht_seed{seed}_inverse_max_abs"] = max_abs(ours.inverse(x[:64]), inv)
        dense = rotation.random_rotation_dense(BLOCK, np.random.default_rng(seed))
        checks[f"dense_qr_seed{seed}_max_abs"] = max_abs(
            DenseQRRotation(seed, BLOCK).q, dense
        )

    for bits in (3, 4):
        for seed in (KEY_SEED, VALUE_SEED):
            spec = KVCodecSpec(CodecKind.POLAR, bits=bits, seed=seed)
            ref_pq = polar_quant.PolarQuant(
                BLOCK, bits, seed=seed, norm_correction=True
            )
            ref_idx, ref_norms = ref_pq.quantize(x)
            ref_xhat = ref_pq.dequantize(ref_idx, ref_norms)
            ours = PolarQuantReference(spec, BLOCK, Rotation.DENSE_QR)
            idx, norms = ours.encode(x)
            tag = f"dense_qr_{bits}bit_seed{seed}"
            checks[f"{tag}_index_mismatches"] = int(np.sum(idx != ref_idx))
            checks[f"{tag}_norm_max_abs"] = max_abs(norms[:, 0], ref_norms)
            checks[f"{tag}_decode_max_rel"] = max_abs(
                ours.decode(idx, norms) / np.maximum(norms, 1e-300),
                ref_xhat / np.maximum(norms, 1e-300),
            )

            s1, s2, _ = rotation.random_rotation_fast(
                BLOCK, np.random.default_rng(seed)
            )
            safe = np.where(ref_norms > 0, ref_norms, 1.0)[:, None]
            y = rotation.apply_fast_rotation_batch(x / safe, s1, s2, BLOCK)
            fwht_ref_idx = codebook.nearest_centroid_indices(y, ref_pq.centroids)
            fwht = PolarQuantReference(spec, BLOCK, Rotation.FWHT)
            fidx, fnorms = fwht.encode(x)
            ftag = f"fwht_{bits}bit_seed{seed}"
            checks[f"{ftag}_index_mismatches"] = int(np.sum(fidx != fwht_ref_idx))
            y_hat = ref_pq.centroids[fwht_ref_idx]
            y_hat = y_hat / np.linalg.norm(y_hat, axis=1, keepdims=True)
            fwht_ref_xhat = (
                np.stack(
                    [
                        rotation.apply_fast_rotation_transpose(v, s1, s2, BLOCK)
                        for v in y_hat
                    ]
                )
                * ref_norms[:, None]
            )
            checks[f"{ftag}_decode_max_rel"] = max_abs(
                fwht.decode(fidx, fnorms) / np.maximum(fnorms, 1e-300),
                fwht_ref_xhat / np.maximum(fnorms, 1e-300),
            )
            checks[f"{ftag}_pack_equal_reference"] = bool(
                np.array_equal(
                    pack_indices(fidx, bits).reshape(-1),
                    utils.pack_indices(fwht_ref_idx, bits),
                )
            )

    failures = []
    for name, value in checks.items():
        if isinstance(value, bool):
            ok = value
        elif name.endswith("_index_mismatches"):
            ok = value == 0
        else:
            ok = value <= 1e-10
        if not ok:
            failures.append(name)
    report["failures"] = failures
    return report


def write_golden(ref: SimpleNamespace, path: Path) -> None:
    codebook, rotation, utils = ref.codebook, ref.rotation, ref.utils
    rng = np.random.default_rng(20260915)
    x = np.concatenate(
        [
            rng.standard_normal((6, BLOCK)) * rng.uniform(0.1, 300, (6, 1)),
            np.zeros((1, BLOCK)),
            np.full((1, BLOCK), 1e-3),
        ]
    )
    cases: dict[str, Any] = {"inputs": x.tolist(), "cases": {}}
    for bits, seed in ((4, KEY_SEED), (4, VALUE_SEED), (3, VALUE_SEED)):
        s1, s2, _ = rotation.random_rotation_fast(BLOCK, np.random.default_rng(seed))
        centroids = codebook.optimal_centroids(bits, BLOCK)
        norms = np.linalg.norm(x, axis=1)
        safe = np.where(norms > 0, norms, 1.0)[:, None]
        y = rotation.apply_fast_rotation_batch(x / safe, s1, s2, BLOCK)
        idx = codebook.nearest_centroid_indices(y, centroids)
        y_hat = centroids[idx]
        y_hat = y_hat / np.linalg.norm(y_hat, axis=1, keepdims=True)
        x_hat = (
            np.stack(
                [
                    rotation.apply_fast_rotation_transpose(v, s1, s2, BLOCK)
                    for v in y_hat
                ]
            )
            * norms[:, None]
        )
        packed = np.stack([utils.pack_indices(row, bits) for row in idx])
        cases["cases"][f"{bits}bit_seed{seed}"] = {
            "bits": bits,
            "seed": seed,
            "indices": idx.tolist(),
            "norms": norms.tolist(),
            "packed_hex": [bytes(row).hex() for row in packed.astype(np.uint8)],
            "decoded": x_hat.tolist(),
        }
    cases["generator"] = "scripts/llm/turboquant/verify_reference.py"
    cases["reference_commit"] = REFERENCE_COMMIT
    path.write_text(json.dumps(cases, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--write-golden", action="store_true")
    args = parser.parse_args()

    Path("/tmp/claude").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir="/tmp/claude") as tmp:
        ref = extract_reference(args.reference_repo.expanduser(), Path(tmp))
        report = run_checks(ref, test_vectors(np.random.default_rng(7)))
        if args.write_golden:
            write_golden(ref, GOLDEN_PATH)
            report["golden"] = str(GOLDEN_PATH)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
