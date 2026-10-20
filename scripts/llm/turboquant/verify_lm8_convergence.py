# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Final artifact audit; does not execute the device or repeat any measurement."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_asymmetric_once import digest, read, save_new
from benchmark_lm8_convergence import GROUPS, validate

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.lloyd_max_k8 import (
    codebook,
    metadata,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    protocol = validate(root)
    for name, sha in read(root / "source_identity.json")["files_sha256"].items():
        assert digest(Path(name)) == sha, name
    optimized = read(root / "codebook/optimization.json")
    checks = {}
    fp16_hashes = set()
    for trial in optimized["trials"]:
        path = root / "codebook" / (trial["name"] + ".csv")
        assert digest(path) == trial["history_sha256"]
        frame = pd.read_csv(path)
        assert len(frame) == trial["iterations"] + 1
        assert np.array_equal(frame["iteration"], np.arange(len(frame)))
        assert np.isfinite(frame.to_numpy()).all()
        assert np.all(np.diff(frame["mse"]) <= 2e-16)
        assert trial["converged"] and trial["iterations"] <= optimized["max_iterations"]
        assert frame.iloc[-1]["stable_steps"] >= optimized["patience"]
        assert np.all(
            frame["max_centroid_delta_sigma"].iloc[-10:]
            <= optimized["centroid_tolerance_sigma"]
        )
        assert np.all(
            frame["relative_mse_change"].iloc[-10:]
            <= optimized["relative_mse_tolerance"]
        )
        c = np.array(trial["positive_centroids"])
        fp16_hashes.add(
            hashlib.sha256(np.r_[-c[::-1], c].astype("<f2").tobytes()).hexdigest()
        )
    assert fp16_hashes == {metadata()["codebook_fp16_sha256"]}
    checks["all_four_trials_converged_same_fp16"] = True
    checks["independent_gaussian_integral"] = read(
        root / "codebook/independent_integral_check.json"
    )["passed"]
    c = np.load(root / "codebook/centroids_fp64.npy")
    assert np.array_equal(c, codebook())
    assert np.array_equal(
        np.load(root / "codebook/centroids_fp16.npy"), c.astype(np.float16)
    )
    b = (c[:-1] + c[1:]) / 2
    assert np.array_equal(
        np.load(root / "codebook/boundaries_fp32.npy"), b.astype(np.float32)
    )
    assert np.array_equal(
        np.load(root / "codebook/boundaries_fp16.npy"),
        b.astype(np.float32).astype(np.float16),
    )
    checks["frozen_tables_boundaries_and_hashes"] = True
    compiled_graphs = 0
    for group in GROUPS:
        audit = read(root / "reports" / f"audit_{group}.json")
        assert audit["passed"]
        compiled_graphs += len(audit["graphs"])
        for split in ("legacy", "heldout"):
            for i in range(4):
                report = read(root / "reports" / f"{group}_score_{split}_{i}.json")
                assert (
                    report["scored_tokens"] == 1023
                    and report["kv_store_bytes"] == 30277632
                )
                assert (
                    report["config_hash"] == protocol["profiles"][group]["config_hash"]
                )
                assert (
                    report["assets"]["tokens_sha256"]
                    == protocol["assets_manifest"]["sha256"][
                        report["assets"]["tokens_file"]
                    ]
                )
        assert len(list((root / "reports").glob(f"{group}_perf_*.json"))) == 1
        reset = read(root / "reports" / f"{group}_reset.json")
        assert reset["sessions"][0]["generated"] == reset["sessions"][1]["generated"]
    assert read(root / "reports/codebook_only_audit.json")["passed"]
    checks["source_codebook_only_and_compiled_contracts"] = True
    costs = read(root / "codec_costs.json")
    assert len(costs["decoder"]) == 16
    assert all(row["bit_exact_fraction"] == 1 for row in costs["decoder"])
    assert costs["structured_lut_arithmetic_same_output"]
    checks["sixteen_decoder_probes_bit_exact"] = True
    numerical = costs["encoder_numerics"]["lm8_converged"]["graphs"]
    assert all(numerical[f"tq_encode_key_t{t}"]["passed"] for t in (1, 128))
    checks["new_k_encoder_existing_tolerances"] = True
    tests = ET.parse(root / "tests/regression_final.xml").getroot()
    suites = list(tests.iter("testsuite"))
    assert all(
        int(s.get("failures", "0")) == int(s.get("errors", "0")) == 0 for s in suites
    )
    test_count = sum(int(s.get("tests", "0")) for s in suites)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    assert head == protocol["base_commit"] and branch == protocol["branch"]
    assert (
        get_profile("k4_v4_scaled").config_hash()
        == "d7ac74594b85ffd81086018773add722de8e6b9f1eee7339b929f8510374be6e"
    )
    changed = subprocess.check_output(
        ["git", "diff", "--name-only"], text=True
    ).splitlines()
    changed += subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], text=True
    ).splitlines()
    files = [
        root / n
        for n in (
            "protocol.json",
            "source_identity.json",
            "data_manifest.json",
            "environment.json",
            "comparison.json",
            "comparison.csv",
            "document_nll.csv",
            "offline_quality.json",
            "codec_costs.json",
            "tests/regression_final.xml",
        )
    ]
    files += sorted((root / "reports").glob("*.json"))
    assert all(checks.values()), checks
    save_new(
        root / "final_verification.json",
        {
            "branch": branch,
            "head": head,
            "automatic_commit": False,
            "checks": checks,
            "regression_tests": test_count,
            "compiled_graphs": compiled_graphs,
            "performance_sessions": 3,
            "scored_windows": 24,
            "all_encoder_graphs_passed": False,
            "encoder_interpolation_bit_exact_to_lut": False,
            "limits": "Existing LM100 K AR128 and common standalone V4 AR128 scale gates fail. LM affine-pair FP16 interpolation differs from direct LUT; not silently fixed in constants-only experiment. Single performance session, no significance claim.",
            "changed_files_sha256": {p: digest(Path(p)) for p in sorted(changed)},
            "artifact_sha256": {str(p): digest(p) for p in files},
        },
    )
    print(
        "Verified",
        compiled_graphs,
        "compiled graphs,",
        test_count,
        "tests, 3 performance sessions, 24 scoring windows",
    )


if __name__ == "__main__":
    main()
