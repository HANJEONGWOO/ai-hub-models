# SPDX-License-Identifier: BSD-3-Clause
"""Immutable host-kernel test log and final experiment audit; no device rerun."""

from __future__ import annotations

import argparse
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from benchmark_asymmetric_once import digest, read, save_new
from benchmark_uniform_k8 import GROUPS, validate

from qai_hub_models.models.templates.llm.turboquant.config import PROFILES, get_profile
from qai_hub_models.models.templates.llm.turboquant.uniform_k8 import (
    codebook,
    coefficients,
)


def preservation(root: Path) -> None:
    protocol = read(root / "protocol.json")
    source_path = "src/qai_hub_models/models/templates/llm/turboquant/config.py"
    source = subprocess.check_output(
        ["git", "show", protocol["base_commit"] + ":" + source_path], text=True
    )
    module = types.ModuleType("_uniform_k8_base_config_audit")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, source_path, "exec"), module.__dict__)
        old = {name: c.config_hash() for name, c in module.PROFILES.items()}
    finally:
        del sys.modules[module.__name__]
    changed = [name for name, sha in old.items() if PROFILES[name].config_hash() != sha]
    assert not changed, changed
    save_new(
        root / "default_preservation.json",
        {
            "passed": True,
            "base_commit": protocol["base_commit"],
            "previous_profiles": old,
            "added_profiles": sorted(set(PROFILES) - set(old)),
            "automatic_commit": False,
        },
    )
    print("Preserved all", len(old), "pre-existing profile hashes")


def final(root: Path) -> None:
    protocol = validate(root)
    for name, sha in read(root / "source_identity.json")["files_sha256"].items():
        assert digest(Path(name)) == sha, name
    for name, sha in read(root / "offline_quality.json")["capture_sha256"].items():
        assert digest(Path(name)) == sha, name
    optimized = read(root / "codebook/optimization.json")
    assert all(t["converged"] for t in optimized["trials"])
    assert all(
        float(np.float16(t["step_sigma"] / np.sqrt(128))) == optimized["delta"]
        for t in optimized["trials"]
    )
    assert (
        abs(optimized["independent_integral_mse"] / optimized["ideal_gaussian_mse"] - 1)
        < 1e-8
    )
    assert np.array_equal(
        np.load(root / "codebook/centroids_fp16.npy"), codebook().astype(np.float16)
    )
    assert np.array_equal(
        np.load(root / "codebook/coefficients_fp16.npy"),
        coefficients().astype(np.float16),
    )
    assert read(root / "reports/graph_audit.json")["passed"]
    assert read(root / "default_preservation.json")["passed"]
    comparison = read(root / "comparison.json")
    for group in GROUPS:
        assert read(root / "reports" / f"audit_{group}.json")["passed"]
        assert len(list((root / "reports").glob(f"{group}_perf_r*.json"))) == 3
        for label in [f"perf_r{i}" for i in range(3)] + [
            f"score_{s}_{i}" for s in ("legacy", "heldout") for i in range(4)
        ]:
            report = read(root / "reports" / f"{group}_{label}.json")
            assert report["config_hash"] == protocol["profiles"][group]["config_hash"]
            assert (
                report["kv_store_bytes"] == 30277632
                and report["quantize_current_kv"]
                and not report.get("op_profiles")
            )
            assert (
                report["assets"]["tokens_sha256"]
                == protocol["assets_manifest"]["sha256"][
                    report["assets"]["tokens_file"]
                ]
            )
            if label.startswith("score"):
                assert report["scored_tokens"] == 1023
        for name, identity in comparison["groups"][group]["binaries"].items():
            assert digest(root / group / name) == identity["sha256"]
        reset = read(root / "reports" / f"{group}_reset.json")
        assert reset["sessions"][0]["generated"] == reset["sessions"][1]["generated"]
        encoder = read(root / f"encoder_{group}/p2_report.json")
        assert all(encoder["graphs"][f"tq_encode_key_t{t}"]["passed"] for t in (1, 128))
        assert all(
            not v["dlc_ops_missing_from_accelerator_profile"]
            for v in encoder["graphs"].values()
        )
        assert all(
            c["unexplained_index_mismatches"] == 0
            for v in encoder["graphs"].values()
            for c in v["cases"].values()
        )
    costs = read(root / "codec_costs.json")
    assert len(costs["decoder"]) == 8 and all(
        d["bit_exact_fraction"] == 1 for d in costs["decoder"]
    )
    hvx = (root / "tests/hvx.log").read_text()
    assert (
        "Uniform K8 arithmetic/LUT bit-exact" in hvx
        and "Structured K8 arithmetic/LUT bit-exact" in hvx
    )
    suites = list(
        ET.parse(root / "tests/regression_release.xml").getroot().iter("testsuite")
    )
    assert all(
        int(s.get("errors", "0")) == int(s.get("failures", "0")) == 0 for s in suites
    )
    assert (
        get_profile("k4_v4_scaled").config_hash()
        == "d7ac74594b85ffd81086018773add722de8e6b9f1eee7339b929f8510374be6e"
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    assert head == protocol["base_commit"] and branch == protocol["branch"]
    previous = Path(
        "/mnt/d/ai-hub-models/binaries/turboquant/lloyd_max_k8_convergence_20261020/reports"
    )
    structured_unchanged = [
        read(root / "reports" / f"structured_score_legacy_{i}.json")["nll_sum"]
        == read(previous / f"structured_score_legacy_{i}.json")["nll_sum"]
        for i in range(4)
    ]
    assert all(structured_unchanged), "Unchanged Structured quality regressed"
    files = (
        subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
        + subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"], text=True
        ).splitlines()
    )
    artifacts = [
        root / n
        for n in (
            "comparison.json",
            "comparison.csv",
            "document_nll.csv",
            "protocol.json",
            "data_manifest.json",
            "source_identity.json",
            "environment.json",
            "offline_quality.json",
            "codec_costs.json",
            "tests/regression_release.xml",
            "default_preservation.json",
            "tests/hvx.log",
        )
    ] + sorted((root / "reports").glob("*.json"))
    artifacts += [
        *sorted((root / "codebook").iterdir()),
        root / "native/manifest.json",
    ]
    for group in GROUPS:
        artifacts += [
            root / group / "convert_report.json",
            root / f"encoder_{group}/p2_report.json",
            root / f"decoder_{group}/correctness.json",
        ]
    save_new(
        root / "final_verification.json",
        {
            "branch": branch,
            "head": head,
            "automatic_commit": False,
            "regression_tests": sum(int(s.get("tests", "0")) for s in suites),
            "performance_sessions": 6,
            "scored_windows": 16,
            "new_k_encoder_passed": True,
            "decoder_bit_exact_cases": 8,
            "structured_legacy_nll_unchanged": structured_unchanged,
            "all_encoder_graphs_passed": False,
            "known_limit": "Common standalone V4 AR128 scale tolerance fails; not the V2 full-model path. No tolerance relaxed.",
            "changed_files_sha256": {p: digest(Path(p)) for p in sorted(files)},
            "artifact_sha256": {str(p): digest(p) for p in artifacts},
        },
    )
    print(
        "Final audit passed: 6 performance sessions, 16 scoring windows, 8 exact decoder probes; defaults/previous quality preserved"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("hvx", "preservation", "final"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    if a.stage == "hvx":
        with (root / "tests/hvx.log").open("x") as stream:
            subprocess.run(
                [
                    str(root / "native/test_hvx"),
                    str(root / "native/structured_reference.fp16"),
                    str(root / "native/uniform_reference.fp16"),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print("Host HVX regression passed; saved tests/hvx.log")
    elif a.stage == "preservation":
        preservation(root)
    else:
        final(root)
