# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Final immutable-evidence audit; does not rerun search or device inference."""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from benchmark_allocation_v2 import config, validate
from benchmark_asymmetric_once import digest, read, save_new
from prepare_allocation_v2 import LATEST

from qai_hub_models.models.templates.llm.turboquant.config import PROFILES


def final(root: Path) -> None:
    protocol = validate(root)
    for name in (
        "comparison.json",
        "comparison.csv",
        "document_nll.csv",
        "seed_finals.csv",
        "search_history.csv",
        "offline_document_nll.csv",
        "figures/layer_masks.svg",
        "figures/search_history.svg",
        "figures/local_swap_delta.svg",
    ):
        assert (root / name).is_file(), name
    comparison = read(root / "comparison.json")
    selection = read(root / "selection.json")
    data = read(root / "data_manifest.json")
    physical = read(root / "physical_groups.json")
    certificate = read(root / "local_certificate.json")
    scorer = read(root / "offline/scorer_identity.json")
    lock = read(root / "offline/search_lock.json")
    script = Path(__file__)
    assert lock["algorithm_sha256"] == digest(
        script.with_name("search_allocation_v2.py")
    )
    assert lock["legacy_algorithm_sha256"] == digest(
        script.with_name("search_nll_allocation.py")
    )
    for p, sha in scorer["numerical_sources"].items():
        assert digest(Path(p)) == sha, p
    assert scorer["protocol_sha256"] == digest(root / "protocol.json")
    assert scorer["data_manifest_sha256"] == digest(root / "data_manifest.json")
    for name, sha in protocol["previous_profile_hashes"].items():
        assert PROFILES[name].config_hash() == sha, name
    assert len(PROFILES) == len(protocol["previous_profile_hashes"])
    preserved_sources = {}
    for pattern, field in (
        ("reuse_*.json", "source_files_sha256"),
        ("graph_sources_*.json", "files_sha256"),
    ):
        for path in (root / "reports").glob(pattern):
            for p, sha in read(path)[field].items():
                assert digest(Path(p)) == sha, p
                preserved_sources[p] = sha
    preflight = read(root / "replay_preflight_v2.json")
    assert preflight["passed"] and len(preflight["cases"]) == 24
    assert all(
        r["bit_exact_logits"] and r["max_abs_logit_error"] == 0
        for r in preflight["cases"]
    )
    assert {r["length"] for r in preflight["cases"]} == {256, 1024}
    all_tokens = set()
    for role, expected in (("search", 12), ("validation", 4), ("test", 8)):
        assert len(data["windows"][role]) == expected
        for row in data["windows"][role]:
            path = root / "assets" / row["file"]
            assert digest(path) == row["tokens_sha256"]
            assert path.stat().st_size == 1024 * 4
            assert row["tokens_sha256"] not in all_tokens
            all_tokens.add(row["tokens_sha256"])
    snapshot = comparison["score_cache_snapshot"]
    snapshot_path = root / "offline/scores_release.sqlite"
    assert snapshot["path"] == str(snapshot_path)
    assert digest(snapshot_path) == snapshot["sha256"]
    with sqlite3.connect(f"file:{snapshot_path}?mode=ro&immutable=1", uri=True) as db:
        identities = db.execute("SELECT DISTINCT identity FROM scores").fetchall()
        assert identities == [(digest(root / "offline/scorer_identity.json"),)]
        cached_inputs = {
            r[0] for r in db.execute("SELECT DISTINCT document FROM scores")
        }
        test_hashes = set()
        for row in data["windows"]["test"]:
            raw = (root / "assets" / row["file"]).read_bytes()
            for length in (256, 1024):
                test_hashes.add(hashlib.sha256(raw[: length * 4]).hexdigest())
        assert not cached_inputs & test_hashes
        assert (
            db.execute(
                "SELECT COUNT(*) FROM scores WHERE scored_tokens != length-1"
            ).fetchone()[0]
            == 0
        )
        assert not db.execute(
            "SELECT run FROM attempts WHERE finished IS NULL"
        ).fetchall()
        score_count = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        assert {
            table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("scores", "requests", "attempts")
        } == snapshot["rows"]
    assert len(selection["selected_layers"]) == 14 and not selection["test_used"]
    assert (
        read(root / "plans/final_selected.json")["config_hash"]
        == selection["selected_config_hash"]
    )
    assert certificate["selection_sha256"] == digest(root / "selection.json")
    assert certificate["layers"] == selection["selected_layers"]
    assert len(certificate["swaps"]) == 196
    assert not certificate["changes_final_selection"]
    assert len({tuple(r["layers"]) for r in certificate["swaps"]}) == 196
    assert read(root / "device_order.json")["selection_sha256"] == digest(
        root / "selection.json"
    )
    native = read(Path(protocol["native"]) / "manifest.json")
    # All physical groups use the same HTP op package, including old K6/V2 DLCs.
    native_sha = native["libraries"]["hexagon-v81"]["sha256"]
    device_reference = read(root / "reports/uniform_odd_reset.json")
    for group in physical["groups"]:
        c = config(root, group)
        converted = read(root / group / "convert_report.json")
        assert converted["config_hash"] == c.config_hash()
        assert converted["native_decoder"] == native
        assert converted["attention_tile"] == 256
        assert converted["rotated_attention"] and converted["quantize_current_kv"]
        assert converted["context_buckets"] == [1024]
        assert read(root / "reports" / f"audit_{group}.json")["passed"]
        if group != "k6_v2":
            assert converted["structured_k8_decoder"] == "arithmetic"
            assert c.structured_k8 and not c.uniform_k8 and not c.qjl
            bits = [
                (c.codec_for("key", i).bits, c.codec_for("value", i).bits)
                for i in range(28)
            ]
            assert bits.count((8, 2)) == bits.count((4, 2)) == 14
            assert read(root / "reports" / f"allocation_audit_{group}.json")["passed"]
        else:
            assert all(
                c.codec_for("key", i).bits == 6 and c.codec_for("value", i).bits == 2
                for i in range(28)
            )
        ready = read(root / "reports" / f"ready_{group}.json")
        for name, sha in ready["binaries_sha256"].items():
            assert digest(root / group / name) == sha
            assert comparison["groups"][group]["binaries"][name]["sha256"] == sha
        paths = [
            *(root / "reports").glob(f"{group}_perf_r*.json"),
            *(root / "reports").glob(f"{group}_test_*.json"),
            *(root / "reports").glob(f"{group}_legacy_*.json"),
        ]
        assert len(paths) == 3 + 8 + 4
        calibration_paths = [
            *(root / "reports").glob(f"{group}_search_*.json"),
            *(root / "reports").glob(f"{group}_validation_*.json"),
        ]
        assert len(calibration_paths) == (
            16 if group in selection["htp_mean_nll"] else 0
        )
        for path in paths + calibration_paths:
            r = read(path)
            assert r["config_hash"] == c.config_hash()
            assert r["kv_store_bytes"] == 30277632
            assert r["quantize_current_kv"] and not r.get("op_profiles")
            assert r["native_decoder"]["sha256"] == native_sha
            assert "htp " in r["backend_api"] and r["device"]["soc_model"] == "SM8850"
            assert r["device"] == device_reference["device"]
            assert r["backend_api"] == device_reference["backend_api"]
            assert (
                r["assets"]["tokens_sha256"]
                == protocol["assets_manifest"]["sha256"][r["assets"]["tokens_file"]]
            )
            # The runner also refuses the test stage without frozen selection.
            if path in paths:
                assert (
                    path.stat().st_mtime_ns
                    >= (root / "selection.json").stat().st_mtime_ns
                )
        reset = read(root / "reports" / f"{group}_reset.json")["sessions"]
        assert len(reset) == 2 and reset[0]["generated"] == reset[1]["generated"]
    # Reusing identical Odd graphs and the same package must retain legacy quality.
    unchanged = []
    prior_release = read(LATEST / "final_verification.json")["artifact_sha256"]
    for i in range(4):
        previous_path = LATEST / "reports" / f"structured_score_legacy_{i}.json"
        assert digest(previous_path) == prior_release[str(previous_path)]
        previous = read(previous_path)
        current = read(root / "reports" / f"uniform_odd_legacy_{i:02d}.json")
        unchanged.append(previous["nll_sum"] == current["nll_sum"])
    assert all(unchanged), "Unchanged Uniform-Odd quality regressed"
    suites = list(ET.parse(root / "regression_final.xml").getroot().iter("testsuite"))
    assert all(
        int(s.get("errors", "0")) == int(s.get("failures", "0")) == 0 for s in suites
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    assert head == protocol["base_commit"] and branch == protocol["branch"]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only"], text=True
    ).splitlines()
    changed += subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], text=True
    ).splitlines()
    artifacts = [
        *(root / "reports").glob("*.json"),
        *(root / "plans").glob("*.json"),
        *(root / "offline").glob("*.json"),
        *(root / "figures").glob("*.svg"),
        *(
            path
            for group in physical["groups"]
            for path in (root / group).glob("*.json")
        ),
        *root.glob("*.json"),
        *root.glob("*.csv"),
        *root.glob("*.xml"),
        snapshot_path,
        *(root / "logs").glob("pipeline_events.jsonl"),
        *(root / "logs").glob("pipeline_attempt*.log"),
        *(root / "logs").glob("search*.log"),
    ]
    save_new(
        root / "final_verification.json",
        {
            "passed": True,
            "branch": branch,
            "head": head,
            "automatic_commit": False,
            "preserved_profile_count": len(PROFILES),
            "previous_sources_verified": len(preserved_sources),
            "replay_exact_cases": len(preflight["cases"]),
            "cached_document_evaluations": score_count,
            "no_independent_test_in_search_cache": True,
            "legacy_uniform_odd_nll_unchanged": unchanged,
            "regression_tests": sum(int(s.get("tests", "0")) for s in suites),
            "performance_sessions": 3 * len(physical["groups"]),
            "local_certificate_passed": certificate["single_swap_local_optimum"],
            "local_certificate_scope": certificate["scope"],
            "known_limit": "Previous standalone V4 AR128 scale gate failure remains; this fixed-V2 allocation experiment neither reruns nor relaxes that gate.",
            "changed_files_sha256": {p: digest(Path(p)) for p in sorted(set(changed))},
            "artifacts_sha256": {str(p): digest(p) for p in sorted(set(artifacts))},
        },
    )
    print(
        "Final evidence audit passed; defaults and prior results preserved; no commit."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    final(parser.parse_args().root.resolve())
