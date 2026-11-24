# SPDX-License-Identifier: BSD-3-Clause
"""Verify the conditional Qwen3 relative-error study and export its result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from compare_cross_model_budgets import digest, read
from cross_model_plans import _freeze_json
from relative_kv_plans import allocation_study
from report_local_kv_scores import export_csv, verify_nll


def calibration_table(root: Path, docs: list[dict], scores: dict) -> list[dict]:
    if (
        scores["states"] != [[4, 2], [4, 4], [8, 2], [8, 4]]
        or scores["epsilon"] != 1e-12
    ):
        raise ValueError("Unexpected calibration state order or denominator constant")
    energies, numerators = [], []
    for doc in docs:
        path = root / "calibration" / (Path(doc["file"]).stem + ".json")
        row = read(path)
        if (
            row["identity"] != scores["scoring_identity"]
            or row["reference_passes"] != 1
            or row["states"] != scores["states"]
            or row["file"] != doc["file"]
            or row["tokens_sha256"] != doc["tokens_sha256"]
            or digest(path) != scores["documents_sha256"][doc["file"]]
            or digest(root / "assets" / doc["file"]) != doc["tokens_sha256"]
        ):
            raise ValueError(f"Corrupt calibration input or result: {path}")
        energies.append(row["reference_energy"])
        numerators.append(row["error_sums"])
    if np.shape(energies) != (8, 28) or np.shape(numerators) != (8, 28, 4):
        raise ValueError("Incomplete document or layer calibration statistics")
    energy = np.sum(energies, axis=0, dtype=np.float64)
    numerator = np.sum(numerators, axis=0, dtype=np.float64)
    relative = numerator / (energy[:, None] + scores["epsilon"])
    if (
        not np.array_equal(energy, scores["reference_energy"])
        or not np.array_equal(numerator, scores["error_sums"])
        or not np.array_equal(relative, scores["relative_errors"])
    ):
        raise ValueError("Published relative errors disagree with document aggregates")
    return [
        {
            "layer": layer,
            "k_bits": k,
            "v_bits": v,
            "reference_energy": float(energy[layer]),
            "sum_squared_error": float(numerator[layer, col]),
            "relative_error": float(relative[layer, col]),
        }
        for layer in range(28)
        for col, (k, v) in enumerate(scores["states"])
    ]


def report(root: Path, out: Path, prior: Path) -> None:
    identity_path = root / "evaluation_identity.json"
    identity, sha = read(identity_path), digest(identity_path)
    for field, path in [
        ("protocol_sha256", root / "protocol.json"),
        ("data_sha256", root / "data_manifest.json"),
        ("relative_scores_sha256", root / "relative_scores.json"),
        ("prior_plans_sha256", prior / "ranked_plans.json"),
    ]:
        if digest(path) != identity[field]:
            raise ValueError(f"Input checksum differs: {path}")
    windows = read(root / "data_manifest.json")["windows"]
    scores = read(root / "relative_scores.json")
    local_identity_path = root / "relative_scoring_identity.json"
    local_identity = read(local_identity_path)
    if (
        scores["scoring_identity"] != digest(local_identity_path)
        or local_identity["data_manifest_sha256"] != identity["data_sha256"]
        or local_identity["protocol_sha256"] != identity["protocol_sha256"]
    ):
        raise ValueError("Calibration scores and evaluation use different identities")
    layers = calibration_table(root, windows["calibration"], scores)
    gate = read(root / "expansion_gate.json")
    decision = read(root / "validation_decision.json")
    result = read(root / "evaluation_result.json")
    if (
        any(r["identity"] != sha for r in [gate, decision, result])
        or digest(root / "expansion_gate.json") != decision["gate_sha256"]
        or digest(root / "validation_decision.json") != result["decision_sha256"]
    ):
        raise ValueError("Validation-to-test identity chain mismatch")
    historical = read(prior / "ranked_plans.json")
    fixed = allocation_study(scores["relative_errors"], historical)
    if read(root / "fixed_plans.json") != fixed:
        raise ValueError("Fixed masks differ from calibration DP")
    val, test = decision["validation"], result["test"]
    controls = ["early", "absolute_07", "relative_07"]
    opened = val["relative_07"]["mean_nll"] < val["absolute_07"]["mean_nll"]
    expanded = root / "expanded_plans.json"
    if (
        gate["validation"] != {key: val[key] for key in controls}
        or opened != gate["open"]
        or opened != expanded.exists()
    ):
        raise ValueError("Budget expansion gate did not follow fixed validation NLLs")
    study = fixed
    new_ids = ["relative_07"]
    if opened:
        study = allocation_study(scores["relative_errors"], historical, expand=True)
        if read(expanded) != study:
            raise ValueError("Expanded masks differ from calibration DP")
        best = f"relative_{study['best_k']:02}"
        if best not in new_ids:
            new_ids.append(best)
    plans = study["plans"]
    winner = min(
        new_ids, key=lambda key: (val[key]["mean_nll"], key != "relative_07", key)
    )
    if (
        decision["new_candidate_ids"] != new_ids
        or set(val) != set(controls + new_ids)
        or set(test) != {"early", winner}
        or result["selected"] != winner
        or decision["selected"] != winner
        or decision["selected_config_hash"] != plans[winner]["config_hash"]
    ):
        raise ValueError("Test roster or validation winner does not reproduce")

    doc_rows = []
    for stage, record, summaries in [
        ("validation", decision, val),
        ("test", result, test),
    ]:
        if set(record["source_ids"]) != set(summaries):
            raise ValueError("Incomplete source aliases")
        raw_summaries = {}
        for name, raw_id in record["source_ids"].items():
            if (
                plans[name]["config_hash"] != plans[raw_id]["config_hash"]
                or summaries[name] != summaries[raw_id]
            ):
                raise ValueError(f"Mismatched alias: {name}")
            raw_summaries[raw_id] = summaries[raw_id]
        doc_rows.extend(
            verify_nll(root, stage, raw_summaries, plans, windows[stage], sha)
        )
    deltas = np.array(
        [
            test[winner]["documents"][r["file"]] - test["early"]["documents"][r["file"]]
            for r in windows["test"]
        ]
    )
    expected_ci = np.quantile(
        np.random.default_rng(20261124).choice(deltas, size=(10000, 8)).mean(axis=1),
        [0.025, 0.975],
    )
    if (
        float(deltas.mean()) != result["test_delta_nll"]
        or not np.array_equal(expected_ci, result["paired_bootstrap_95"])
        or [int((deltas < 0).sum()), int((deltas == 0).sum()), int((deltas > 0).sum())]
        != result["test_doc_counts_lower_equal_higher"]
    ):
        raise ValueError("Test NLL difference or paired-bootstrap summary mismatch")

    memory = read(root / "cache_memory.json")
    comparisons = []
    for name in dict.fromkeys(controls + new_ids):
        p = plans[name]
        mem = memory[p["config_hash"]]
        if (
            mem["packed_payload_bytes"] != p["cache_bytes_per_token"]["indices"] * 1024
            or mem["norm_bytes"] != p["cache_bytes_per_token"]["fp16_scales"] * 1024
            or mem["allocated_bytes"] != mem["packed_payload_bytes"] + mem["norm_bytes"]
            or mem["allocated_bytes"] != 30_277_632
        ):
            raise ValueError(f"Incorrect measured cache bytes: {name}")
        comparisons.append(
            {
                "plan_id": name,
                "k8_count": len(p["k8_layers"]),
                "v4_count": len(p["v4_layers"]),
                "mean_k_bits": p["key_average_bits"],
                "mean_v_bits": p["value_average_bits"],
                "k8_layers": " ".join(map(str, p["k8_layers"])),
                "v4_layers": " ".join(map(str, p["v4_layers"])),
                "sum_relative_error": p["sum_relative_error"],
                "validation_nll": val[name]["mean_nll"],
                "validation_ppl": val[name]["ppl"],
                "test_nll": test.get(name, {}).get("mean_nll", ""),
                "test_ppl": test.get(name, {}).get("ppl", ""),
                "validation_source_id": decision["source_ids"][name],
                "test_source_id": result["source_ids"].get(name, ""),
                "selected_new": name == winner,
                "allocated_cache_bytes": mem["allocated_bytes"],
                "config_hash": p["config_hash"],
            }
        )
    quota_rows = []
    for k in study["quotas"]:
        plan = plans[f"relative_{k:02}"]
        quota_rows.append(
            {
                "k8_count": k,
                "v4_count": len(plan["v4_layers"]),
                "mean_k_bits": plan["key_average_bits"],
                "mean_v_bits": plan["value_average_bits"],
                "sum_relative_error": plan["sum_relative_error"],
                "best_local_in_evaluated_sweep": k == study["best_k"],
                "k8_layers": " ".join(map(str, plan["k8_layers"])),
                "v4_layers": " ".join(map(str, plan["v4_layers"])),
                "config_hash": plan["config_hash"],
            }
        )
    for filename, rows in [
        ("layer_scores.csv", layers),
        ("quota_plans.csv", quota_rows),
        ("comparison.csv", comparisons),
        ("document_nll.csv", doc_rows),
    ]:
        export_csv(out / filename, rows)
    _freeze_json(out / "selected_plan.json", plans[winner])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    report(
        args.root, args.out, args.prior or args.root.parent / "local_kv_scores_20261123"
    )
