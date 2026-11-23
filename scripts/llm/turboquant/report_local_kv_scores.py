# SPDX-License-Identifier: BSD-3-Clause
"""Verify the 1.7B local-KV study and export paper-ready tables and winner."""

from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path

import numpy as np
from compare_cross_model_budgets import digest, read
from cross_model_plans import _freeze_json
from local_kv_plans import NUM_LAYERS, allocation_study, validation_winner
from refinement_data import publish


def export_csv(path: Path, rows: list[dict]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    publish(path, stream.getvalue().encode("utf-8"))


def verify_nll(
    root: Path, split: str, summaries: dict, plans: dict, docs: list, identity: str
) -> list[dict]:
    rows = []
    for name, summary in summaries.items():
        total_loss = []
        tokens = 0
        for doc in docs:
            file = doc["file"]
            path = root / "nll" / split / name / (Path(file).stem + ".json")
            raw = read(path)
            if (
                raw["evaluation"] != identity
                or raw["plan"] != plans[name]["config_hash"]
                or raw["stage"] != split
                or raw["file"] != file
                or raw["tokens_sha256"] != doc["tokens_sha256"]
                or raw["scored_tokens"] != doc["scored_tokens"]
                or raw["scored_tokens"] != 1023
            ):
                raise ValueError(f"Invalid raw score identity: {path}")
            mean = raw["nll_sum"] / raw["scored_tokens"]
            if not math.isclose(mean, summary["documents"][file], abs_tol=1e-12):
                raise ValueError(f"Document NLL differs: {path}")
            total_loss.append(raw["nll_sum"])
            tokens += raw["scored_tokens"]
            rows.append(
                {
                    "split": split,
                    "plan_id": name,
                    "file": file,
                    "title": doc["title"],
                    "tokens_sha256": raw["tokens_sha256"],
                    "config_hash": raw["plan"],
                    "nll_sum": raw["nll_sum"],
                    "scored_tokens": raw["scored_tokens"],
                    "mean_nll": mean,
                    "ppl": math.exp(mean),
                    "nll_minus_early": mean - summaries["early"]["documents"][file],
                    "raw_sha256": digest(path),
                }
            )
        mean = math.fsum(total_loss) / tokens
        if (
            tokens != summary["scored_tokens"]
            or not math.isclose(mean, summary["mean_nll"], abs_tol=1e-12)
            or not math.isclose(math.exp(mean), summary["ppl"], abs_tol=1e-12)
        ):
            raise ValueError(f"Aggregate differs for {split}/{name}")
    return rows


def report(root: Path, out: Path) -> None:
    local = read(root / "local_scores.json")
    identity = read(root / "evaluation_identity.json")
    evaluation_sha = digest(root / "evaluation_identity.json")
    decision = read(root / "validation_decision.json")
    result = read(root / "evaluation_result.json")
    for field, filename in [
        ("local_scores", "local_scores.json"),
        ("manifest", "data_manifest.json"),
        ("protocol", "protocol.json"),
    ]:
        if digest(root / filename) != identity[field]:
            raise ValueError(f"Identity mismatch: {filename}")
    if (
        decision["identity"] != evaluation_sha
        or result["identity"] != evaluation_sha
        or result["validation_decision_sha256"]
        != digest(root / "validation_decision.json")
    ):
        raise ValueError("Frozen selection/test hashes differ")
    study = allocation_study(local["key_gains"], local["value_gains"])
    if study != read(root / "ranked_plans.json"):
        raise ValueError("Ranked plan file differs from calibration scores")
    plans, ids = study["plans"], study["validation_ids"]
    valid, test = decision["validation"], result["test"]
    winner = validation_winner(study, {i: r["mean_nll"] for i, r in valid.items()})
    if (
        decision["evaluated_ids"] != ids
        or winner != decision["selected"]
        or plans[winner]["config_hash"] != decision["selected_config_hash"]
        or winner != result["selected"]
        or set(test) != {winner, "early"}
    ):
        raise ValueError(
            "Candidate roster or winner differs from the selection protocol"
        )
    windows = read(root / "data_manifest.json")["windows"]
    calibration_mses = []
    capture_id = digest(root / "local_scoring_identity.json")
    for doc in windows["calibration"]:
        raw = read(root / "local_scores" / (Path(doc["file"]).stem + ".json"))
        if (
            raw["source_identity_sha256"] != capture_id
            or raw["token_sha256"] != doc["tokens_sha256"]
            or raw["file"] != doc["file"]
            or raw["reference_passes"] != 1
            or raw["mse_columns"] != local["mse_columns"]
        ):
            raise ValueError(f"Invalid calibration result: {doc['file']}")
        calibration_mses.append(raw["mse"])
    if (
        np.shape(calibration_mses) != (8, NUM_LAYERS, 3)
        or capture_id != local["source_identity_sha256"]
        or not np.array_equal(np.mean(calibration_mses, axis=0), local["mean_mse"])
    ):
        raise ValueError("Local mean errors do not reproduce the calibration summary")
    doc_rows = verify_nll(
        root, "validation", valid, plans, windows["validation"], evaluation_sha
    )
    doc_rows += verify_nll(root, "test", test, plans, windows["test"], evaluation_sha)
    memory = read(root / "cache_memory.json")
    predicted = plans[f"ranked_{study['best_k']:02}"]["config_hash"]
    if local["mse_columns"] != ["E42", "E82", "E44"]:
        raise ValueError("Expected ordered MSE columns E42/E82/E44")
    layer_rows = []
    for i in range(NUM_LAYERS):
        e42, e82, e44 = local["mean_mse"][i]
        gk, gv = local["key_gains"][i], local["value_gains"][i]
        if not (
            math.isclose(e42 - e82, gk, abs_tol=1e-12)
            and math.isclose(e42 - e44, gv, abs_tol=1e-12)
        ):
            raise ValueError(f"Gains differ from measured MSE at layer {i}")
        layer_rows.append(
            {
                "layer": i,
                "mse_k4_v2": e42,
                "mse_k8_v2": e82,
                "mse_k4_v4": e44,
                "key_gain": gk,
                "value_gain": gv,
                "key_rank": study["k_rank"].index(i) + 1,
                "value_rank": study["v_rank"].index(i) + 1,
            }
        )
    evaluated = {plans[name]["config_hash"]: name for name in ids}
    budget_rows = []
    for k in range(NUM_LAYERS // 2 + 1):
        p = plans[f"ranked_{k:02}"]
        budget_rows.append(
            {
                "k8_count": k,
                "v4_count": len(p["v4_layers"]),
                "mean_k_bits": p["key_average_bits"],
                "mean_v_bits": p["value_average_bits"],
                "predicted_gain": p["predicted_gain"],
                "predicted_best": k == study["best_k"],
                "k8_layers": " ".join(map(str, p["k8_layers"])),
                "v4_layers": " ".join(map(str, p["v4_layers"])),
                "validation_plan_id": evaluated.get(p["config_hash"], ""),
                "config_hash": p["config_hash"],
            }
        )
    shortlist = []
    for name in ids:
        p = plans[name]
        shortlist.append(
            {
                "plan_id": name,
                "k8_count": len(p["k8_layers"]),
                "v4_count": len(p["v4_layers"]),
                "predicted_gain": p["predicted_gain"],
                "predicted_best": p["config_hash"] == predicted,
                "validation_nll": valid[name]["mean_nll"],
                "validation_ppl": valid[name]["ppl"],
                "validation_winner": name == winner,
                "test_nll": test.get(name, {}).get("mean_nll", ""),
                "test_ppl": test.get(name, {}).get("ppl", ""),
                "allocated_cache_bytes": memory[name]["allocated_bytes"],
                "config_hash": p["config_hash"],
            }
        )
    for filename, rows in [
        ("layer_scores.csv", layer_rows),
        ("budget_curve.csv", budget_rows),
        ("shortlist.csv", shortlist),
        ("document_nll.csv", doc_rows),
    ]:
        export_csv(out / filename, rows)
    _freeze_json(out / "selected_plan.json", plans[winner])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.out)
