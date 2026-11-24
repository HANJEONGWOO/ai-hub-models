# SPDX-License-Identifier: BSD-3-Clause
"""Validate the fixed four-state allocation, conditionally expand the budget, then score test."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from allocation_offline_model import Model
from compare_cross_model_budgets import digest, read, sources
from cross_model_plans import _freeze_json
from evaluate_local_kv_scores import MODEL_ID, score_split, verified_windows
from relative_kv_plans import allocation_study

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import get_profile

EXPERIMENTS = Path(__file__).parent / "experiments"
PROTOCOL = EXPERIMENTS / "relative_kv/protocol.json"
STATES = [[4, 2], [4, 4], [8, 2], [8, 4]]


def verify_inputs(root: Path, archive: Path) -> tuple[dict, list, dict, str]:
    prior = archive / "local_kv_scores_20261123"
    windows = verified_windows(root)
    old_windows = read(prior / "data_manifest.json")["windows"]
    if windows["calibration"] != old_windows["calibration"]:
        raise ValueError("Expected the exact previous eight calibration windows")
    previous_articles = {
        r["document_sha256"] for cohort in old_windows.values() for r in cohort
    }
    if any(
        r["document_sha256"] in previous_articles
        for split in ("validation", "test")
        for r in windows[split]
    ):
        raise ValueError("New evaluation split overlaps a previous article")
    local = read(root / "relative_scores.json")
    if local["states"] != STATES or digest(root / "protocol.json") != digest(PROTOCOL):
        raise ValueError("Calibration state order or experiment protocol changed")
    previous_source = prior / "evaluation_identity.json"
    legacy = read(previous_source)
    for name, expected in legacy["graphs_and_inputs"].items():
        if digest(Path(name)) != expected:
            raise ValueError(
                f"W4A16 source identity differs from the preceding run: {name}"
            )
    old_plans = read(prior / "ranked_plans.json")
    with (EXPERIMENTS / "local_kv_scores/shortlist.csv").open() as f:
        frozen_controls = {r["plan_id"]: r["config_hash"] for r in csv.DictReader(f)}
    for name in ("early", "ranked_07"):
        if old_plans["plans"][name]["config_hash"] != frozen_controls[name]:
            raise ValueError(f"Historical control changed: {name}")
    identity = _freeze_json(
        root / "evaluation_identity.json",
        {
            "protocol_sha256": digest(PROTOCOL),
            "data_sha256": digest(root / "data_manifest.json"),
            "relative_scores_sha256": digest(root / "relative_scores.json"),
            "prior_plans_sha256": digest(prior / "ranked_plans.json"),
            "previous_w4a16_identity_sha256": digest(previous_source),
            "software": {
                s: digest(Path(__file__).with_name(s))
                for s in (
                    "evaluate_relative_kv.py",
                    "relative_kv_plans.py",
                    "select_mixed_layers.py",
                    "evaluate_local_kv_scores.py",
                    "allocation_offline_model.py",
                )
            },
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
        },
    )
    return windows, local["relative_errors"], old_plans, identity


def compare(root: Path, archive: Path) -> None:
    start = time.monotonic()
    windows, costs, old_plans, identity = verify_inputs(root, archive)
    rank_start = time.monotonic()
    checks_s = rank_start - start
    study = allocation_study(costs, old_plans, expand=False)
    plans = study["plans"]
    _freeze_json(root / "fixed_plans.json", study)
    fixed_s = time.monotonic() - rank_start
    bundle, assets, split = sources(archive, MODEL_ID)
    load_start = time.monotonic()
    model = Model(
        bundle,
        assets,
        device="cuda",
        split_dir=split,
        stream_parts=True,
        stream_weights=True,
    )
    if sorted(i for part in model.parts for i in part.macros) != list(range(28)):
        raise ValueError("Model is missing attention layers")
    for part in model.parts:
        part.to("cuda")
    model.stream_parts = False
    torch.cuda.synchronize()
    val_start = time.monotonic()
    memory = {}
    cache = {"validation": {}, "test": {}}
    aliases = {"validation": {}, "test": {}}

    def evaluate(stage: str, name: str) -> dict:
        plan = plans[name]
        cfg = get_profile(plan["profile"], layer_plan=plan)
        mask_hash = cfg.config_hash()
        if mask_hash != plan["config_hash"]:
            raise ValueError(f"Codec identity mismatch for {name}")
        if mask_hash not in memory:
            state = TurboQuantKVCache(cfg, 28, 8, 128, 1024)
            memory[mask_hash] = state.memory_report(1024).to_dict()
            if memory[mask_hash]["allocated_bytes"] != 30277632:
                raise ValueError(f"Unequal physical packed capacity for {name}")
            del state
        entries = cache[stage]
        if mask_hash not in entries:
            summaries = score_split(
                root, model, identity, stage, windows[stage], [name], plans
            )
            entries[mask_hash] = (name, summaries[name])
        raw_id, summary = entries[mask_hash]
        aliases[stage][name] = raw_id
        return summary

    fixed = ["early", "absolute_07", "relative_07"]
    val = {name: evaluate("validation", name) for name in fixed}
    expand = val["relative_07"]["mean_nll"] < val["absolute_07"]["mean_nll"]
    _freeze_json(
        root / "expansion_gate.json",
        {"identity": identity, "validation": val, "open": expand},
    )
    print(f"Fixed-quota expansion gate: {expand}", flush=True)
    new_ids = ["relative_07"]
    expanded_s = 0.0
    if expand:
        rank_start = time.monotonic()
        study = allocation_study(costs, old_plans, expand=True)
        plans = study["plans"]
        _freeze_json(root / "expanded_plans.json", study)
        expanded_s = time.monotonic() - rank_start
        best = f"relative_{study['best_k']:02d}"
        if best not in val:
            new_ids.append(best)
            val[best] = evaluate("validation", best)
    winner = min(new_ids, key=lambda n: (val[n]["mean_nll"], n != "relative_07", n))
    decision_sha = _freeze_json(
        root / "validation_decision.json",
        {
            "identity": identity,
            "gate_sha256": digest(root / "expansion_gate.json"),
            "selected": winner,
            "selected_config_hash": plans[winner]["config_hash"],
            "new_candidate_ids": new_ids,
            "validation": val,
            "source_ids": aliases["validation"],
        },
    )
    test_start = time.monotonic()
    test = {name: evaluate("test", name) for name in ("early", winner)}
    _freeze_json(root / "cache_memory.json", memory)
    deltas = np.asarray(
        [
            test[winner]["documents"][r["file"]] - test["early"]["documents"][r["file"]]
            for r in windows["test"]
        ]
    )
    ci = np.quantile(
        np.random.default_rng(20261124)
        .choice(deltas, size=(10000, 8), replace=True)
        .mean(axis=1),
        [0.025, 0.975],
    )
    _freeze_json(
        root / "evaluation_result.json",
        {
            "identity": identity,
            "decision_sha256": decision_sha,
            "selected": winner,
            "test": test,
            "source_ids": aliases["test"],
            "test_delta_nll": float(deltas.mean()),
            "paired_bootstrap_95": ci.tolist(),
            "test_doc_counts_lower_equal_higher": [
                int((deltas < 0).sum()),
                int((deltas == 0).sum()),
                int((deltas > 0).sum()),
            ],
        },
    )
    timer = root / "evaluation_timing.json"
    if not timer.exists():
        _freeze_json(
            timer,
            {
                "source_checks_s": checks_s,
                "fixed_planning_s": fixed_s,
                "expanded_planning_s": expanded_s,
                "model_load_s": val_start - load_start,
                "validation_and_gate_wall_s": test_start - val_start,
                "selection_wall_s": test_start - start,
                "test_and_summary_s": time.monotonic() - test_start,
                "validation_forward_s": sum(
                    row["forward_seconds"] for _, row in cache["validation"].values()
                ),
                "test_forward_s": sum(
                    row["forward_seconds"] for _, row in cache["test"].values()
                ),
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--archive", type=Path, default=Path("/mnt/d/ai-hub-models/binaries/turboquant")
    )
    args = p.parse_args()
    compare(args.root, args.archive)
