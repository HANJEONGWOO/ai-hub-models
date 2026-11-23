# SPDX-License-Identifier: BSD-3-Clause
"""Freeze Qwen3-1.7B attention-ranked plans, select on validation, then open test."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from allocation_offline_model import Model
from compare_cross_model_budgets import digest, read, sources
from cross_model_plans import _freeze_json
from local_kv_plans import allocation_study, validation_winner

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import get_profile

MODEL_ID = "qwen3_1_7b"
PROTOCOL = Path(__file__).parent / "experiments/local_kv_scores/protocol.json"


def verified_windows(root: Path) -> dict:
    windows = read(root / "data_manifest.json")["windows"]
    rows = [
        row for split in ("calibration", "validation", "test") for row in windows[split]
    ]
    if any(len(windows[s]) != 8 for s in ("calibration", "validation", "test")):
        raise ValueError("Expected eight documents in each of three disjoint splits")
    for field in ("document_sha256", "tokens_sha256", "file"):
        if len({row[field] for row in rows}) != 24:
            raise ValueError(f"Repeated {field}")
    for row in rows:
        path = root / "assets" / row["file"]
        if path.stat().st_size != 4096 or digest(path) != row["tokens_sha256"]:
            raise ValueError(f"Token window identity mismatch: {path}")
    return windows


def source_identity(root: Path, archive: Path) -> tuple:
    bundle, assets, split = sources(archive, MODEL_ID)
    previous = archive / "cross_model_pu_budget_20261120/scorer" / (MODEL_ID + ".json")
    identity = read(previous)
    files = {
        **identity["graphs_sha256"],
        str(Path(__file__).with_name("allocation_offline_model.py")): identity[
            "scorer"
        ],
        str(assets / "assets.json"): identity["input_assets"],
        str(split / "split_manifest.json"): identity["split_manifest_sha256"],
    }
    asset_index = read(assets / "assets.json")
    rope = asset_index["rope"]
    files[str(assets / rope)] = asset_index["sha256"][rope]
    inventory = PROTOCOL.parent.parent / "cross_model_pu_budget/weights.csv"
    with inventory.open() as stream:
        weights = [r for r in csv.DictReader(stream) if r["model_id"] == MODEL_ID]
    if len(weights) != 4:
        raise ValueError("Incomplete external weight inventory")
    for record in weights:
        graph = archive.parents[1] / record["replay_graph"]
        weight = graph.parent / record["external_location"]
        if not weight.exists() or weight.stat().st_size != int(record["weight_bytes"]):
            raise ValueError(f"Missing or truncated model weights: {weight}")
        files[str(weight)] = record["sha256"]
    if any(digest(Path(p)) != sha for p, sha in files.items()):
        raise ValueError(
            "W4A16 replay artifacts or source differ from the historical scorer"
        )
    key = _freeze_json(
        root / "evaluation_identity.json",
        {
            "protocol": digest(PROTOCOL),
            "manifest": digest(root / "data_manifest.json"),
            "local_scores": digest(root / "local_scores.json"),
            "ranking_code": digest(Path(__file__).with_name("local_kv_plans.py")),
            "scoring_code": digest(Path(__file__)),
            "historical_scorer": digest(previous),
            "graphs_and_inputs": files,
            "device": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        },
    )
    return bundle, assets, split, key


def score_split(
    root: Path,
    model: Model,
    identity: str,
    stage: str,
    windows: list[dict],
    ids: list[str],
    plans: dict,
) -> dict:
    summaries = {}
    for name in ids:
        plan = plans[name]
        rows = []
        for doc in windows:
            dest = root / "nll" / stage / name / (Path(doc["file"]).stem + ".json")
            key = {
                "evaluation": identity,
                "plan": plan["config_hash"],
                "tokens_sha256": doc["tokens_sha256"],
                "file": doc["file"],
                "stage": stage,
            }
            if not dest.exists():
                tokens = np.fromfile(root / "assets" / doc["file"], dtype="<i4")
                stats, logits = model.score(
                    tokens, plan["k8_layers"], v4_layers=plan["v4_layers"]
                )
                del logits
                _freeze_json(dest, {**key, **stats})
            row = read(dest)
            if (
                any(row[k] != v for k, v in key.items())
                or row["scored_tokens"] != 1023
                or not math.isfinite(row["nll_sum"])
            ):
                raise ValueError(f"Invalid raw score: {dest}")
            rows.append(row)
        count = sum(r["scored_tokens"] for r in rows)
        mean = math.fsum(r["nll_sum"] for r in rows) / count
        summaries[name] = {
            "mean_nll": mean,
            "ppl": math.exp(mean),
            "scored_tokens": count,
            "documents": {r["file"]: r["nll_sum"] / r["scored_tokens"] for r in rows},
            "forward_seconds": math.fsum(r["elapsed_s"] for r in rows),
        }
        print(f"{stage} {name} NLL={mean:.7f}, PPL={math.exp(mean):.5f}", flush=True)
    return summaries


def run(root: Path, archive: Path) -> None:
    started = time.monotonic()
    if digest(root / "protocol.json") != digest(PROTOCOL):
        raise ValueError("Selection protocol does not match the frozen experiment")
    windows = verified_windows(root)
    bundle, assets, split, ident = source_identity(root, archive)
    scores = read(root / "local_scores.json")
    rank_start = time.monotonic()
    study = allocation_study(scores["key_gains"], scores["value_gains"])
    _freeze_json(root / "ranked_plans.json", study)
    rank_seconds = time.monotonic() - rank_start
    plans, ids = study["plans"], study["validation_ids"]
    control = read(
        archive / "cross_model_pu_budget_20261120/plans" / MODEL_ID / "k5_v3_early.json"
    )
    if (
        plans["early"]["config_hash"] != control["config_hash"]
        or plans["early"]["bits"] != control["bits"]
    ):
        raise ValueError("Unexpected historical Early mask")
    memory = {}
    for name in ids:
        plan = plans[name]
        cfg = get_profile(plan["profile"], layer_plan=plan)
        if cfg.config_hash() != plan["config_hash"]:
            raise ValueError(f"Invalid canonical codec: {name}")
        cache = TurboQuantKVCache(cfg, 28, 8, 128, 1024)
        memory[name] = cache.memory_report(1024).to_dict()
        del cache
        if memory[name]["allocated_bytes"] != 30277632:
            raise ValueError(f"Wrong packed capacity for {name}")
    _freeze_json(root / "cache_memory.json", memory)
    model_start = time.monotonic()
    model = Model(
        bundle,
        assets,
        device="cuda",
        split_dir=split,
        stream_parts=True,
        stream_weights=True,
    )
    if sorted(i for p in model.parts for i in p.macros) != list(range(28)):
        raise ValueError("Incomplete W4A16 replay")
    for part in model.parts:
        part.to("cuda")
    model.stream_parts = False
    torch.cuda.synchronize()
    loaded = time.monotonic()
    valid = score_split(
        root, model, ident, "validation", windows["validation"], ids, plans
    )
    selected = validation_winner(
        study, {name: r["mean_nll"] for name, r in valid.items()}
    )
    _freeze_json(
        root / "validation_decision.json",
        {
            "identity": ident,
            "evaluated_ids": ids,
            "validation": valid,
            "selected": selected,
            "selected_config_hash": plans[selected]["config_hash"],
        },
    )
    selected_at = time.monotonic()
    test_ids = list(dict.fromkeys([selected, "early"]))
    test = score_split(root, model, ident, "test", windows["test"], test_ids, plans)
    deltas = np.array(
        [
            test[selected]["documents"][d["file"]]
            - test["early"]["documents"][d["file"]]
            for d in windows["test"]
        ]
    )
    ci = np.quantile(
        np.random.default_rng(20261123)
        .choice(deltas, size=(10000, 8), replace=True)
        .mean(axis=1),
        [0.025, 0.975],
    ).tolist()
    _freeze_json(
        root / "evaluation_result.json",
        {
            "identity": ident,
            "validation_decision_sha256": digest(root / "validation_decision.json"),
            "selected": selected,
            "test": test,
            "mean_delta_nll": float(deltas.mean()),
            "paired_bootstrap_95": ci,
            "distinct_test_plans": len(test_ids),
        },
    )
    timer = root / "evaluation_timing.json"
    if not timer.exists():
        _freeze_json(
            timer,
            {
                "ranking_and_freeze_s": rank_seconds,
                "model_load_s": loaded - model_start,
                "validation_wall_s": selected_at - loaded,
                "selection_total_including_checks_s": selected_at - started,
                "test_and_summary_wall_s": time.monotonic() - selected_at,
                "validation_forward_s": sum(
                    x["forward_seconds"] for x in valid.values()
                ),
                "test_forward_s": sum(x["forward_seconds"] for x in test.values()),
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--archive", type=Path, default=Path("/mnt/d/ai-hub-models/binaries/turboquant")
    )
    options = parser.parse_args()
    run(options.root, options.archive)
