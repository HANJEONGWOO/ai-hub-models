# SPDX-License-Identifier: BSD-3-Clause
"""Score all six adjacent K/V allocations on the frozen validation cohort."""

from __future__ import annotations

import argparse
import fcntl
import math
import time
from pathlib import Path

import numpy as np
import torch
from adjacent_kv_inputs import ALIASES, reusable, verify_inputs
from adjacent_kv_plans import build_plans
from allocation_offline_model import Model
from compare_cross_model_budgets import digest, read, sources
from cross_model_plans import _freeze_json
from refinement_data import publish

SCRIPTS = Path(__file__).resolve().parent
PROTOCOL = SCRIPTS / "experiments/adjacent_kv/protocol.json"
QUOTAS = SCRIPTS / "experiments/relative_kv/quota_plans.csv"


def load_model(archive: Path) -> Model:
    bundle, assets, split = sources(archive, "qwen3_1_7b")
    model = Model(
        bundle,
        assets,
        device="cuda",
        split_dir=split,
        stream_parts=True,
        stream_weights=True,
    )
    if sorted(i for part in model.parts for i in part.macros) != list(range(28)):
        raise ValueError("Expected 28 W4A16 attention blocks")
    for part in model.parts:
        part.to("cuda")
    model.stream_parts = False
    torch.cuda.synchronize()
    return model


def compare(root: Path, prior: Path) -> None:
    started = time.monotonic()
    publish(root / "protocol.json", PROTOCOL.read_bytes())
    protocol = read(PROTOCOL)
    if digest(QUOTAS) != protocol["source_quota_table_sha256"]:
        raise ValueError("Frozen DP quota table changed")
    plans = build_plans(QUOTAS)
    if set(plans) != set(protocol["candidates"]):
        raise ValueError("Candidate roster differs from protocol")
    plans_sha = _freeze_json(root / "plans.json", plans)
    docs, previous_identity = verify_inputs(prior, plans)
    if digest(prior / "data_manifest.json") != protocol["prior_manifest_sha256"]:
        raise ValueError("Unexpected validation cohort")
    identity = _freeze_json(
        root / "evaluation_identity.json",
        {
            **previous_identity,
            "protocol_sha256": digest(PROTOCOL),
            "plans_sha256": plans_sha,
            "software_sha256": {
                f: digest(SCRIPTS / f)
                for f in (
                    "evaluate_adjacent_kv.py",
                    "adjacent_kv_inputs.py",
                    "adjacent_kv_plans.py",
                )
            },
        },
    )
    print(f"Verified input and evaluator identity {identity}", flush=True)
    verified_at = time.monotonic()
    model = None
    load_s = 0.0
    new_forward_s = 0.0
    invoked = 0
    summaries = {}
    for name in protocol["candidates"]:
        plan = plans[name]
        rows = []
        for doc in docs:
            stem = Path(doc["file"]).stem + ".json"
            dest = root / "nll/validation" / name / stem
            key = {
                "evaluation": identity,
                "file": doc["file"],
                "plan": plan["config_hash"],
                "tokens_sha256": doc["tokens_sha256"],
                "stage": "validation",
            }
            old = reusable(prior, name, plan, doc, previous_identity)
            if old is not None:
                source_name = ALIASES[name]
                old_path = prior / "nll/validation" / source_name / stem
                reused = {
                    **old,
                    **key,
                    "source_kind": "reused",
                    "source_plan_id": source_name,
                    "source_raw_sha256": digest(old_path),
                }
                _freeze_json(dest, reused)
            elif not dest.exists():
                if model is None:
                    before = time.monotonic()
                    model = load_model(prior.parent)
                    load_s = time.monotonic() - before
                tokens = np.fromfile(prior / "assets" / doc["file"], dtype="<i4")
                stats, logits = model.score(
                    tokens, plan["k8_layers"], v4_layers=plan["v4_layers"]
                )
                del logits
                new_forward_s += stats["elapsed_s"]
                invoked += 1
                _freeze_json(dest, {**key, **stats, "source_kind": "new"})
            row = read(dest)
            if (
                any(row[k] != v for k, v in key.items())
                or row["source_kind"] != ("reused" if old is not None else "new")
                or row["scored_tokens"] != 1023
                or not math.isfinite(row["nll_sum"])
                or not math.isclose(
                    row["mean_nll"], row["nll_sum"] / 1023, rel_tol=1e-12
                )
            ):
                raise ValueError(f"Invalid document score: {dest}")
            rows.append(row)
        count = sum(r["scored_tokens"] for r in rows)
        if len(rows) != 8 or count != 8184:
            raise ValueError(f"Incomplete validation: {name}")
        nll = math.fsum(r["nll_sum"] for r in rows)
        reused_count = sum(r["source_kind"] == "reused" for r in rows)
        summaries[name] = {
            "documents": {r["file"]: r["nll_sum"] / r["scored_tokens"] for r in rows},
            "mean_nll": nll / count,
            "ppl": math.exp(nll / count),
            "nll_sum": nll,
            "scored_tokens": count,
            "reused": reused_count,
            "new": len(rows) - reused_count,
        }
        print(
            f"{name} NLL={nll / count:.8f} reused={reused_count}/{len(rows)}",
            flush=True,
        )
    ranking = sorted(
        summaries, key=lambda n: (summaries[n]["mean_nll"], n != "early_07", n)
    )
    winner = ranking[0]
    total_reused = sum(v["reused"] for v in summaries.values())
    total_new = sum(v["new"] for v in summaries.values())
    _freeze_json(
        root / "validation_result.json",
        {
            "identity": identity,
            "selected": winner,
            "selected_config_hash": plans[winner]["config_hash"],
            "ranking": ranking,
            "summaries": summaries,
            "reused": total_reused,
            "new": total_new,
        },
    )
    if not (root / "timing.json").exists():
        _freeze_json(
            root / "timing.json",
            {
                "verification_and_planning_s": verified_at - started,
                "model_load_s": load_s,
                "new_forward_s": new_forward_s,
                "invocation_wall_s": time.monotonic() - started,
                "invocation_new_forwards": invoked,
                "resumed_new_rows": total_new - invoked,
            },
        )
    print(
        f"Selected {winner} from 48 validation scores ({total_reused} prior, {total_new} new)",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--prior", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    prior = args.prior or root.parent / "relative_kv_20261124"
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        compare(root, prior.resolve())
