# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Cached multi-start legacy greedy and interaction-aware two-stage beam search."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from benchmark_asymmetric_once import digest, read, save_new
from benchmark_nll_allocation import STATIC
from search_nll_allocation import greedy, mask_key

from qai_hub_models.models.templates.llm.turboquant.config import (
    structured_allocation_plan,
)

Mask = tuple[int, ...]


def canonical_mask(layers: list[int] | Mask, *, diagnostic: bool = False) -> Mask:
    if (
        len(layers) not in ((13, 14, 15) if diagnostic else (14,))
        or any(type(i) is not int or not 0 <= i < 28 for i in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError("Invalid layer mask/count")
    return tuple(sorted(layers))


def mask_number(mask: Mask) -> int:
    return sum(1 << i for i in mask)


def swaps(mask: Mask) -> list[dict]:
    current = set(canonical_mask(mask))
    result = [
        {
            "promote": p,
            "demote": d,
            "layers": tuple(sorted((current - {d}) | {p})),
        }
        for p in range(28)
        if p not in current
        for d in sorted(current)
    ]
    assert len(result) == len({x["layers"] for x in result}) == 196
    return sorted(result, key=lambda x: mask_number(x["layers"]))


def save_consistent(path: Path, value: dict) -> None:
    # JSON normalizes tuples. Resume may replay arithmetic from immutable scores.
    normalized = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if read(path) != normalized:
            raise ValueError(f"Existing checkpoint differs: {path}")
    else:
        save_new(path, normalized)


def beam_search(
    coarse: Callable[[Mask], float],
    full: Callable[[Mask], float],
    initial: list[int],
    *,
    width: int = 2,
    steps: int = 8,
    shortlist: int = 4,
    epsilon: float = 1e-6,
    checkpoint: Callable[[dict], None] | None = None,
) -> dict:
    if not 1 <= width <= 3 or steps < 1 or shortlist < width:
        raise ValueError("Invalid beam budget")
    start = canonical_mask(initial)
    incumbent, incumbent_nll = start, full(start)
    initial_nll = incumbent_nll
    frontier = [(start, incumbent_nll)]
    visited = {start}
    history = []
    for step in range(1, steps + 1):
        parents, candidates = [], set()
        for parent, parent_nll in frontier:
            neighbors = [
                {**s, "coarse_nll": coarse(s["layers"])} for s in swaps(parent)
            ]
            ranked = sorted(
                (s for s in neighbors if s["layers"] not in visited),
                key=lambda s: (s["coarse_nll"], mask_number(s["layers"])),
            )
            selected = ranked[:shortlist]
            candidates.update(s["layers"] for s in selected)
            parents.append(
                {
                    "layers": parent,
                    "full_nll": parent_nll,
                    "all_196_swaps": neighbors,
                    "full_shortlist": [s["layers"] for s in selected],
                }
            )
        if not candidates:
            break
        evaluated = sorted(
            ((mask, full(mask)) for mask in sorted(candidates, key=mask_number)),
            key=lambda x: (x[1], mask_number(x[0])),
        )
        visited.update(candidates)
        improved = evaluated[0][1] < incumbent_nll - epsilon
        if improved:
            incumbent, incumbent_nll = evaluated[0]
        # Do not keep parents in the frontier: permit temporary worsening while
        # the separately retained incumbent can never regress.
        frontier = evaluated[:width]
        row = {
            "step": step,
            "parents": parents,
            "full_evaluated": [{"layers": m, "mean_nll": n} for m, n in evaluated],
            "next_frontier": [{"layers": m, "mean_nll": n} for m, n in frontier],
            "incumbent_layers": incumbent,
            "incumbent_nll": incumbent_nll,
            "improved_incumbent": improved,
        }
        history.append(row)
        if checkpoint:
            checkpoint(row)
    return {
        "initial_layers": start,
        "initial_nll": initial_nll,
        "final_layers": incumbent,
        "final_nll": incumbent_nll,
        "history": history,
        "full_masks_considered": len(visited),
    }


def local_certificate(
    full: Callable[[Mask], float], layers: list[int], epsilon: float = 1e-6
) -> dict:
    mask = canonical_mask(layers)
    baseline = full(mask)
    rows = [{**s, "mean_nll": full(s["layers"])} for s in swaps(mask)]
    for row in rows:
        row["delta_nll"] = row["mean_nll"] - baseline
    best = min(rows, key=lambda x: (x["mean_nll"], mask_number(x["layers"])))
    return {
        "layers": mask,
        "mean_nll": baseline,
        "swaps": rows,
        "best_swap": best,
        "single_swap_local_optimum": best["delta_nll"] >= -epsilon,
        "improving_swaps": sum(r["delta_nll"] < -epsilon for r in rows),
        "scope": "Full offline search-12 only; not an HTP or global-optimality certificate",
        "changes_final_selection": False,
    }


class Scores:
    """Append-only per-document scores; requests track logical versus real work."""

    def __init__(self, root: Path, replays: dict, identity: dict) -> None:
        self.root, self.replays = root, replays
        save_consistent(root / "offline/scorer_identity.json", identity)
        self.identity = digest(root / "offline/scorer_identity.json")
        self.data = read(root / "data_manifest.json")["windows"]
        self.inputs = {
            d["file"]: np.fromfile(root / "assets" / d["file"], dtype="<i4")
            for role in ("search", "validation")
            for d in self.data[role]
        }
        self.db = sqlite3.connect(root / "offline/scores.sqlite", timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS scores(
                identity TEXT, config TEXT, mask TEXT, document TEXT, length INTEGER,
                nll_sum REAL NOT NULL, scored_tokens INTEGER NOT NULL,
                elapsed_s REAL NOT NULL, owner TEXT NOT NULL,
                PRIMARY KEY(identity,config,mask,document,length));
            CREATE TABLE IF NOT EXISTS requests(
                run TEXT, identity TEXT, config TEXT, mask TEXT, document TEXT,
                length INTEGER, cache_hit INTEGER,
                PRIMARY KEY(run,identity,config,mask,document,length));
            CREATE TABLE IF NOT EXISTS attempts(
                id INTEGER PRIMARY KEY, run TEXT, started REAL, finished REAL);
        """)
        self.db.commit()
        self.run = "unassigned"
        self.calls = 0

    def __call__(self, layers: list[int] | Mask, split: str = "search") -> float:
        mask = canonical_mask(layers, diagnostic=True)
        if split not in ("coarse", "search", "validation"):
            raise ValueError("The independent test is never an offline selection input")
        if len(mask) == 14:
            config = structured_allocation_plan(list(mask))["config_hash"]
        else:
            # These 13/15-layer sensitivity probes are legacy-algorithm-only.
            recipe = {
                "type": "nondeployable_single_flip",
                "layers": mask,
                "base": structured_allocation_plan(STATIC["uniform_odd"])["config"],
            }
            config = hashlib.sha256(
                json.dumps(recipe, sort_keys=True).encode()
            ).hexdigest()
        rows = self.data["search"][:2] if split == "coarse" else self.data[split]
        length = 256 if split == "coarse" else 1024
        total, count = 0.0, 0
        for doc in rows:
            tokens = self.inputs[doc["file"]][:length]
            sha = hashlib.sha256(tokens.tobytes()).hexdigest()
            key = (self.identity, config, mask_key(list(mask)), sha, length)
            found = self.db.execute(
                "SELECT nll_sum,scored_tokens,elapsed_s FROM scores "
                "WHERE identity=? AND config=? AND mask=? AND document=? AND length=?",
                key,
            ).fetchone()
            hit = found is not None
            if found is None:
                value = self.replays[length].score(tokens, list(mask))
                if value["scored_tokens"] != length - 1 or not np.isfinite(
                    value["nll_sum"]
                ):
                    raise ValueError("Invalid NLL record")
                found = (value["nll_sum"], value["scored_tokens"], value["elapsed_s"])
                self.db.execute(
                    "INSERT INTO scores VALUES(?,?,?,?,?,?,?,?,?)",
                    (*key, *found, self.run),
                )
            self.db.execute(
                "INSERT OR IGNORE INTO requests VALUES(?,?,?,?,?,?,?)",
                (self.run, *key, int(hit)),
            )
            self.db.commit()
            total += found[0]
            count += found[1]
        self.calls += 1
        if split != "coarse" or self.calls % 32 == 0:
            print(
                "SCORE",
                self.run,
                split,
                mask_key(list(mask)),
                total / count,
                flush=True,
            )
        return total / count

    def cost(self, runs: list[str]) -> dict:
        marks = ",".join("?" for _ in runs)
        logical = self.db.execute(
            f"SELECT length,COUNT(*) FROM (SELECT DISTINCT identity,config,mask,document,length "
            f"FROM requests WHERE run IN ({marks})) GROUP BY length",
            runs,
        ).fetchall()
        physical = self.db.execute(
            f"SELECT length,COUNT(*),SUM(elapsed_s) FROM scores WHERE owner IN ({marks}) GROUP BY length",
            runs,
        ).fetchall()
        attempts = self.db.execute(
            f"SELECT run,started,finished FROM attempts WHERE run IN ({marks})", runs
        ).fetchall()
        return {
            "logical_unique_mask_document_evaluations": {str(n): k for n, k in logical},
            "logical_scored_tokens": sum((n - 1) * k for n, k in logical),
            "physical_cache_misses": {str(n): k for n, k, _ in physical},
            "physical_evaluation_seconds": sum(t for _, _, t in physical),
            "attempts": [{"run": r, "start": s, "end": e} for r, s, e in attempts],
            "completed_attempt_wall_seconds": sum(
                e - s for _, s, e in attempts if e is not None
            ),
            "incomplete_attempts": sum(e is None for _, _, e in attempts),
            "limits": "Incremental wall/GPU time benefits from globally shared cache. Logical distinct requests are a from-scratch workload proxy, not measured cold-cache latency.",
        }


def initialize(root: Path) -> tuple[dict, Scores]:
    from allocation_v2_replay import PREPARED, GraphReplay, load

    protocol = read(root / "protocol.json")
    preflight = read(root / "replay_preflight_v2.json")
    assert preflight["passed"]
    script = Path(__file__)
    numerical = [
        script.with_name(n)
        for n in ("allocation_v2_replay.py", "allocation_offline_model.py")
    ]
    for n in ("config", "reference", "structured_k8"):
        numerical.append(
            Path("src/qai_hub_models/models/templates/llm/turboquant") / f"{n}.py"
        )
    identity = {
        "protocol_sha256": digest(root / "protocol.json"),
        "data_manifest_sha256": digest(root / "data_manifest.json"),
        "prepared_model_sha256": preflight["prepared_model_sha256"],
        "numerical_sources": {str(p): digest(p) for p in numerical},
        "replay_preflight_sha256": digest(root / "replay_preflight_v2.json"),
        "device": preflight["device"],
        "torch": preflight["torch"],
        "preflight_exact_cases": len(preflight["cases"]),
        "limits": "Shape-matched eager exact; existing surrogate is not HTP exact",
    }
    assert (
        identity["numerical_sources"][str(numerical[0])]
        == preflight["replay_source_sha256"]
    )
    assert (
        identity["numerical_sources"][str(numerical[1])]
        == preflight["surrogate_source_sha256"]
    )
    assert protocol["frozen_files"][str(PREPARED)] == identity["prepared_model_sha256"]
    save_consistent(
        root / "offline/search_lock.json",
        {
            "algorithm_sha256": digest(script),
            "legacy_algorithm_sha256": digest(
                script.with_name("search_nll_allocation.py")
            ),
            "protocol_sha256": identity["protocol_sha256"],
        },
    )
    data = read(root / "data_manifest.json")["windows"]
    for role in ("search", "validation"):
        for row in data[role]:
            assert digest(root / "assets" / row["file"]) == row["tokens_sha256"]
    model = load()
    ids = np.fromfile(root / "assets" / data["search"][0]["file"], dtype="<i4")
    replays = {n: GraphReplay(model, ids[:n]) for n in (256, 1024)}
    print("Ready: cached exact offline replay, two fixed lengths", flush=True)
    return protocol, Scores(root, replays, identity)


def run_search(root: Path, protocol: dict, scorer: Scores) -> None:
    results = {"greedy": {}, "beam": {}}
    for method, method_results in results.items():
        for name, initial in STATIC.items():
            run = method + "_" + name
            path = root / "offline" / (run + ".json")
            if path.exists():
                method_results[name] = read(path)
                continue
            scorer.run = run
            attempt = scorer.db.execute(
                "INSERT INTO attempts(run,started) VALUES(?,?)", (run, time.time())
            ).lastrowid
            scorer.db.commit()

            def checkpoint(row: dict, run: str = run) -> None:
                step = row.get("step", row.get("iteration"))
                save_consistent(root / "offline" / f"{run}_step{step:02d}.json", row)
                print("STEP", run, step, flush=True)

            if method == "greedy":
                p = protocol["search"]["legacy"]
                result = greedy(
                    scorer,
                    initial,
                    p["max_steps"],
                    p["shortlist_per_direction"],
                    protocol["search"]["epsilon"],
                    checkpoint,
                )
            else:
                p = protocol["search"]["beam"]
                result = beam_search(
                    lambda m: scorer(m, "coarse"),
                    scorer,
                    initial,
                    width=p["width"],
                    steps=p["max_steps"],
                    shortlist=p["full_shortlist_per_parent"],
                    epsilon=protocol["search"]["epsilon"],
                    checkpoint=checkpoint,
                )
            scorer.db.execute(
                "UPDATE attempts SET finished=? WHERE id=?", (time.time(), attempt)
            )
            scorer.db.commit()
            result.update(
                run=run, cost=scorer.cost([run]), scorer_identity=scorer.identity
            )
            save_new(path, result)
            method_results[name] = result
            print("FINAL", run, result["final_layers"], result["final_nll"], flush=True)

    def best(rows: list[dict]) -> dict:
        return min(
            rows, key=lambda r: (r["final_nll"], mask_number(tuple(r["final_layers"])))
        )

    nominees = {
        "legacy_greedy": results["greedy"]["uniform_odd"],
        "multi_greedy": best(list(results["greedy"].values())),
        "multi_beam": best(list(results["beam"].values())),
    }
    unique = {tuple(r["final_layers"]) for r in nominees.values()}
    extra = 0
    for r in sorted(
        [v for method in results.values() for v in method.values()],
        key=lambda r: (r["final_nll"], mask_number(tuple(r["final_layers"]))),
    ):
        if len(unique) >= 2:
            break
        mask = tuple(r["final_layers"])
        if mask not in unique:
            extra += 1
            nominees[f"extra_candidate_{extra}"] = r
            unique.add(mask)
    assert len(unique) <= 3
    rows = {}
    scorer.run = "offline_validation"
    for name, result in nominees.items():
        plan = structured_allocation_plan(list(result["final_layers"]))
        save_consistent(root / "plans" / f"{name}.json", plan)
        rows[name] = {
            "layers": result["final_layers"],
            "search_nll": result["final_nll"],
            "validation_nll": scorer(result["final_layers"], "validation"),
            "config_hash": plan["config_hash"],
            "source_run": result["run"],
        }
    save_new(
        root / "nominations.json",
        {
            "nominees": rows,
            "all_seed_finals": results,
            "method_costs": {
                "legacy_greedy": scorer.cost(["greedy_uniform_odd"]),
                "multi_greedy": scorer.cost(["greedy_" + k for k in STATIC]),
                "multi_beam": scorer.cost(["beam_" + k for k in STATIC]),
            },
            "selection_rule": "Nominees minimize full search-12 NLL, not test or offline validation. Actual HTP validation-4 decides final mask.",
            "test_used": False,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("search", "certificate"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    target = root / (
        "nominations.json" if a.stage == "search" else "local_certificate.json"
    )
    if target.exists():
        raise FileExistsError(target)
    protocol, scorer = initialize(root)
    try:
        if a.stage == "search":
            run_search(root, protocol, scorer)
        else:
            selected = read(root / "selection.json")
            scorer.run = "final_local_certificate"
            result = local_certificate(
                scorer, selected["selected_layers"], protocol["search"]["epsilon"]
            )
            result["cost"] = scorer.cost([scorer.run])
            result["selection_sha256"] = digest(root / "selection.json")
            save_new(target, result)
    finally:
        scorer.db.close()


if __name__ == "__main__":
    main()
