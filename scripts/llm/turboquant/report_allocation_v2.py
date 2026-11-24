# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Read-only progress or final allocation-search quality/cost comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import statistics
import time
from pathlib import Path

from benchmark_asymmetric_once import digest, read, save_new
from summarize_native_results import performance


def database(root: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{root / 'offline/scores.sqlite'}?mode=ro", uri=True)


def snapshot_cache(root: Path) -> dict:
    """Preserve all committed WAL rows in a standalone, non-overwriting DB."""
    path = root / "offline/scores_release.sqlite"
    with path.open("xb"):
        pass
    source, target = database(root), sqlite3.connect(path)
    try:
        source.backup(target)
        assert target.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
        counts = {
            table: target.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("scores", "requests", "attempts")
        }
        assert target.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        source.close()
        target.close()
    return {"path": str(path), "sha256": digest(path), "rows": counts}


def statistics_of(values: list[float]) -> dict:
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected finite nonempty measurements")
    return {
        "mean": statistics.mean(values),
        "std_ddof1": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def paired(candidate: dict, reference: dict) -> dict:
    left, right = candidate["documents"], reference["documents"]
    if [d["tokens_sha256"] for d in left] != [d["tokens_sha256"] for d in right]:
        raise ValueError("Cannot compare unmatched documents")
    delta = [a["mean_nll"] - b["mean_nll"] for a, b in zip(left, right, strict=True)]
    positive_gains = [-d for d in delta if d < 0]
    return {
        "delta_mean_nll": candidate["mean_nll"] - reference["mean_nll"],
        "ppl_relative_improvement_percent": 100
        * (1 - candidate["ppl"] / reference["ppl"]),
        "document_deltas": delta,
        "improved_documents": sum(d < -1e-6 for d in delta),
        "worsened_documents": sum(d > 1e-6 for d in delta),
        "tied_documents": sum(abs(d) <= 1e-6 for d in delta),
        "largest_document_share_of_positive_gains": (
            max(positive_gains) / sum(positive_gains) if positive_gains else None
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def progress(root: Path) -> dict:
    with database(root) as db:
        counts = db.execute(
            "SELECT length,COUNT(*),SUM(elapsed_s) FROM scores GROUP BY length"
        ).fetchall()
        runs = db.execute(
            "SELECT run,started,finished FROM attempts ORDER BY id"
        ).fetchall()
    finals = []
    for method in ("greedy", "beam"):
        for name in read(root / "protocol.json")["search"]["starts"]:
            path = root / "offline" / f"{method}_{name}.json"
            if path.exists():
                try:
                    row = read(path)
                except json.JSONDecodeError:
                    # Live status can briefly see an in-progress checkpoint.
                    continue
                finals.append(
                    {
                        "run": row["run"],
                        "initial_nll": row["initial_nll"],
                        "final_nll": row["final_nll"],
                        "steps": len(row["history"]),
                        "layers": row["final_layers"],
                    }
                )
    journal = root / "logs/pipeline_events.jsonl"
    last_event = None
    if journal.exists():
        for line in reversed(journal.read_text().splitlines()):
            try:
                last_event = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    return {
        "physical_document_evaluations": {str(n): c for n, c, _ in counts},
        "evaluation_seconds": sum(t for _, _, t in counts),
        "runs": [
            {"run": n, "finished": end is not None, "seconds": (end or time.time()) - s}
            for n, s, end in runs
        ],
        "finals": finals,
        "pipeline_event": last_event,
        "htp_ready": [
            p.stem.removeprefix("ready_")
            for p in sorted((root / "reports").glob("ready_*.json"))
        ],
        "stages": {
            n: (root / (n + ".json")).exists()
            for n in ("nominations", "selection", "local_certificate", "comparison")
        },
    }


def quality(root: Path, group: str, role: str, config_hash: str) -> dict:
    protocol = read(root / "protocol.json")
    data = read(root / "data_manifest.json")
    docs = (
        [
            {"file": f, "title": f}
            for f in protocol["assets_manifest"]["wikitext_windows"]
        ]
        if role == "legacy"
        else data["windows"][role]
    )
    rows = []
    for i, doc in enumerate(docs):
        path = root / "reports" / f"{group}_{role}_{i:02d}.json"
        r = read(path)
        sha = protocol["assets_manifest"]["sha256"][doc["file"]]
        assert r["config_hash"] == config_hash
        assert r["assets"]["tokens_sha256"] == sha
        assert r["scored_tokens"] == 1023 and r["kv_store_bytes"] == 30277632
        assert "htp " in r["backend_api"] and r["device"]["soc_model"] == "SM8850"
        nll = r["nll_sum"] / r["scored_tokens"]
        assert math.isfinite(nll) and math.isclose(
            r["ppl"], math.exp(nll), rel_tol=1e-7
        )
        rows.append(
            {
                "physical_group": group,
                "split": role,
                "window": i,
                "title": doc["title"].strip("= ").strip(),
                "tokens_sha256": sha,
                "scored_tokens": r["scored_tokens"],
                "nll_sum": r["nll_sum"],
                "mean_nll": nll,
                "ppl": math.exp(nll),
                "report_sha256": digest(path),
            }
        )
    total, tokens = (
        sum(r["nll_sum"] for r in rows),
        sum(r["scored_tokens"] for r in rows),
    )
    return {
        "mean_nll": total / tokens,
        "ppl": math.exp(total / tokens),
        "nll_sum": total,
        "scored_tokens": tokens,
        "documents": rows,
    }


def offline_quality(root: Path, layers: list[int], role: str) -> dict | None:
    mask = f"{sum(1 << i for i in layers):07x}"
    data = read(root / "data_manifest.json")["windows"][role]
    identity = digest(root / "offline/scorer_identity.json")
    rows = []
    with database(root) as db:
        for doc in data:
            r = db.execute(
                "SELECT nll_sum,scored_tokens FROM scores "
                "WHERE identity=? AND mask=? AND document=? AND length=1024",
                (identity, mask, doc["tokens_sha256"]),
            ).fetchall()
            if not r:
                return None
            assert len(r) == 1 and r[0][1] == 1023
            total, count = r[0]
            rows.append(
                {
                    "title": doc["title"].strip("= ").strip(),
                    "tokens_sha256": doc["tokens_sha256"],
                    "nll_sum": total,
                    "scored_tokens": count,
                    "mean_nll": total / count,
                }
            )
    return {
        "mean_nll": sum(r["nll_sum"] for r in rows)
        / sum(r["scored_tokens"] for r in rows),
        "documents": rows,
    }


def seed_rows(nominations: dict) -> tuple[list[dict], list[dict]]:
    finals, steps = [], []
    for method, starts in nominations["all_seed_finals"].items():
        for start, r in starts.items():
            finals.append(
                {
                    "method": method,
                    "start": start,
                    "initial_nll": r["initial_nll"],
                    "final_nll": r["final_nll"],
                    "delta_nll": r["final_nll"] - r["initial_nll"],
                    "steps": len(r["history"]),
                    "layers": " ".join(map(str, r["final_layers"])),
                    "logical_1024_document_evaluations": r["cost"][
                        "logical_unique_mask_document_evaluations"
                    ].get("1024", 0),
                    "logical_256_document_evaluations": r["cost"][
                        "logical_unique_mask_document_evaluations"
                    ].get("256", 0),
                    "incremental_evaluation_seconds": r["cost"][
                        "physical_evaluation_seconds"
                    ],
                    "incremental_wall_seconds": r["cost"][
                        "completed_attempt_wall_seconds"
                    ],
                }
            )
            steps.append(
                {"method": method, "start": start, "step": 0, "nll": r["initial_nll"]}
            )
            for h in r["history"]:
                steps.append(
                    {
                        "method": method,
                        "start": start,
                        "step": h.get("step", h.get("iteration")),
                        "nll": (
                            h["incumbent_nll"]
                            if method == "beam"
                            else h["best_swap"]["mean_nll"]
                            if h["accepted"]
                            else h["before_nll"]
                        ),
                    }
                )
    return finals, steps


def mask_workload(root: Path, runs: list[str]) -> dict:
    marks = ",".join("?" for _ in runs)
    with database(root) as db:
        rows = db.execute(
            f"SELECT DISTINCT mask,length FROM requests WHERE run IN ({marks})",
            runs,
        ).fetchall()
    return {
        str(length): {
            "unique_14_layer_masks": sum(
                n == length and int(mask, 16).bit_count() == 14 for mask, n in rows
            ),
            "unique_13_or_15_layer_diagnostic_masks": sum(
                n == length and int(mask, 16).bit_count() in (13, 15)
                for mask, n in rows
            ),
        }
        for length in (256, 1024)
    }


def offline_documents(root: Path) -> list[dict]:
    labels = {}
    data = read(root / "data_manifest.json")["windows"]
    for role in ("search", "validation"):
        for doc in data[role]:
            raw = (root / "assets" / doc["file"]).read_bytes()
            for length in (256, 1024):
                labels[(hashlib.sha256(raw[: length * 4]).hexdigest(), length)] = (
                    role,
                    doc["title"].strip("= ").strip(),
                )
    with database(root) as db:
        values = db.execute(
            "SELECT mask,document,length,nll_sum,scored_tokens,owner,config,identity "
            "FROM scores ORDER BY mask,length,document"
        ).fetchall()
    odd = f"{sum(1 << i for i in range(1, 28, 2)):07x}"
    baseline = {
        (sha, n): total / count
        for mask, sha, n, total, count, *_ in values
        if mask == odd
    }
    rows = []
    for mask, sha, n, total, count, owner, config_hash, identity in values:
        role, title = labels[(sha, n)]
        nll = total / count
        reference = baseline.get((sha, n))
        rows.append(
            {
                "mask_hex": mask,
                "k8_count": int(mask, 16).bit_count(),
                "diagnostic_only": int(mask, 16).bit_count() != 14,
                "split": role,
                "title": title,
                "input_length": n,
                "tokens_sha256": sha,
                "nll_sum": total,
                "scored_tokens": count,
                "mean_nll": nll,
                "delta_vs_odd_if_measured": None
                if reference is None
                else nll - reference,
                "physical_owner_run": owner,
                "config_hash": config_hash,
                "scorer_identity": identity,
            }
        )
    return rows


def greedy_full_scores(root: Path) -> dict[tuple[int, ...], float]:
    """Only search-set scores already available before any Beam start."""
    docs = read(root / "data_manifest.json")["windows"]["search"]
    hashes = [digest(root / "assets" / d["file"]) for d in docs]
    marks = ",".join("?" for _ in hashes)
    with database(root) as db:
        rows = db.execute(
            "SELECT mask,SUM(nll_sum)/SUM(scored_tokens) FROM scores "
            f"WHERE length=1024 AND document IN ({marks}) "
            "AND identity=? AND owner GLOB 'greedy_*' "
            "GROUP BY mask HAVING COUNT(*)=?",
            [*hashes, digest(root / "offline/scorer_identity.json"), len(hashes)],
        ).fetchall()
    return {
        tuple(i for i in range(28) if int(mask, 16) & (1 << i)): nll
        for mask, nll in rows
        if int(mask, 16).bit_count() == 14
    }


def screening_diagnostics(nominations: dict, cached: dict) -> dict:
    """Read-only diagnostic; never uses validation/test or changes nominations."""
    rows = []
    for start, result in nominations["all_seed_finals"]["beam"].items():
        if not result["history"]:
            continue
        visited = {tuple(result["initial_layers"])}
        incumbent = result["initial_nll"]
        for step in result["history"]:
            evaluated = {tuple(r["layers"]) for r in step["full_evaluated"]}
            selected_best = min(r["mean_nll"] for r in step["full_evaluated"])
            retained = step["incumbent_nll"]
            assert retained <= incumbent + 1e-6
            neighbors = {
                tuple(r["layers"])
                for parent in step["parents"]
                for r in parent["all_196_swaps"]
            }
            covered = (neighbors - visited - evaluated) & cached.keys()
            best = (
                min(covered, key=lambda m: (cached[m], sum(1 << i for i in m)))
                if covered
                else None
            )
            ranks = []
            if best is not None:
                for parent in step["parents"]:
                    ranked = sorted(
                        (
                            r
                            for r in parent["all_196_swaps"]
                            if tuple(r["layers"]) not in visited
                        ),
                        key=lambda r: (
                            r["coarse_nll"],
                            sum(1 << i for i in r["layers"]),
                        ),
                    )
                    for rank, row in enumerate(ranked, 1):
                        if tuple(row["layers"]) == best:
                            assert rank > len(parent["full_shortlist"])
                            ranks.append(rank)
            gain = max(0.0, retained - cached[best]) if best is not None else 0.0
            rows.append(
                {
                    "start": start,
                    "step": step["step"],
                    "previous_incumbent_nll": incumbent,
                    "selected_step_best_nll": selected_best,
                    "retained_best_nll": retained,
                    "excluded_neighbors_with_prior_greedy_full_scores": len(covered),
                    "best_excluded_layers": list(best) if best is not None else None,
                    "best_excluded_full_nll": cached[best]
                    if best is not None
                    else None,
                    "coarse_ranks_among_unvisited_neighbors": ranks,
                    "additional_search_nll_reduction_missed": gain,
                }
            )
            visited.update(evaluated)
            incumbent = step["incumbent_nll"]
    return {
        "steps_examined": len(rows),
        "steps_with_missed_improvement": sum(
            r["additional_search_nll_reduction_missed"] > 1e-6 for r in rows
        ),
        "rows": rows,
        "limits": [
            "Only pre-Beam Greedy full search-set scores; no extra inference.",
            "Coverage is a subset of neighbors, not an oracle over all 196 swaps.",
            "Search-set opportunity, not evidence of HTP or independent-test benefit.",
            "Diagnostic only: no nomination, selected mask or search rule changes.",
        ],
    }


def figures(root: Path, groups: dict, steps: list[dict], selected: str) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = root / "figures"
    target.mkdir(exist_ok=False)
    structured = {g: r for g, r in groups.items() if r["k8_layers"] is not None}
    array = np.array(
        [
            [8 if i in r["k8_layers"] else 4 for i in range(28)]
            for r in structured.values()
        ]
    )
    fig, ax = plt.subplots(figsize=(14, max(3, 0.45 * len(structured) + 1.5)))
    ax.imshow(array, aspect="auto", cmap="Blues", vmin=4, vmax=8)
    ax.set_xticks(range(28))
    ax.set_yticks(
        range(len(structured)),
        [n + (" [selected]" if n == selected else "") for n in structured],
    )
    ax.set_xlabel("Layer (zero-based); all V2")
    ax.set_title("14 Structured K8 / 14 LM K4; 28.875 MiB allocated KV")
    for row in range(len(structured)):
        for col in range(28):
            ax.text(
                col,
                row,
                str(array[row, col]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if array[row, col] == 8 else "black",
            )
    fig.tight_layout()
    fig.savefig(target / "layer_masks.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for method, ax in zip(("greedy", "beam"), axes, strict=True):
        for start in read(root / "protocol.json")["search"]["starts"]:
            rows = [r for r in steps if r["method"] == method and r["start"] == start]
            ax.plot(
                [r["step"] for r in rows], [r["nll"] for r in rows], ".-", label=start
            )
        ax.set_title(method + " (unequal total budgets)")
        ax.set_xlabel("Search step")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Full offline search-12 incumbent mean token NLL")
    fig.tight_layout()
    fig.savefig(target / "search_history.svg")
    plt.close(fig)
    certificate = read(root / "local_certificate.json")
    demotions = sorted(certificate["layers"])
    promotions = sorted(set(range(28)) - set(demotions))
    lookup = {(s["promote"], s["demote"]): s["delta_nll"] for s in certificate["swaps"]}
    assert len(demotions) == len(promotions) == 14 and len(lookup) == 196
    deltas = np.array([[lookup[p, d] for p in promotions] for d in demotions])
    assert np.isfinite(deltas).all()
    span = max(float(np.abs(deltas).max()), 1e-12)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(deltas, cmap="RdBu_r", vmin=-span, vmax=span)
    ax.set_xticks(range(14), promotions)
    ax.set_yticks(range(14), demotions)
    ax.set_xlabel("Promote K4 to K8 (layer)")
    ax.set_ylabel("Demote K8 to K4 (layer)")
    ax.set_title("All 196 swaps: full offline search-12, not HTP/test")
    row, col = np.unravel_index(np.argmin(deltas), deltas.shape)
    ax.scatter([col], [row], marker="*", s=110, c="black", label="Best checked swap")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.colorbar(im, ax=ax, label="Swap NLL - frozen selected NLL")
    fig.tight_layout()
    fig.savefig(target / "local_swap_delta.svg")
    plt.close(fig)


def summary(root: Path) -> None:
    selection = read(root / "selection.json")
    certificate = read(root / "local_certificate.json")
    assert certificate["selection_sha256"] == digest(root / "selection.json")
    physical = read(root / "physical_groups.json")
    nominations = read(root / "nominations.json")
    protocol = read(root / "protocol.json")
    groups, documents = {}, []
    for group in physical["groups"]:
        converted = read(root / group / "convert_report.json")
        sha = converted["config_hash"]
        perf = [
            performance(root / "reports" / f"{group}_perf_r{i}.json", 897)
            for i in range(3)
        ]
        stats = {
            k: statistics_of([p[k] for p in perf])
            for k in (
                "ttft_ms",
                "prefill_tok_per_s",
                "decode_tok_per_s",
                "host_kv_MiB",
                "io_buffer_MiB",
                "end_VmRSS_MiB",
                "process_VmHWM_MiB",
                "mean_decode_prepare_ms",
                "mean_decode_qnn_ms",
                "mean_decode_commit_ms",
            )
        }
        assert stats["host_kv_MiB"]["samples"] == [28.875] * 3
        roles = ["test", "legacy"]
        if group in selection["htp_mean_nll"]:
            roles += ["search", "validation"]
        q = {role: quality(root, group, role, sha) for role in roles}
        for result in q.values():
            documents.extend(result["documents"])
        groups[group] = {
            "config_hash": sha,
            "k8_layers": None
            if group == "k6_v2"
            else read(root / "plans" / f"{group}.json")["k8_layers"],
            "quality": q,
            "performance": stats,
            "raw_performance": perf,
            "binaries": {
                p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                for p in sorted((root / group).glob("part*_of_4.bin"))
            },
        }
    rows, contrasts = [], {}
    for label, group in physical["aliases"].items():
        entry = groups[group]
        contrasts[label] = {
            reference: paired(
                entry["quality"]["test"],
                groups[physical["aliases"][reference]]["quality"]["test"],
            )
            for reference in (
                "uniform_odd",
                "early",
                "boundary",
                "legacy_greedy",
                "multi_greedy",
                "k6_v2",
            )
        }
        rows.append(
            {
                "configuration": label,
                "physical_group": group,
                "k8_layers": " ".join(map(str, entry["k8_layers"] or [])),
                "config_hash": entry["config_hash"],
                **{
                    f"{s}_{m}": entry["quality"][s][m]
                    for s in ("test", "legacy")
                    for m in ("mean_nll", "ppl")
                },
                **{
                    f"{k}_{stat}": v[stat]
                    for k, v in entry["performance"].items()
                    for stat in ("mean", "std_ddof1", "min", "max")
                },
                "binary_bytes": sum(v["bytes"] for v in entry["binaries"].values()),
            }
        )
    transfer = {}
    for group in selection["htp_mean_nll"]:
        transfer[group] = {
            role: {
                "offline": offline_quality(root, groups[group]["k8_layers"], role),
                "htp": groups[group]["quality"][role],
            }
            for role in ("search", "validation")
        }
    finals, steps = seed_rows(nominations)
    method_runs = {
        "legacy_greedy": ["greedy_uniform_odd"],
        "multi_greedy": ["greedy_" + n for n in protocol["search"]["starts"]],
        "multi_beam": ["beam_" + n for n in protocol["search"]["starts"]],
    }
    result = {
        "rows": rows,
        "groups": groups,
        "aliases": physical["aliases"],
        "test_contrasts": contrasts,
        "calibration_transfer": transfer,
        "method_costs": nominations["method_costs"],
        "method_mask_workloads": {
            name: mask_workload(root, runs) for name, runs in method_runs.items()
        },
        "seed_finals": finals,
        "coarse_screen_diagnostics": screening_diagnostics(
            nominations, greedy_full_scores(root)
        ),
        "selection": selection,
        "local_certificate": certificate,
        "protocol_sha256": digest(root / "protocol.json"),
        "data_manifest_sha256": digest(root / "data_manifest.json"),
        "score_cache_snapshot": snapshot_cache(root),
        "limits": [
            "New independent test is not a claim of pretraining/calibration exclusion.",
            "Offline CUDA replay is eager-exact, not HTP-exact; missing offline validation is not remeasured.",
            "Three crossed timing sessions report variability, not statistical superiority.",
            "Search methods have unequal total budgets; shared-cache incremental wall time is not cold-cache latency.",
            "Method clocks start after model/replay setup and do not separately charge HTP compilation or calibration; concurrent CPU builds can affect elapsed time.",
            "physical_evaluation_seconds is synchronized end-to-end offline scoring wall time, not GPU busy time.",
            "A 196-swap check applies only to the frozen offline search set, not HTP/global optimality.",
            "Historical WikiText scores are reference-only and not used to choose allocations.",
            "Inherited standalone encoder AR128 scale limitation is not a new validation pass.",
        ],
        "performance_sessions": 3 * len(groups),
        "test_scored_tokens_per_group": 8 * 1023,
        "requested_groups": protocol["evaluation"]["groups"],
    }
    for name, table in (
        ("comparison.csv", rows),
        ("document_nll.csv", documents),
        ("seed_finals.csv", finals),
        ("search_history.csv", steps),
        ("offline_document_nll.csv", offline_documents(root)),
    ):
        write_csv(root / name, table)
    figures(root, groups, steps, selection["selected_group"])
    save_new(root / "comparison.json", result)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("progress", "summary"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--brief", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "progress":
        value = progress(root)
        if args.brief:
            active = next(
                (r["run"] for r in reversed(value["runs"]) if not r["finished"]), None
            )
            value = {
                "active": active,
                "completed_steps": len(
                    list((root / "offline").glob(f"{active}_step*.json"))
                )
                if active
                else None,
                "completed_starts": len(value["finals"]),
                "document_evaluations": value["physical_document_evaluations"],
                "pipeline": value["pipeline_event"],
                "ready": value["htp_ready"],
                "stages": value["stages"],
            }
        print(json.dumps(value, indent=None if args.brief else 2))
    else:
        summary(root)
