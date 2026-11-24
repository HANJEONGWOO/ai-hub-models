# SPDX-License-Identifier: BSD-3-Clause
"""NIAH scoring/reporting only: no model execution, model changes, or LLM judge.

Scores are calculated from the decoded *generated continuation*, with special
tokens removed by the evaluator. Prompts and raw token IDs must be retained in
the experiment's source records, not prepended to ``response`` here.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

GROUPS = ("fp16", "k4_v4", "boundary_lm", "boundary_structured")
GROUP_LABELS = {
    "fp16": "FP16 KV",
    "k4_v4": "TurboQuant K4/V4",
    "boundary_lm": "Boundary converged LM8",
    "boundary_structured": "Boundary Structured K8",
}
CAPACITIES = (1024, 4096, 8192, 16384)
DEPTHS = (0, 25, 50, 75, 100)
BOOTSTRAP_SEED = 20261102
SUCCESS_STATUSES = frozenset(("ok", "success", "completed"))
ERROR_STATUSES = frozenset(("error", "runtime_error", "failed"))
CODE = re.compile(r"(?<!\w)[0-9]{6}(?!\w)")


def score_answer(response: str, expected: str) -> dict[str, Any]:
    """Exact match strips only outside whitespace; secondary match is unique.

    Repeated identical codes still constitute multiple candidates and fail the
    secondary metric. A correct six-digit response is never parsed as an int.
    """
    if not isinstance(expected, str) or re.fullmatch(r"[0-9]{6}", expected) is None:
        raise ValueError("expected_answer must be a six-digit ASCII string")
    if not isinstance(response, str):
        raise TypeError("response must be decoded generated text")
    stripped = response.strip()
    candidates = CODE.findall(stripped)
    return {
        "exact_match": stripped == expected,
        "extracted_match": len(candidates) == 1 and candidates[0] == expected,
        "empty_response": not stripped,
        "format_issue": re.fullmatch(r"[0-9]{6}", stripped) is None,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _case(record: dict[str, Any]) -> dict[str, Any]:
    value = dict(record)
    if "expected_answer" not in value and "answer" in value:
        value["expected_answer"] = value["answer"]
    for key in ("case_id", "base_id", "expected_answer"):
        if key not in value:
            raise ValueError(f"case missing {key}: {value.get('case_id')}")
    value["case_id"] = str(value["case_id"])
    value["base_id"] = str(value["base_id"])
    value["capacity"] = int(value["capacity"])
    value["depth"] = int(value["depth"])
    value["split"] = value.get("split", "test")
    score_answer("", value["expected_answer"])
    return value


def _truncated(record: dict[str, Any]) -> bool:
    if "output_truncated" in record:
        return bool(record["output_truncated"])
    reason = record.get("finish_reason")
    if reason in ("length", "max_tokens", "max_new_tokens"):
        return True
    if reason in ("eos", "stop"):
        return False
    generated, limit = record.get("generated_tokens"), record.get("max_new_tokens")
    if isinstance(generated, list):
        generated = len(generated)
    return generated is not None and limit is not None and int(generated) >= int(limit)


def _statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] in SUCCESS_STATUSES]
    counts = Counter(row["status"] for row in rows)
    n = len(complete)
    exact = sum(row["exact_match"] for row in complete)
    extracted = sum(row["extracted_match"] for row in complete)
    return {
        "n_expected": len(rows),
        "n_completed": n,
        "n_runtime_error": sum(counts[key] for key in ERROR_STATUSES),
        "n_unsupported": counts["unsupported"],
        "n_missing": counts["missing"],
        "n_other_unavailable": sum(
            count
            for key, count in counts.items()
            if key not in SUCCESS_STATUSES | ERROR_STATUSES | {"unsupported", "missing"}
        ),
        "coverage_pct": 100 * n / len(rows) if rows else 0.0,
        "complete": bool(rows) and n == len(rows),
        "exact_successes": exact,
        "extracted_successes": extracted,
        "exact_accuracy_pct": 100 * exact / n if n else None,
        "extracted_accuracy_pct": 100 * extracted / n if n else None,
        "n_empty": sum(row["empty_response"] for row in complete),
        "n_format_issue": sum(row["format_issue"] for row in complete),
        "n_multiple_candidates": sum(row["candidate_count"] > 1 for row in complete),
        "n_output_truncated": sum(row["output_truncated"] for row in complete),
        "status_counts": dict(sorted(counts.items())),
    }


def _cluster_comparisons(
    rows: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    groups: tuple[str, ...],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Use only entire length/depth cells complete in *every* comparison group."""
    conditions = sorted({(row["capacity"], row["depth"]) for row in cells})
    by_cell = {(row["group"], row["capacity"], row["depth"]): row for row in cells}
    common = [
        condition
        for condition in conditions
        if all(by_cell[(group, *condition)]["complete"] for group in groups)
    ]
    common_set = set(common)
    selected = [row for row in rows if (row["capacity"], row["depth"]) in common_set]
    base_ids = sorted({row["base_id"] for row in selected})
    values = np.zeros((len(groups), len(base_ids)), dtype=np.float64)
    counts = np.zeros_like(values)
    cases_by_group: dict[str, set[str]] = {group: set() for group in groups}
    for row in selected:
        g, b = groups.index(row["group"]), base_ids.index(row["base_id"])
        values[g, b] += row["exact_match"]
        counts[g, b] += 1
        cases_by_group[row["group"]].add(row["case_id"])
    if len({frozenset(value) for value in cases_by_group.values()}) > 1:
        raise ValueError("paired comparison requires identical complete case IDs")
    bootstrap = None
    if len(base_ids) >= 2 and samples:
        indices = np.random.default_rng(seed).integers(
            0, len(base_ids), size=(samples, len(base_ids))
        )
        bootstrap = (
            100 * values[:, indices].sum(axis=2) / counts[:, indices].sum(axis=2)
        )
    estimates = {
        group: 100 * values[i].sum() / counts[i].sum() if counts[i].sum() else None
        for i, group in enumerate(groups)
    }
    comparisons = []
    for reference, candidate in itertools.combinations(groups, 2):
        a, b = groups.index(reference), groups.index(candidate)
        interval = (
            np.percentile(bootstrap[b] - bootstrap[a], [2.5, 97.5]).tolist()
            if bootstrap is not None
            else None
        )
        paired = {row["case_id"]: row for row in selected if row["group"] == reference}
        candidate_rows = [row for row in selected if row["group"] == candidate]
        comparisons.append(
            {
                "reference": reference,
                "candidate": candidate,
                "n_paired_cases": len(paired),
                "n_base_clusters": len(base_ids),
                "candidate_minus_reference_pp": (
                    estimates[candidate] - estimates[reference] if paired else None
                ),
                "paired_cluster_bootstrap_95pct_pp": interval,
                "reference_correct_candidate_wrong": sum(
                    paired[row["case_id"]]["exact_match"] and not row["exact_match"]
                    for row in candidate_rows
                ),
                "reference_wrong_candidate_correct": sum(
                    not paired[row["case_id"]]["exact_match"] and row["exact_match"]
                    for row in candidate_rows
                ),
            }
        )
    return {
        "selection_rule": "entire capacity/depth cells complete in every group",
        "complete_common_conditions": [
            {"capacity": capacity, "depth": depth} for capacity, depth in common
        ],
        "n_base_clusters": len(base_ids),
        "base_ids": base_ids,
        "bootstrap_seed": seed,
        "bootstrap_samples": samples,
        "common_condition_exact_accuracy_pct": estimates,
        "comparisons": comparisons,
        "limitation": (
            "Resampling base IDs preserves dependence across lengths/depths. "
            "Ten base cases is a small cluster sample; no independent-device or "
            "unmeasured-context generalization follows. Incomplete conditions "
            "are explicitly omitted from paired estimates, not scored as wrong."
        ),
    }


def aggregate_records(
    records: list[dict[str, Any]],
    manifest: dict[str, Any] | list[dict[str, Any]] | None = None,
    *,
    groups: tuple[str, ...] = GROUPS,
    capacities: tuple[int, ...] = CAPACITIES,
    depths: tuple[int, ...] = DEPTHS,
    split: str = "test",
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Aggregate explicit denominators and retain failed/missing planned cases."""
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    manifest_cases = (
        manifest.get("cases", []) if isinstance(manifest, dict) else manifest or []
    )
    prototypes = {}
    for source in manifest_cases:
        if source.get("split", "test") != split:
            continue
        case = _case(source)
        if case["case_id"] in prototypes:
            raise ValueError(f"duplicate manifest case {case['case_id']}")
        prototypes[case["case_id"]] = case
    observed = {}
    for source in records:
        if source.get("split", "test") != split:
            continue
        group = source.get("group")
        if group not in groups:
            raise ValueError(f"unknown group {group!r}")
        identifier = str(source.get("case_id"))
        known = prototypes.get(identifier, {})
        case = _case({**known, **source})
        if manifest is not None and identifier not in prototypes:
            raise ValueError(f"result is not in selected manifest: {identifier}")
        if case["capacity"] not in capacities or case["depth"] not in depths:
            raise ValueError(f"result outside predeclared conditions: {identifier}")
        for field in (
            "base_id",
            "expected_answer",
            "capacity",
            "depth",
            "input_tokens",
        ):
            if field in known and field in case and known[field] != case[field]:
                raise ValueError(f"input contract mismatch {identifier}: {field}")
        key = group, identifier
        if key in observed:
            raise ValueError(f"duplicate result: {key}; do not silently pick a retry")
        observed[key] = case
        if not known:
            prototypes[identifier] = {
                key: value
                for key, value in case.items()
                if key not in ("group", "response", "status", "error")
            }
    normalized = []
    for group in groups:
        for identifier, prototype in prototypes.items():
            if (
                prototype["capacity"] not in capacities
                or prototype["depth"] not in depths
            ):
                raise ValueError(f"manifest outside planned conditions: {identifier}")
            source = observed.get((group, identifier))
            row = {
                **prototype,
                **(source or {}),
                "group": group,
                "status": source.get("status", "missing") if source else "missing",
                "response": source.get("response", "") if source else "",
            }
            if row["status"] in SUCCESS_STATUSES:
                if source is None or "response" not in source:
                    raise ValueError(
                        f"completed result missing response: {group}/{identifier}"
                    )
                row.update(score_answer(row["response"], row["expected_answer"]))
                row["output_truncated"] = _truncated(row)
            else:
                row.update(
                    dict.fromkeys(
                        (
                            "exact_match",
                            "extracted_match",
                            "empty_response",
                            "format_issue",
                            "candidate_count",
                            "output_truncated",
                        )
                    )
                )
                row["candidates"] = []
            normalized.append(row)
    normalized.sort(
        key=lambda row: (
            groups.index(row["group"]),
            row["capacity"],
            row["depth"],
            row["case_id"],
        )
    )
    cells = [
        {
            "group": group,
            "capacity": capacity,
            "depth": depth,
            **_statistics(
                [
                    row
                    for row in normalized
                    if (row["group"], row["capacity"], row["depth"])
                    == (group, capacity, depth)
                ]
            ),
        }
        for group, capacity, depth in itertools.product(groups, capacities, depths)
    ]
    summary = {
        group: _statistics([row for row in normalized if row["group"] == group])
        for group in groups
    }
    complete = bool(normalized) and all(cell["complete"] for cell in cells)
    return {
        "status": "complete" if complete else "incomplete",
        "split": split,
        "groups": list(groups),
        "capacities": list(capacities),
        "depths": list(depths),
        "manifest_supplied": manifest is not None,
        "cases": normalized,
        "cells": cells,
        "summary": summary,
        "paired": _cluster_comparisons(
            normalized, cells, groups, bootstrap_samples, bootstrap_seed
        ),
        "limitations": [
            "This is a single-needle six-digit retrieval evaluation, not a proof of general long-context reasoning.",
            "No PPL or throughput comparison is inferred from retrieval accuracy.",
            "Per-group accuracy denominators include completed model generations only; infrastructure errors, unsupported, and missing cases are reported separately.",
            "An incomplete report does not establish the full planned context range. Unmeasured contexts are not zero-accuracy observations.",
            "K4/V4 versus Boundary changes K/V allocation as well as layer placement; Structured versus LM changes quantizer and decoder together.",
        ],
    }


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def plot_heatmaps(data: dict[str, Any], destination: Path) -> None:
    """Incomplete cells stay gray; all panels share the full planned axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups, capacities, depths = data["groups"], data["capacities"], data["depths"]
    cells = {(c["group"], c["capacity"], c["depth"]): c for c in data["cells"]}
    labels = []
    for capacity in capacities:
        inputs = sorted(
            {
                int(row["input_tokens"])
                for row in data["cases"]
                if row["capacity"] == capacity and row.get("input_tokens") is not None
            }
        )
        actual = (
            (str(inputs[0]) if len(inputs) == 1 else f"{inputs[0]}-{inputs[-1]}")
            if inputs
            else "unknown"
        )
        labels.append(f"{capacity // 1024}K\nin {actual}")
    colourmap = plt.get_cmap("RdYlGn").with_extremes(bad="#d0d0d0")
    figure, axes = plt.subplots(
        1,
        len(groups),
        figsize=(5.3 * len(groups), 5.3),
        squeeze=False,
        constrained_layout=True,
    )
    for group, axis in zip(groups, axes[0], strict=True):
        values = np.full((len(depths), len(capacities)), np.nan)
        for y, depth in enumerate(depths):
            for x, capacity in enumerate(capacities):
                cell = cells[(group, capacity, depth)]
                if cell["complete"]:
                    values[y, x] = cell["exact_accuracy_pct"]
                    label = f"{cell['exact_accuracy_pct']:.0f}%\n{cell['exact_successes']}/{cell['n_completed']}"
                elif cell["n_unsupported"] and not cell["n_completed"]:
                    label = f"unsupported\n0/{cell['n_expected']}"
                else:
                    label = f"incomplete\n{cell['n_completed']}/{cell['n_expected']}"
                axis.text(x, y, label, ha="center", va="center", fontsize=8)
        im = axis.imshow(
            np.ma.masked_invalid(values),
            cmap=colourmap,
            vmin=0,
            vmax=100,
            aspect="auto",
        )
        axis.set_xticks(range(len(capacities)), labels, fontsize=8)
        axis.set_yticks(range(len(depths)), [f"{depth}%" for depth in depths])
        axis.set_title(GROUP_LABELS.get(group, group))
        axis.set_xlabel("Capacity / actual prompt tokens")
        axis.set_ylabel("Needle depth in background")
    figure.colorbar(im, ax=list(axes[0]), label="Exact-match accuracy (%)", shrink=0.8)
    figure.suptitle(
        f"NIAH — {data['split']} — {data['status']}\nGray = incomplete/unavailable, not 0% accuracy"
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"niah_accuracy.{suffix}", dpi=180)
    plt.close(figure)
    if not {"boundary_lm", "boundary_structured"}.issubset(groups):
        return
    values = np.full((len(depths), len(capacities)), np.nan)
    figure, axis = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
    for y, depth in enumerate(depths):
        for x, capacity in enumerate(capacities):
            lm, structured = [
                cells[(group, capacity, depth)]
                for group in ("boundary_lm", "boundary_structured")
            ]
            if lm["complete"] and structured["complete"]:
                values[y, x] = (
                    structured["exact_accuracy_pct"] - lm["exact_accuracy_pct"]
                )
                label = f"{values[y, x]:+.0f} pp"
            else:
                label = "unavailable"
            axis.text(x, y, label, ha="center", va="center", fontsize=9)
    divergent = plt.get_cmap("RdBu").with_extremes(bad="#d0d0d0")
    im = axis.imshow(
        np.ma.masked_invalid(values), cmap=divergent, vmin=-100, vmax=100, aspect="auto"
    )
    axis.set_xticks(range(len(capacities)), labels)
    axis.set_yticks(range(len(depths)), [f"{depth}%" for depth in depths])
    axis.set_xlabel("Capacity / actual prompt tokens")
    axis.set_ylabel("Needle depth in background")
    axis.set_title("Boundary Structured - converged LM\nMatched complete cells only")
    figure.colorbar(im, ax=axis, label="Exact-match accuracy difference (pp)")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"niah_structured_minus_lm.{suffix}", dpi=180)
    plt.close(figure)


def write_report(
    data: dict[str, Any], destination: Path, *, plots: bool = True
) -> None:
    """Use a fresh directory; never overwrite an earlier experiment report."""
    destination.mkdir(parents=True, exist_ok=False)
    _csv(destination / "cases.csv", data["cases"])
    _csv(destination / "cells.csv", data["cells"])
    _csv(
        destination / "summary.csv",
        [{"group": group, **row} for group, row in data["summary"].items()],
    )
    _csv(destination / "paired_comparisons.csv", data["paired"]["comparisons"])
    with (destination / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    lines = [
        f"NIAH retrieval report — {data['status']} — split={data['split']}",
        "Primary metric: exact match after leading/trailing whitespace removal only.",
        "Secondary: exactly one standalone six-digit candidate, matching the answer.",
        "Repeated identical candidates count as multiple; leading zeros are preserved.",
        "Infrastructure errors are not model retrieval failures or successes.",
        "Group | completed/planned | exact successes | exact accuracy | extraction accuracy",
    ]
    for group, row in data["summary"].items():
        exact = (
            f"{row['exact_accuracy_pct']:.2f}%"
            if row["n_completed"]
            else "not measured"
        )
        extracted = (
            f"{row['extracted_accuracy_pct']:.2f}%"
            if row["n_completed"]
            else "not measured"
        )
        lines.append(
            f"{group} | {row['n_completed']}/{row['n_expected']} | {row['exact_successes']} | {exact} | {extracted}"
        )
        lines.append("  statuses: " + json.dumps(row["status_counts"], sort_keys=True))
    lines += [
        "",
        "Paired estimates use complete capacity/depth conditions common to all groups.",
        json.dumps(data["paired"], indent=2, ensure_ascii=False),
        "",
        "LIMITATIONS",
        *data["limitations"],
        "Failed/missing/unsupported records remain in cases.csv with explicit status.",
        "A partial gray heatmap is a progress/blocked-state artifact, not completed full-context evaluation.",
    ]
    with (destination / "report.txt").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    if plots:
        plot_heatmaps(data, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest
    if manifest_path is None:
        for name in ("data/data_manifest.json", "data_manifest.json", "manifest.json"):
            candidate = args.root / name
            if candidate.is_file():
                manifest_path = candidate
                break
    if manifest_path is None:
        parser.error("A fixed manifest is required to count planned/missing cases")
    records = []
    for path in sorted((args.root / "results").glob("*.json")):
        value = json.loads(path.read_text())
        if isinstance(value, list):
            records.extend(value)
        elif "records" in value:
            records.extend(value["records"])
        else:
            records.append(value)
    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("cases", []) if isinstance(manifest, dict) else manifest
    declared_capacities = (
        manifest.get("capacities", manifest.get("protocol", {}).get("capacities"))
        if isinstance(manifest, dict)
        else None
    )
    capacities = (
        tuple(sorted(declared_capacities or {int(case["capacity"]) for case in cases}))
        or CAPACITIES
    )
    data = aggregate_records(records, manifest, split=args.split, capacities=capacities)
    write_report(data, args.output_dir or args.root / "report", plots=not args.no_plots)
    print(json.dumps({"status": data["status"], "summary": data["summary"]}, indent=2))


if __name__ == "__main__":
    main()
