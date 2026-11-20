# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Report the frozen three-model W4A16 KV budget comparison on WikiText."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
from pathlib import Path

from cross_model_plans import MODELS, PLACEMENTS
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

MODEL_NAMES = {
    "qwen3_1_7b": "Qwen3-1.7B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
}


def load_scores(root: Path) -> dict[tuple[str, str], dict]:
    """Match every measured row to one immutable model-specific plan."""
    summary = json.loads((root / "offline_comparison.json").read_bytes())
    files = json.loads((root / "selection_protocol.json").read_bytes())[
        "plan_files_sha256"
    ]
    expected_ids = ["k4_v4"] + [
        f"k{k}_v{8 - k}_{placement}" for k in (5, 6) for placement in PLACEMENTS
    ]
    expected = [(model, plan) for model in MODELS for plan in expected_ids]
    scores = {(r["model_id"], r["plan_id"]): r for r in summary["rows"]}
    if len(scores) != len(summary["rows"]) or set(scores) != set(expected):
        raise ValueError("Incomplete, duplicate, or unexpected offline comparison rows")
    if set(files) != {f"plans/{model}/{plan}.json" for model, plan in expected}:
        raise ValueError("Plan manifest differs from the requested 33-condition cohort")
    verified = {}
    for model, plan_id in expected:
        filename = f"plans/{model}/{plan_id}.json"
        data = (root / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != files[filename]:
            raise ValueError(f"Frozen plan hash changed: {filename}")
        plan = json.loads(data)
        row = scores[model, plan_id]
        if any(
            plan[field] != row[field]
            for field in ("model_id", "plan_id", "config_hash")
        ):
            raise ValueError(f"Score identity differs from frozen plan: {filename}")
        if (
            row["scored_tokens"] != 4092
            or row["cache_format_bytes_cl1024"]
            != 1024 * plan["cache_bytes_per_token"]["total"]
            or not math.isfinite(row["mean_nll"])
            or row["mean_nll"] < 0
            or not (math.isfinite(row["ppl"]) and row["ppl"] >= 1)
            or not math.isclose(
                math.log(row["ppl"]), row["mean_nll"], rel_tol=1e-10, abs_tol=1e-12
            )
        ):
            raise ValueError(
                f"Inconsistent metrics, target count, or KV cache storage: {filename}"
            )
        verified[model, plan_id] = {
            "model_id": model,
            "plan_id": plan_id,
            "placement": plan["placement"],
            "key_average_bits": plan["key_average_bits"],
            "value_average_bits": plan["value_average_bits"],
            "scored_tokens": row["scored_tokens"],
            "mean_nll": row["mean_nll"],
            "ppl": row["ppl"],
            "cache_format_bytes_cl1024": row["cache_format_bytes_cl1024"],
            "config_hash": row["config_hash"],
        }
    return verified


def paired_rows(scores: dict[tuple[str, str], dict]) -> list[dict]:
    rows = []
    for model in MODELS:
        base = scores[model, "k4_v4"]
        for placement in PLACEMENTS:
            k5 = scores[model, f"k5_v3_{placement}"]
            k6 = scores[model, f"k6_v2_{placement}"]
            rows.append(
                {
                    "model_id": model,
                    "placement": placement,
                    "k5_minus_k6_nll": k5["mean_nll"] - k6["mean_nll"],
                    "k5_minus_k4_nll": k5["mean_nll"] - base["mean_nll"],
                    "k6_minus_k4_nll": k6["mean_nll"] - base["mean_nll"],
                }
            )
    return rows


def budget_heatmap(scores: dict[tuple[str, str], dict], pairs: list[dict]) -> Figure:
    """Plot both alternative budgets on a shared NLL scale across all three models."""
    keys = ("k5_minus_k4_nll", "k6_minus_k4_nll")
    largest = max(abs(pair[key]) for pair in pairs for key in keys)
    limit = max(largest, 1e-4)
    ticks = MaxNLocator(nbins=5, symmetric=True).tick_values(-limit, limit)
    norm = Normalize(vmin=ticks[0], vmax=ticks[-1])
    colors = colormaps["RdBu_r"]

    figure = Figure(figsize=(7.25, 5.8), layout="constrained")
    FigureCanvasAgg(figure)
    layout = figure.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.16])
    figure.suptitle(
        "Prediction quality by KV budget and layer placement",
        weight="semibold",
        fontsize=14,
    )
    for panel, model in enumerate(MODELS):
        axis = figure.add_subplot(layout[panel])
        model_pairs = {
            pair["placement"]: pair for pair in pairs if pair["model_id"] == model
        }
        values = [
            [model_pairs[placement][key] for placement in PLACEMENTS] for key in keys
        ]
        image = axis.imshow(
            values, aspect="auto", cmap=colors, norm=norm, interpolation="nearest"
        )
        baseline = scores[model, "k4_v4"]["ppl"]
        axis.set_title(
            f"{MODEL_NAMES[model]}    |    K4/V4 baseline PPL {baseline:.2f}",
            loc="left",
            fontsize=10,
            fontweight="semibold",
        )
        axis.set_yticks([0, 1], labels=["K5/V3", "K6/V2"])
        axis.set_xticks(
            range(len(PLACEMENTS)), labels=[p.capitalize() for p in PLACEMENTS]
        )
        axis.set_xticks([i + 0.5 for i in range(len(PLACEMENTS) - 1)], minor=True)
        axis.set_yticks([0.5], minor=True)
        axis.grid(which="minor", color="white", linewidth=0.7)
        axis.tick_params(which="both", length=0, labelsize=9)
        axis.spines[:].set_visible(False)
        for row, numbers in enumerate(values):
            for column, value in enumerate(numbers):
                r, g, b, _ = colors(norm(value))
                lightness = 0.2126 * r + 0.7152 * g + 0.0722 * b
                axis.text(
                    column,
                    row,
                    f"{value:+.4f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black" if lightness >= 0.55 else "white",
                )
    legend = figure.colorbar(
        image, cax=figure.add_subplot(layout[-1]), orientation="horizontal", ticks=ticks
    )
    legend.outline.set_visible(False)
    legend.ax.tick_params(length=2, labelsize=8)
    legend.set_label(
        "Mean NLL difference from K4/V4 (nats per target)\n"
        "Blue indicates lower loss and red indicates higher loss.",
        fontsize=9,
    )
    return figure


def write_report(root: Path, out: Path) -> None:
    scores = load_scores(root)
    pairs = paired_rows(scores)
    k5_counts = []
    for model in MODELS:
        placements = [p for p in pairs if p["model_id"] == model]
        better = sum(
            p["k5_minus_k6_nll"] < 0 and p["k5_minus_k4_nll"] < 0 for p in placements
        )
        k5_counts.append(
            f"{better}/{len(placements)} placements for {MODEL_NAMES[model]}"
        )
    lines = [
        "# Qwen3 offline KV budget comparison",
        "",
        "All 33 configurations were scored on the same four historical 1,024-token WikiText windows "
        "(4,092 next-token targets per configuration). Mean NLL is weighted by the number of targets. "
        "PPL is its exponential. Lower values indicate better prediction on these windows.",
        "",
        "Each budget uses eight payload bits per K/V coordinate pair on average. K4 uses Lloyd-Max, "
        "K8 uses the two-region piecewise-uniform quantizer, and V2 and V4 use Lloyd-Max. "
        "The cache sizes below include the packed indices and effective FP16 scales at CL1024. "
        "The measurements come from offline W4A16 replay. Device throughput and latency were unmeasured.",
        "",
        "## Lowest observed NLL",
        "",
        "These minima describe the four scored windows. Every mask was fixed before scoring, "
        "and none was tuned or selected for deployment on these results.",
        "",
        "| Model | Plan | Mean NLL | PPL | CL1024 cache (MiB) |",
        "|---|---|---:|---:|---:|",
    ]
    for model in MODELS:
        group = [r for (m, _), r in scores.items() if m == model]
        best = min(r["mean_nll"] for r in group)
        for r in group:
            if r["mean_nll"] == best:
                lines.append(
                    f"| {MODEL_NAMES[model]} | `{r['plan_id']}` | {best:.6f} | {r['ppl']:.5f} | {r['cache_format_bytes_cl1024'] / (1 << 20):.3f} |"
                )
    lines += [
        "",
        "## Matched budget differences",
        "",
        "Differences are in nats per target. A negative difference means the first budget has lower "
        "mean NLL on the matched windows. K4/V4 uses one mask for every placement. K5/V3 and K6/V2 "
        "apply the named placement separately to their K8 and V4 quotas, so their higher-precision "
        "layers can overlap. Independent choices for K and V placement remain untested.",
        "",
        "K5/V3 has lower mean NLL than both alternatives in "
        + ", ".join(k5_counts)
        + ".",
        "",
        "![Mean NLL differences from each model's K4/V4 baseline by budget and placement](offline_budget_heatmap.png)",
        "",
        "| Model | Placement | K5/V3 - K6/V2 | K5/V3 - K4/V4 | K6/V2 - K4/V4 |",
        "|---|---|---:|---:|---:|",
    ]
    for pair in pairs:
        lines.append(
            f"| {MODEL_NAMES[pair['model_id']]} | {pair['placement']} | "
            f"{pair['k5_minus_k6_nll']:+.6f} | {pair['k5_minus_k4_nll']:+.6f} | {pair['k6_minus_k4_nll']:+.6f} |"
        )
    lines += [
        "",
        "## All configurations",
        "",
        "| Model | Plan | Mean NLL | PPL |",
        "|---|---|---:|---:|",
    ]
    for row in scores.values():
        lines.append(
            f"| {MODEL_NAMES[row['model_id']]} | `{row['plan_id']}` | {row['mean_nll']:.6f} | {row['ppl']:.5f} |"
        )
    lines += [
        "",
        "The score table includes full-precision metrics and the codec configuration hashes. "
        "[All scores](offline_scores.csv) · [Paired differences](paired_nll.csv)",
        "",
        "## Reproduce the comparison",
        "",
        "Run from the repository root with the archived split graphs, weights, calibration grids "
        "and token files available. The scoring command validates the frozen cohort, resumes "
        "completed windows, and writes the aggregate results under the experiment archive.",
        "",
        "```bash",
        f"run_root={shlex.quote(str(root.resolve()))}",
        f"report_dir={shlex.quote(str(out))}",
        "export PYTHONPATH=src OPENBLAS_NUM_THREADS=1",
        'venv/bin/python scripts/llm/turboquant/compare_cross_model_budgets.py --root "$run_root"',
        'venv/bin/python scripts/llm/turboquant/report_cross_model_budgets.py --root "$run_root" --out "$report_dir"',
        "```",
        "",
        f"Frozen selection protocol SHA-256 `{hashlib.sha256((root / 'selection_protocol.json').read_bytes()).hexdigest()}`. "
        f"Offline score file SHA-256 `{hashlib.sha256((root / 'offline_comparison.json').read_bytes()).hexdigest()}`.",
        "",
    ]
    out.mkdir(parents=True, exist_ok=True)
    budget_heatmap(scores, pairs).savefig(out / "offline_budget_heatmap.png", dpi=300)
    for filename, records in (
        ("offline_scores.csv", list(scores.values())),
        ("paired_nll.csv", pairs),
    ):
        with (out / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (out / "offline_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True, type=Path, help="Frozen experiment root"
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Report output directory"
    )
    args = parser.parse_args()
    write_report(args.root, args.out)


if __name__ == "__main__":
    main()
