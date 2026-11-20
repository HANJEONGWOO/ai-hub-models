# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Report same-memory KV refinements against each model's historical Early anchor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shlex
from pathlib import Path

from model_identity import QWEN_SHAPES

MODEL_NAMES = {
    "qwen3_1_7b": "Qwen3-1.7B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
}
OUT = Path(__file__).parent / "experiments" / "cross_model_pu_refinement"
BOOTSTRAP_SEED = 20261120
BOOTSTRAP_REPS = 10_000
FAMILY_NAMES = {
    "anchor": "Early anchor",
    "budget_only": "Budget only",
    "placement_only": "Placement only",
    "budget_and_placement": "Budget and placement",
}


def paired_interval(deltas: list[float]) -> tuple[float, float]:
    """Percentile interval for the equal-document mean, resampling complete pairs."""
    if len(set(deltas)) == 1:
        return deltas[0], deltas[0]
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(
        sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
        for _ in range(BOOTSTRAP_REPS)
    )

    def percentile(p: float) -> float:
        rank = (len(means) - 1) * p
        low = math.floor(rank)
        return means[low] + (means[math.ceil(rank)] - means[low]) * (rank - low)

    return percentile(0.025), percentile(0.975)


def score_rows(model: str, group: dict) -> list[dict]:
    layers, _, _, kv_heads, head_dim = QWEN_SHAPES[model]
    anchor_id = group["anchor"]
    selected_id = group["selected"]
    calibration = {r["plan_id"]: r for r in group["candidates"]}
    expected_choice = min(
        calibration, key=lambda p: (calibration[p]["nll"], p != anchor_id, p)
    )
    if selected_id != expected_choice:
        raise ValueError(f"{model} selection differs from the calibration tie rule")
    controls = group.get("category_controls", {})
    for category, finalist in controls.items():
        members = [r for r in calibration.values() if r["category"] == category]
        best = min(members, key=lambda r: (r["nll"], r["plan_id"]))
        if best["plan_id"] != finalist:
            raise ValueError(
                f"{model}/{category} finalist differs from calibration winner"
            )
    prefixes = {
        (len(r["k8_layers"]), len(r["v4_layers"])): r["plan_id"]
        for r in calibration.values()
        if all(
            sorted(r[k]) == list(range(len(r[k]))) for k in ("k8_layers", "v4_layers")
        )
    }
    rows = []
    for split, scores in (
        ("calibration", group["candidates"]),
        ("heldout", group["heldout"]),
    ):
        plans = {r["plan_id"]: r for r in scores}
        if (
            len(plans) != len(scores)
            or not {anchor_id, selected_id, *controls.values()} <= plans.keys()
        ):
            raise ValueError(
                f"{model}/{split} has duplicate plans or missing anchor/selection"
            )
        anchor = plans[anchor_id]
        doc_ids = sorted(anchor["documents"])
        if len(doc_ids) < 2:
            raise ValueError(f"{model}/{split} requires multiple paired documents")
        if (
            split == "heldout"
            and set(doc_ids) & calibration[anchor_id]["documents"].keys()
        ):
            raise ValueError(f"{model} calibration and held-out document IDs overlap")
        for plan_id, score in plans.items():
            if plan_id not in calibration or any(
                score[field] != calibration[plan_id][field]
                for field in ("category", "k8_layers", "v4_layers", "config_hash")
            ):
                raise ValueError(f"{model}/{plan_id} changed across splits")
            if sorted(score["documents"]) != doc_ids:
                raise ValueError(f"{model}/{split}/{plan_id} has unmatched documents")
            nll, ppl = score["nll"], score["ppl"]
            doc_nll = {d: score["documents"][d] for d in doc_ids}
            if not (
                all(math.isfinite(v) and v >= 0 for v in [nll, *doc_nll.values()])
                and math.isfinite(ppl)
                and ppl >= 1
                and math.isclose(math.log(ppl), nll, rel_tol=1e-6, abs_tol=1e-8)
                and math.isclose(
                    nll,
                    sum(doc_nll.values()) / len(doc_nll),
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                )
            ):
                raise ValueError(f"Invalid metrics for {model}/{split}/{plan_id}")
            k8, v4 = score["k8_layers"], score["v4_layers"]
            for mask in (k8, v4):
                if len(set(mask)) != len(mask) or any(
                    type(i) is not int or not 0 <= i < layers for i in mask
                ):
                    raise ValueError(f"Invalid layer mask for {model}/{plan_id}")
            if 2 * len(k8) + len(v4) != layers:
                raise ValueError(
                    f"{model}/{plan_id} does not average eight payload bits"
                )
            prefix_id = prefixes[len(k8), len(v4)]
            control = plans.get(prefix_id)
            moved = [
                kind
                for kind in ("k8", "v4")
                if set(score[kind + "_layers"])
                != set(calibration[prefix_id][kind + "_layers"])
            ]
            if len(moved) > 1:
                raise ValueError(
                    f"Expected at most one placement swap for {model}/{plan_id}"
                )
            paired = {d: doc_nll[d] - anchor["documents"][d] for d in doc_ids}
            lo, hi = paired_interval(list(paired.values()))
            rows.append(
                dict(
                    model_id=model,
                    split=split,
                    plan_id=plan_id,
                    category=score["category"],
                    anchor=plan_id == anchor_id,
                    calibration_choice=plan_id == selected_id,
                    category_control=controls.get(score["category"]) == plan_id,
                    config_hash=score["config_hash"],
                    placement_control=prefix_id,
                    moved_mask=moved[0] if moved else "",
                    k8_layers=sorted(k8),
                    v4_layers=sorted(v4),
                    average_key_bits=4 + 4 * len(k8) / layers,
                    average_value_bits=2 + 2 * len(v4) / layers,
                    cache_format_bytes_cl1024=1024 * layers * kv_heads * (head_dim + 4),
                    mean_nll=nll,
                    scored_targets=len(doc_ids) * 1023,
                    ppl=ppl,
                    nll_minus_anchor=nll - anchor["nll"],
                    prefix_nll_minus_anchor=control["nll"] - anchor["nll"]
                    if control
                    else None,
                    nll_minus_prefix=nll - control["nll"] if control else None,
                    ppl_minus_anchor=ppl - anchor["ppl"],
                    document_mean_nll_minus_anchor=sum(paired.values()) / len(paired),
                    document_bootstrap_p025=lo,
                    document_bootstrap_p975=hi,
                    documents_lower=sum(v < 0 for v in paired.values()),
                    documents_tied=sum(v == 0 for v in paired.values()),
                    documents_higher=sum(v > 0 for v in paired.values()),
                    document_nll=doc_nll,
                    document_nll_minus_anchor=paired,
                )
            )
    return rows


def compact_layers(mask: list[int]) -> str:
    """Display zero-based inclusive ranges and single layers."""
    runs = []
    for i in sorted(mask):
        if runs and runs[-1][-1] == i - 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return (
        ", ".join(str(r[0]) if len(r) == 1 else f"{r[0]}-{r[-1]}" for r in runs) or "∅"
    )


def mask_change(old: list[int], new: list[int]) -> str:
    added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
    parts = []
    if added:
        parts.append(f"add {compact_layers(added)}")
    if removed:
        parts.append(f"remove {compact_layers(removed)}")
    return ", ".join(parts) or "same"


def write_report(root: Path, out: Path) -> None:
    source = (root / "summary.json").read_bytes()
    summary = json.loads(source)
    models = summary["models"]
    if set(models) != MODEL_NAMES.keys():
        raise ValueError(f"Expected all three models, got {sorted(models)}")
    rows = [row for m in MODEL_NAMES for row in score_rows(m, models[m])]
    calibration = [r for r in rows if r["split"] == "calibration"]
    heldout = [r for r in rows if r["split"] == "heldout"]
    selected = [r for r in heldout if r["calibration_choice"]]
    anchors = {r["model_id"]: r for r in heldout if r["anchor"]}
    cal_docs = set(calibration[0]["document_nll"])
    held_docs = set(heldout[0]["document_nll"])
    if any(
        set(r["document_nll"])
        != (cal_docs if r["split"] == "calibration" else held_docs)
        for r in rows
    ):
        raise ValueError("Document IDs differ between models on the same split")

    lines = [
        "# Same-memory K/V refinement on Qwen3",
        "",
        "Offline W4A16 replay compared local K/V allocations on WikiText-2. "
        f"Each model used {len(cal_docs)} documents ({len(cal_docs) * 1023:,} targets) for calibration and {len(held_docs)} separate documents "
        f"({len(held_docs) * 1023:,} targets) for held-out evaluation, with one 1,024-token window per document. "
        "The historical Early anchors use K5/V3 for Qwen3-1.7B and Qwen3-4B and K6/V2 for Qwen3-8B. "
        "[Historical placement comparison](../cross_model_pu_budget/offline_report.md)",
        "",
        "## Held-out performance of the calibration choice",
        "",
        "NLL is mean next-token negative log-likelihood and PPL is its exponential. Each difference is the chosen allocation minus its model's anchor on the same targets. "
        "Negative differences favor the chosen allocation. Document counts give lower, equal and higher NLL in that order.",
        "",
        "| Model | Chosen plan | Anchor PPL | Chosen PPL | Δ PPL | Δ NLL (nats/target) | Document Δ NLL, 95% interval | Lower / equal / higher |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in selected:
        a = anchors[r["model_id"]]
        lines.append(
            f"| {MODEL_NAMES[r['model_id']]} | `{r['plan_id']}` | {a['ppl']:.5f} | {r['ppl']:.5f} | "
            f"{r['ppl_minus_anchor']:+.5f} | {r['nll_minus_anchor']:+.6f} | "
            f"[{r['document_bootstrap_p025']:+.6f}, {r['document_bootstrap_p975']:+.6f}] | "
            f"{r['documents_lower']} / {r['documents_tied']} / {r['documents_higher']} |"
        )
    lines.append("")
    for r in selected:
        name = MODEL_NAMES[r["model_id"]]
        if r["anchor"]:
            lines.append(
                f"- {name} retained its anchor on calibration. The zero difference compares that allocation with itself."
            )
            continue
        direction = (
            "lower"
            if r["nll_minus_anchor"] < 0
            else "higher"
            if r["nll_minus_anchor"] > 0
            else "equal"
        )
        interval = (
            "falls below zero"
            if r["document_bootstrap_p975"] < 0
            else "falls above zero"
            if r["document_bootstrap_p025"] > 0
            else "includes zero"
        )
        lines.append(
            f"- {name} had {direction} held-out NLL with the chosen allocation. The document interval {interval}."
        )
    lines += [
        "",
        f"Intervals summarize variation across these WikiText-2 documents. Each of {BOOTSTRAP_REPS:,} bootstrap replicates "
        f"resamples complete anchor/candidate document pairs with replacement and averages their NLL differences with equal document weight "
        f"(seed {BOOTSTRAP_SEED}, 2.5th and 97.5th percentiles). The small sample limits precision. "
        "The comparisons address the measured local moves, model checkpoints and corpus. A budget exchange changes both K and V, so it measures their joint effect.",
        "",
        "## Chosen allocations",
        "",
        "Layer indices start at zero and ranges include both endpoints. Listed key layers use piecewise-uniform K8 and all other key layers use Lloyd-Max K4. "
        "Listed value layers use Lloyd-Max V4 and all other value layers use Lloyd-Max V2. "
        "All rows average eight payload bits per K/V coordinate pair within the model. "
        "The CL1024 format size includes the packed indices and one FP16 scale for each K and V head vector.",
        "",
        "| Model | Mean K / V bits | CL1024 MiB | K8 layers | V4 layers | Change to K8 from anchor | Change to V4 from anchor |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for r in selected:
        a = anchors[r["model_id"]]
        lines.append(
            f"| {MODEL_NAMES[r['model_id']]} | {r['average_key_bits']:.3f} / {r['average_value_bits']:.3f} | "
            f"{r['cache_format_bytes_cl1024'] / (1 << 20):.3f} | {compact_layers(r['k8_layers'])} | {compact_layers(r['v4_layers'])} | "
            f"{mask_change(a['k8_layers'], r['k8_layers'])} | {mask_change(a['v4_layers'], r['v4_layers'])} |"
        )
    if any(r["category_control"] for r in heldout):
        lines += [
            "",
            "## Held-out controls",
            "",
            "The best budget-only candidate and the best placement-only candidate on calibration were also frozen for each model. "
            "Their held-out comparisons with the anchor are exploratory. A candidate that also won overall appears in both tables.",
            "",
            "| Model | Control family | Frozen plan | Δ NLL | Δ PPL | Lower / equal / higher |",
            "|---|---|---|---:|---:|---:|",
        ]
        for r in heldout:
            if r["category_control"]:
                lines.append(
                    f"| {MODEL_NAMES[r['model_id']]} | {FAMILY_NAMES[r['category']]} | `{r['plan_id']}` | "
                    f"{r['nll_minus_anchor']:+.6f} | {r['ppl_minus_anchor']:+.5f} | "
                    f"{r['documents_lower']} / {r['documents_tied']} / {r['documents_higher']} |"
                )
    lines += [
        "",
        "## Calibration search",
        "",
        "A budget move exchanges one K8 layer for two V4 layers or vice versa. A placement swap moves the higher precision "
        "between layers at a fixed K or V count. Each swap moves the last higher-precision layer in an Early prefix "
        "to the adjacent, middle or final lower-precision layer. Joint candidates vary both allocation counts and placement. "
        "The 8B anchor has zero V4 layers, so V placement first becomes testable after one budget exchange. "
        "The lowest calibration NLL fixes the model's choice before held-out scoring. Ties favor the anchor, then the lexicographically first plan ID.",
        "",
        "| Model | Candidate family | Plans scored | Lowest calibration NLL plan | Δ NLL vs anchor | Budget contribution | Placement contribution |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for model, name in MODEL_NAMES.items():
        group = [r for r in calibration if r["model_id"] == model]
        for family, moved in sorted({(r["category"], r["moved_mask"]) for r in group}):
            members = [
                r for r in group if (r["category"], r["moved_mask"]) == (family, moved)
            ]
            best = min(
                members, key=lambda r: (r["mean_nll"], not r["anchor"], r["plan_id"])
            )
            label = FAMILY_NAMES.get(family, family)
            if moved:
                label += f" (move {moved.upper()})"
            lines.append(
                f"| {name} | {label} | {len(members)} | `{best['plan_id']}` | "
                f"{best['nll_minus_anchor']:+.6f} | {best['prefix_nll_minus_anchor']:+.6f} | {best['nll_minus_prefix']:+.6f} |"
            )
    lines += [
        "",
        "The budget contribution compares the Early prefix at the candidate's quotas with the anchor. The placement contribution compares the candidate "
        "with that same prefix. Their sum is the candidate's calibration Δ NLL. The placement effect can depend on the K/V quotas and the layers exchanged.",
        "",
        f"The CSV contains all {len(calibration)} calibration scores and {len(heldout)} held-out scores, including absolute NLL, PPL, "
        "signed differences from the corresponding anchor, per-document scores, exact layer lists and configuration hashes. "
        "[Download all scores](scores.csv)",
        "",
    ]
    if "protocol_sha256" in summary:
        protocol_bytes = (root / "evaluation_protocol.json").read_bytes()
        protocol = json.loads(protocol_bytes)
        data_bytes = (root / "data_manifest.json").read_bytes()
        if (
            hashlib.sha256(protocol_bytes).hexdigest() != summary["protocol_sha256"]
            or hashlib.sha256(data_bytes).hexdigest() != protocol["data_sha256"]
            or hashlib.sha256(
                (root / "selection_protocol.json").read_bytes()
            ).hexdigest()
            != protocol["plans_sha256"]
        ):
            raise ValueError(
                "Archive protocol or dataset hash differs from the scored summary"
            )
        data = json.loads(data_bytes)
        lines += [
            "## Evaluation documents",
            "",
            f"The corpus is `{data['dataset']}` at revision `{data['revision']}`. Calibration uses the official training split. "
            "The held-out pool combines validation and test. The dataset builder excluded archived article titles and content hashes, "
            "screened for shared 16-token sequences against inventoried tokens and previously selected articles, and checked that the three tokenizers agree. "
            "These checks cover recorded local experiments. Model pretraining and original activation-calibration data membership are unknown.",
            "",
            "| Role | Official split | Window ID in CSV | Article | Token offset |",
            "|---|---|---|---|---:|",
        ]
        for role, expected_docs in (("calibration", cal_docs), ("heldout", held_docs)):
            windows = data["windows"][role]
            if {w["file"] for w in windows} != expected_docs:
                raise ValueError(
                    f"Dataset manifest and scored {role} document IDs differ"
                )
            for w in windows:
                title = w["title"].strip().strip("=").strip().replace("|", "\\|")
                lines.append(
                    f"| {role} | {w['official_split']} | `{w['file']}` | {title} | {w['offset']} |"
                )
        lines += [
            "",
            f"Dataset manifest SHA-256 `{protocol['data_sha256']}`. Candidate protocol SHA-256 `{protocol['plans_sha256']}`. "
            f"Evaluation protocol SHA-256 `{summary['protocol_sha256']}`.",
            "",
        ]
    lines += [
        "## Reproduce the experiment",
        "",
        "Run from the repository root with the archived source corpus, checkpoints, split graphs and historical scores available. "
        "The commands freeze candidates and documents, run W4A16 replay, verify raw scores and regenerate the report.",
        "",
        "```bash",
        f"run_root={shlex.quote(str(root.resolve()))}",
        "export PYTHONPATH=src",
        'venv/bin/python scripts/llm/turboquant/refinement_plans.py --out "$run_root"',
        'venv/bin/python scripts/llm/turboquant/refinement_data.py --root "$run_root"',
        'venv/bin/python scripts/llm/turboquant/run_kv_refinement.py --root "$run_root"',
        'venv/bin/python scripts/llm/turboquant/verify_kv_refinement.py --root "$run_root"',
        'python3 scripts/llm/turboquant/report_kv_refinement.py --root "$run_root"',
        "```",
        "",
        f"Input `summary.json` SHA-256 `{hashlib.sha256(source).hexdigest()}`.",
        "",
    ]
    out.mkdir(parents=True, exist_ok=True)
    plans = json.loads((root / "selection_protocol.json").read_bytes())[
        "plan_files_sha256"
    ]
    target = out / "selected_plans"
    target.mkdir(exist_ok=True)
    for r in selected:
        model_id = r["model_id"]
        name = f"plans/{model_id}/{r['plan_id']}.json"
        encoded = (root / name).read_bytes()
        if (
            hashlib.sha256(encoded).hexdigest() != plans[name]
            or json.loads(encoded)["config_hash"] != r["config_hash"]
        ):
            raise ValueError(f"Selected configuration mismatch: {name}")
        (target / f"{model_id}.json").write_bytes(encoded)
    lines += [
        "",
        "Calibration-selected allocation files: "
        + " · ".join(
            f"[{MODEL_NAMES[r['model_id']]}](selected_plans/{r['model_id']}.json)"
            for r in selected
        )
        + ".",
        "",
    ]
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, separators=(",", ":"))
                    if isinstance(v, (dict, list))
                    else v
                    for k, v in row.items()
                }
            )
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Experiment archive containing summary.json",
    )
    parser.add_argument(
        "--out",
        default=OUT,
        type=Path,
        help="Output directory for README.md and scores.csv",
    )
    args = parser.parse_args()
    write_report(args.root, args.out)


if __name__ == "__main__":
    main()
