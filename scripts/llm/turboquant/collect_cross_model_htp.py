# SPDX-License-Identifier: BSD-3-Clause
"""Recompute the matched WikiText PPL rows from archived Galaxy S26 Ultra reports.

Each row uses four HTP score reports and one 897+128 generation report.
The archives have no PU K5/V3 runs. The 4B and 8B archives cover only
boundary and interleaved PU K6/V2 among the five fixed placements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from functools import lru_cache
from itertools import combinations
from pathlib import Path

from model_identity import sha256_file

MIXED = "hardware_aligned_mixed_kv_20261010"
ALLOCATION = "structured_k8_nll_allocation_20261013"
SCALING = "structured_k8_model_scaling_20261026"
ORIGINAL_ARCHIVE = Path("/mnt/d/ai-hub-models/binaries/turboquant")
# Published comparison digests from final_verification.json and final_report/scaling.json.
SUMMARIES = {
    MIXED: "e53463d25fc43c2a1e20d6bb02873c13355993876afd756440d620dc44d5eda1",
    ALLOCATION: "b84f6bb59e7046b2167d501d480a13a3bb16477169b1ca1a661eba047cb41250",
    f"{SCALING}/qwen3_4b": "94384ebef7c1cb8a17ada0d45622a9b872305b0c3646111e808fc6982f52e7c7",
    f"{SCALING}/qwen3_8b": "a41e1f1967048148646cafcadff23129e43dabfcd8bd275bb11d0b73610478e6",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


@lru_cache
def digest(path: Path) -> str:
    return sha256_file(path)


def check(condition: bool, context: object) -> None:
    if not condition:
        raise ValueError(f"Archive mismatch: {context}")


def active_codec(config: dict, layers: int) -> dict:
    """Compare active codebooks, rotations and per-layer widths across profile names."""
    cfg = dict(config)
    del cfg["profile"]
    plan = cfg.pop("layer_plan", {})
    cfg["bits"] = (
        plan.get("bits") or [[cfg["key"]["bits"], cfg["value"]["bits"]]] * layers
    )
    if all(pair == [4, 4] for pair in cfg["bits"]):
        # This joint profile has an unused V2 default and unused PU-K8 metadata.
        # Resolve the selected V4 codebook from its layer plan, retaining V rotation.
        cfg["value"] = {
            **cfg["value"],
            "bits": 4,
            "codebook_sha256": plan.get("codebook_sha256", {}).get(
                "4", cfg["value"]["codebook_sha256"]
            ),
        }
        cfg.pop("structured_k8", None)
        if cfg["scalar_indexing"] == "lloyd_tree_except_structured_k8_arithmetic":
            cfg["scalar_indexing"] = "lloyd_tree"
    else:
        cfg["active_codebooks"] = plan["codebook_sha256"]
        cfg["packing"] = plan["packing"]
    return cfg


def archive_groups(source: str) -> dict[str, str]:
    if source == MIXED:
        return {"k4_v4": "k4_v4"}
    if source == ALLOCATION:
        return {
            **{f"k6_v2_{p}": p for p in ("early", "late", "middle", "boundary")},
            "k6_v2_interleaved": "uniform_odd",
        }
    return {
        "k4_v4": "k4_v4",
        "k6_v2_boundary": "structured_boundary",
        "k6_v2_interleaved": "structured_odd",
    }


def collect(archive: Path, study: Path) -> list[dict]:
    plan_hashes = read(study / "selection_protocol.json")["plan_files_sha256"]
    baseline_assets = archive / "device_assets_cl1024"
    rows = []
    for source, summary_hash in SUMMARIES.items():
        root = archive / source
        large = source.startswith(SCALING)
        model = root.name if large else "qwen3_1_7b"
        summary_path = root / "comparison.json"
        check(digest(summary_path) == summary_hash, summary_path)
        groups = read(summary_path)["groups"]
        protocol = read(root / "protocol.json")
        split_path = (
            archive
            / Path(protocol["split"]).relative_to(ORIGINAL_ARCHIVE)
            / "split_manifest.json"
        )
        split_hash = digest(split_path)
        check(read(split_path)["model_id"] == model, split_path)
        offline = (
            root / "offline/graphs"
            if large
            else archive / "structured_k8_layer_search_v2_20261020/uniform_odd"
        )
        encodings = sorted(offline.glob("prompt_ar128_cl1024_*.encodings"))
        check(len(encodings) == len(read(split_path)["parts"]) - 1, offline)
        recorded = {}
        if source == ALLOCATION:
            recorded = read(root / "final_verification.json")[
                "measurement_reports_sha256"
            ]
        for plan_id, group in archive_groups(source).items():
            key = f"plans/{model}/{plan_id}.json"
            check(digest(study / key) == plan_hashes[key], key)
            plan = read(study / key)
            bundle = root / group
            for file in encodings:
                check(digest(file) == digest(bundle / file.name), bundle / file.name)
            convert = read(bundle / "convert_report.json")
            config_hash = hashlib.sha256(
                json.dumps(
                    convert["config"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            check(config_hash == convert["config_hash"], bundle)
            check(
                active_codec(convert["config"], plan["num_layers"])
                == active_codec(plan["config"], plan["num_layers"]),
                plan_id,
            )
            check(
                convert["context_buckets"] == [1024]
                and convert["attention_tile"] == 256
                and convert["rotated_attention"]
                and convert["quantize_current_kv"],
                bundle,
            )
            quality = groups[group]
            hashes = dict(recorded)
            if large:
                quality = quality["wikitext"]
                for document in quality["documents"]:
                    hashes[Path(document["report"]).name] = document["report_sha256"]
            elif source == MIXED:
                quality = quality["quality"]
            score_suffix = (
                "wikitext_w" if large else ("score_w" if source == MIXED else "test_")
            )
            perf_suffix = (
                "perf_once" if large else ("perf_long" if source == MIXED else "perf")
            )
            score_files = [
                root / "reports" / f"{group}_{score_suffix}{i}.json" for i in range(4)
            ]
            perf_path = root / "reports" / f"{group}_{perf_suffix}.json"
            expected_kv = plan["cache_bytes_per_token"]["total"] * 1024
            nll = 0.0
            for index, path in enumerate([*score_files, perf_path]):
                report = read(path)
                check(
                    report["device"]["soc_model"] == "SM8850"
                    and "/htp " in report["backend_api"]
                    and report["context_length"] == 1024
                    and report["kv_store_bytes"] == expected_kv
                    and report["config_hash"] == config_hash,
                    path,
                )
                if large:
                    check(report["model"]["model_id"] == model, path)
                    check(report["split_manifest_sha256"] == split_hash, path)
                if path.name in hashes:
                    check(digest(path) == hashes[path.name], path)
                if index < 4:
                    token_file = f"wikitext_w{index}.bin"
                    check(
                        report["mode"] == "score"
                        and report["scored_tokens"] == 1023
                        and report["prompt_tokens"] == 1024
                        and report["assets"]["tokens_file"] == token_file
                        and report["assets"]["tokens_sha256"]
                        == digest(baseline_assets / token_file),
                        path,
                    )
                    nll += report["nll_sum"]
            perf = read(perf_path)
            check(
                perf["mode"] == "generate"
                and perf["prompt_tokens"] == 897
                and len(perf["generated"]) == 128
                and len(perf["decode_step_s"]) == 127
                and len(perf["sessions"]) == 1
                and perf["burst_power"]
                and perf["assets"]["tokens_sha256"]
                == digest(baseline_assets / "boundary_prompt_cl1024.bin"),
                perf_path,
            )
            mean_nll = nll / 4092
            ppl = math.exp(mean_nll)
            check(math.isclose(ppl, quality["ppl"], rel_tol=1e-10), summary_path)
            rows.append(
                {
                    "model_id": model,
                    "plan_id": plan_id,
                    "scored_tokens": 4092,
                    "mean_nll": mean_nll,
                    "ppl": ppl,
                    "kv_store_bytes": expected_kv,
                    "ttft_ms": perf["ttft_s"] * 1000,
                    "prefill_tok_per_s": perf["prefill_tok_per_s"],
                    "decode_tok_per_s": perf["decode_tok_per_s"],
                    "timing_runs": 1,
                    "split_manifest_sha256": split_hash,
                    "plan_sha256": plan_hashes[key],
                    "htp_config_hash": config_hash,
                    "archive": source,
                    "group": group,
                    "comparison_sha256": summary_hash,
                    "score_files": " ".join(p.name for p in score_files),
                    "score_sha256": " ".join(digest(p) for p in score_files),
                    "window_token_sha256": " ".join(
                        digest(baseline_assets / f"wikitext_w{i}.bin") for i in range(4)
                    ),
                    "performance_file": perf_path.name,
                    "performance_sha256": digest(perf_path),
                }
            )
    return rows


def compare_ranks(rows: list[dict], offline_summary: Path, study: Path) -> str:
    """Compare every shared pair within a model against the complete offline cohort."""
    source_rows = read(offline_summary)["rows"]
    offline = {(r["model_id"], r["plan_id"]): r for r in source_rows}
    protocol = study / "selection_protocol.json"
    plans = read(protocol)["plan_files_sha256"]
    check(len(source_rows) == len(offline) == len(plans) == 33, offline_summary)
    check({f"plans/{m}/{p}.json" for m, p in offline} == set(plans), offline_summary)
    coverage = []
    reversals = []
    agreed_total = 0
    pair_total = 0
    for model in dict.fromkeys(r["model_id"] for r in rows):
        scorer = read(study / "scorer" / f"{model}.json")
        check(scorer["plans_sha256"] == digest(protocol), model)
        cohort = {}
        for row in rows:
            if row["model_id"] == model:
                plan_id = row["plan_id"]
                matched = offline[model, plan_id]
                plan = read(study / "plans" / model / f"{plan_id}.json")
                check(
                    row["split_manifest_sha256"] == scorer["split_manifest_sha256"],
                    model,
                )
                check(matched["config_hash"] == plan["config_hash"], plan_id)
                check(matched["scored_tokens"] == row["scored_tokens"], plan_id)
                check(math.isfinite(matched["mean_nll"]), plan_id)
                nll_sum = 0.0
                for i, token_hash in enumerate(row["window_token_sha256"].split()):
                    window = read(
                        study / "scores" / model / plan_id / f"wikitext_w{i}.json"
                    )
                    check(
                        window["token_hash"] == token_hash
                        and window["scored_tokens"] == 1023,
                        plan_id,
                    )
                    check(window["config_hash"] == matched["config_hash"], plan_id)
                    nll_sum += window["nll_sum"]
                check(
                    math.isclose(
                        nll_sum / row["scored_tokens"],
                        matched["mean_nll"],
                        rel_tol=1e-12,
                    ),
                    plan_id,
                )
                cohort[plan_id] = (matched["mean_nll"], row["mean_nll"])
        pairs = list(combinations(cohort, 2))
        agreements = 0
        ties = 0
        name = {
            "qwen3_1_7b": "Qwen3-1.7B",
            "qwen3_4b": "Qwen3-4B",
            "qwen3_8b": "Qwen3-8B",
        }[model]
        for a, b in pairs:
            off_a, htp_a = cohort[a]
            off_b, htp_b = cohort[b]
            off_diff, htp_diff = off_a - off_b, htp_a - htp_b
            direction = off_diff * htp_diff
            if direction > 0:
                agreements += 1
            else:
                ties += int(direction == 0)
                reversals.append(
                    f"| {name} | `{a}` | `{b}` | {off_diff:+.6f} | {htp_diff:+.6f} |"
                )
        agreed_total += agreements
        pair_total += len(pairs)
        coverage.append(
            f"| {name} | {len(cohort)}/11 | {len(pairs)} | {agreements} | {len(pairs) - agreements - ties} | {ties} |"
        )

    lines = [
        "# Agreement with archived Galaxy S26 Ultra measurements",
        "",
        f"The offline replay and the Snapdragon SM8850 HTP give the same direction for **{agreed_total} of {pair_total}** pairwise comparisons across the 12 configurations measured on both. Each pair compares two distinct masks or budgets within the same model. The quality metric is mean next-token NLL across the same four WikiText windows and 4,092 targets. Sorting by perplexity gives the same order.",
        "",
        "| Model | Matched masks | Pairs | Same order | Reversed | Tied in either evaluation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *coverage,
        "",
        "## Reversals and ties",
        "",
        "Each difference is the mean NLL of plan A minus that of plan B, in nats per token. A negative value favors A.",
        "",
        "| Model | Plan A | Plan B | Offline difference | HTP difference |",
        "| --- | --- | --- | ---: | ---: |",
        *reversals,
        "",
        "## Scope",
        "",
        "The archive covers K4/V4 on all three models and all five PU K6/V2 placements on 1.7B. Its 4B and 8B PU K6/V2 measurements cover boundary and interleaved placement. PU K5/V3 has no matching archived device score. Earlier K5/V3 measurements use different key codecs.",
        "",
        "The generator matches the W4A16 split, AR128 attention encodings, active codebooks, rotations, layer widths and token hashes. The 4B and 8B splits are specific to their respective checkpoints. All device timings in the reference are single archived 897-token prefill and 128-token generation runs. The replay approximates HTP arithmetic. The four fixed windows and overlapping within-model pairs support descriptive comparisons only.",
        "",
        "The reference CSV records each raw score and timing report with its SHA-256. The offline summary SHA-256 is "
        f"`{digest(offline_summary)}`. [HTP reference](htp_reference.csv) · [Offline scores](offline_scores.csv)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ORIGINAL_ARCHIVE)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--offline-summary",
        type=Path,
        help="Optional offline_comparison.json for rank agreement.",
    )
    parser.add_argument(
        "--agreement-report", type=Path, help="Save the Markdown rank comparison."
    )
    args = parser.parse_args()
    if args.agreement_report and not args.offline_summary:
        parser.error("--agreement-report requires --offline-summary")
    rows = collect(args.archive, args.study_root)
    if args.offline_summary:
        agreement = compare_ranks(rows, args.offline_summary, args.study_root)
        if args.agreement_report:
            args.agreement_report.parent.mkdir(parents=True, exist_ok=True)
            args.agreement_report.write_text(agreement)
        else:
            print(agreement)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
