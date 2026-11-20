# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Verify the raw-to-summary provenance of the local KV refinement without inference."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import pandas as pd
from compare_cross_model_budgets import digest, read, sources
from model_identity import QWEN_SHAPES
from niah_data import document_rows
from refinement_plans import ANCHORS, model_plans
from transformers import AutoTokenizer


def check(ok: bool, location: str | Path) -> None:
    if not ok:
        raise ValueError(f"Verification failed at {location}")


def check_hash(path: Path, sha: str) -> None:
    check(digest(path) == sha, path)


def verify_documents(root: Path, data: dict) -> None:
    """Bind every scored token payload to its article hash in the original split."""
    documents = {}
    for split, source in data["parquet"].items():
        path = Path(source["path"])
        check_hash(path, source["sha256"])
        for doc in document_rows(
            pd.read_parquet(path, columns=["text"])["text"].tolist()
        ):
            documents[split, doc["sha256"]] = doc
    check(set(data["tokenizers"]) == ANCHORS.keys(), "model tokenizers")
    tokenizers = []
    for info in data["tokenizers"].values():
        checkpoint = Path(info["checkpoint"])
        for name, sha in info["files_sha256"].items():
            check_hash(checkpoint / name, sha)
        tokenizers.append(
            AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        )
    seen_docs = set(data["excluded_document_sha256"])
    seen_titles = set(data["excluded_canonical_titles"])
    seen_sha = set(data["prior_token_files_sha256"].values())
    for role, count, official in (
        ("calibration", 4, {"train"}),
        ("heldout", 8, {"validation", "test"}),
    ):
        rows = data["windows"][role]
        check(len(rows) == count, f"{role} window count")
        for index, row in enumerate(rows):
            path = root / "assets" / row["file"]
            check(row["file"] == f"{role}_{index:02d}.bin", path)
            check(row["official_split"] in official, path)
            sha = row["document_sha256"]
            article = documents[row["official_split"], sha]
            title = " ".join(article["title"].strip("= ").lower().split())
            check(
                row["title"] == article["title"] and row["canonical_title"] == title,
                path,
            )
            check(sha not in seen_docs and title not in seen_titles, path)
            check(row["tokens_sha256"] not in seen_sha, path)
            seen_docs.add(sha)
            seen_titles.add(title)
            seen_sha.add(row["tokens_sha256"])
            check_hash(path, row["tokens_sha256"])
            payload = path.read_bytes()
            check((row["tokens"], row["scored_tokens"]) == (1024, 1023), path)
            for tokenizer in tokenizers:
                ids = tokenizer.encode(article["text"], add_special_tokens=False)
                off = row["offset"]
                check(
                    len(ids) == row["document_tokens"]
                    and off == min(256, len(ids) - 1024),
                    path,
                )
                check(payload == struct.pack("<1024i", *ids[off : off + 1024]), path)


def verify_raw(
    root: Path,
    model: str,
    split: str,
    plan: dict,
    windows: list[dict],
    scorer: str,
    summary: dict,
) -> tuple[float, set[Path]]:
    """Reconstruct a model/plan/split row from the manifest-identified raw windows."""
    plan_id = plan["plan_id"]
    group = root / "scores" / model / split / plan_id
    keys = ("plan_id", "category", "config_hash", "k8_layers", "v4_layers")
    check(all(summary[k] == plan[k] for k in keys), group)
    losses, means, paths = [], {}, set()
    for doc in windows:
        name = doc["file"]
        path = group / (Path(name).stem + ".json")
        paths.add(path)
        raw = read(path)
        expected = {
            "config_hash": plan["config_hash"],
            "scorer_sha256": scorer,
            "tokens_sha256": doc["tokens_sha256"],
            "window": name,
            "split": split,
            "scored_tokens": 1023,
        }
        check(all(raw[k] == v for k, v in expected.items()), path)
        loss = raw["nll_sum"]
        check(math.isfinite(loss) and loss >= 0, path)
        means[name] = loss / raw["scored_tokens"]
        check(math.isclose(raw["mean_nll"], means[name], rel_tol=1e-12), path)
        losses.append(loss)
    pooled = sum(losses) / sum(d["scored_tokens"] for d in windows)
    check(summary["documents"] == means, group)
    check(math.isclose(summary["nll"], pooled, rel_tol=1e-12), group)
    check(math.isclose(summary["ppl"], math.exp(pooled), rel_tol=1e-12), group)
    return pooled, paths


def verify(root: Path, archive: Path) -> dict:
    protocol = read(root / "evaluation_protocol.json")
    for file, sha in (
        ("selection_protocol.json", "plans_sha256"),
        ("data_manifest.json", "data_sha256"),
    ):
        check_hash(root / file, protocol[sha])
    check_hash(
        Path(__file__).with_name("run_kv_refinement.py"), protocol["runner_sha256"]
    )
    protocol_sha = digest(root / "evaluation_protocol.json")
    manifest = read(root / "selection_protocol.json")
    data = read(root / "data_manifest.json")
    summary = read(root / "summary.json")
    check(summary["protocol_sha256"] == protocol_sha, "summary protocol")
    check(
        set(summary["models"]) == set(protocol["model_ids"]) == ANCHORS.keys(),
        "model cohort",
    )
    verify_documents(root, data)
    plan_files, raw_files = set(), set()
    for model, group in summary["models"].items():
        scorer_path = root / "scorer" / (model + ".json")
        identity, scorer = read(scorer_path), digest(scorer_path)
        check(
            identity["model"] == model and identity["protocol_sha256"] == protocol_sha,
            scorer_path,
        )
        prior = archive / "cross_model_pu_budget_20261120"
        parent = prior / "scorer" / scorer_path.name
        check_hash(parent, identity["previous_scorer_sha256"])
        previous = read(parent)
        check_hash(
            Path(__file__).with_name("allocation_offline_model.py"), previous["scorer"]
        )
        _, assets, split = sources(archive, model)
        check_hash(split / "split_manifest.json", previous["split_manifest_sha256"])
        check_hash(assets / "assets.json", previous["input_assets"])
        asset_manifest = read(assets / "assets.json")
        rope = asset_manifest["rope"]
        check_hash(assets / rope, asset_manifest["sha256"][rope])
        for path, sha in previous["graphs_sha256"].items():
            check_hash(Path(path), sha)
        plans = model_plans(model)
        n, _, _, heads, dim = QWEN_SHAPES[model]
        for plan_id, plan in plans.items():
            name = f"plans/{model}/{plan_id}.json"
            plan_files.add(name)
            check_hash(root / name, manifest["plan_files_sha256"][name])
            check(read(root / name) == plan, name)
            total_bits = sum(k + v for k, v in plan["bits"])
            check(total_bits == 8 * n, name)
            indices, scales = total_bits * heads * dim // 8, n * heads * 4
            check(
                plan["cache_bytes_per_token"]
                == {
                    "indices": indices,
                    "fp16_scales": scales,
                    "total": indices + scales,
                },
                name,
            )
        historical = read(prior / "plans" / model / (ANCHORS[model] + ".json"))
        check(
            plans["anchor"]["bits"] == historical["bits"]
            and plans["anchor"]["config_hash"] == historical["config_hash"],
            model,
        )
        control = read(root / "controls" / (model + ".json"))
        reference = read(prior / "scores" / model / ANCHORS[model] / "wikitext_w0.json")
        check_hash(assets / reference["window"], control["tokens_sha256"])
        check(
            control["config_hash"]
            == reference["config_hash"]
            == historical["config_hash"]
            and control["tokens_sha256"] == reference["token_hash"]
            and control["reference_mean_nll"] == reference["mean_nll"]
            and abs(control["delta_mean_nll"]) <= 1e-5,
            f"{model} historical anchor parity",
        )
        chosen = read(root / "selection" / (model + ".json"))
        check(chosen["scorer_sha256"] == scorer, f"{model} selection scorer")
        check(
            group == read(root / "completed" / (model + ".json")),
            f"{model} completed summary",
        )
        check(
            group["anchor"] == chosen["anchor"] == "anchor"
            and group["candidates"] == chosen["calibration"],
            model,
        )
        expected_ids = set(plans)
        for role, key in (("calibration", "candidates"), ("heldout", "heldout")):
            rows = {r["plan_id"]: r for r in group[key]}
            check(
                len(rows) == len(group[key]) and rows.keys() == expected_ids,
                f"{model}/{role} cohort",
            )
            means = {}
            for plan_id, row in rows.items():
                means[plan_id], paths = verify_raw(
                    root,
                    model,
                    role,
                    plans[plan_id],
                    data["windows"][role],
                    scorer,
                    row,
                )
                raw_files.update(paths)
            if role == "calibration":
                ranked = sorted(means, key=lambda p: (means[p], p != "anchor", p))
                controls = {
                    c: next(p for p in ranked if plans[p]["category"] == c)
                    for c in ("budget_only", "placement_only")
                }
                check(
                    group["selected"] == chosen["selected"] == ranked[0],
                    f"{model} primary winner",
                )
                check(
                    group["category_controls"]
                    == chosen["category_controls"]
                    == controls,
                    f"{model} category controls",
                )
                expected_ids = {"anchor", ranked[0], *controls.values()}
    check(manifest["plan_files_sha256"].keys() == plan_files, "plan manifest cohort")
    check(set((root / "scores").rglob("*.json")) == raw_files, "raw score file cohort")
    return {
        "models": len(summary["models"]),
        "plans": len(plan_files),
        "documents": 12,
        "raw_score_files": len(raw_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--archive", type=Path, help="Parent archive containing the historical runs"
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.archive or args.root.parent)))


if __name__ == "__main__":
    main()
