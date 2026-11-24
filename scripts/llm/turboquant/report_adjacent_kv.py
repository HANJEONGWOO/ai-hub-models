# SPDX-License-Identifier: BSD-3-Clause
"""Verify the six adjacent-budget validation runs and export their ranking."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from adjacent_kv_plans import build_plans
from compare_cross_model_budgets import digest, read
from cross_model_plans import _freeze_json
from report_local_kv_scores import export_csv

EXPERIMENTS = Path(__file__).parent / "experiments"
REFERENCE = EXPERIMENTS / "relative_kv"
ANCHOR = "early_07"


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def report(root: Path, out: Path, prior: Path) -> None:
    identity_path = root / "evaluation_identity.json"
    identity, ident = read(identity_path), digest(identity_path)
    old_identity = digest(prior / "evaluation_identity.json")
    protocol = read(root / "protocol.json")
    if (
        identity["prior_evaluation_sha256"] != old_identity
        or identity["protocol_sha256"] != digest(root / "protocol.json")
        or digest(root / "protocol.json")
        != digest(EXPERIMENTS / "adjacent_kv/protocol.json")
        or identity["plans_sha256"] != digest(root / "plans.json")
        or protocol["source_quota_table_sha256"]
        != digest(REFERENCE / "quota_plans.csv")
        or identity["prior_document_nll_sha256"]
        != digest(REFERENCE / "document_nll.csv")
        or protocol["prior_manifest_sha256"] != identity["data_manifest_sha256"]
        or protocol["prior_manifest_sha256"] != digest(prior / "data_manifest.json")
        or digest(prior / "data_manifest.json")
        != read(prior / "evaluation_identity.json")["data_sha256"]
    ):
        raise ValueError("Protocol, plans or previous evaluation identity mismatch")
    plans = read(root / "plans.json")
    result = read(root / "validation_result.json")
    names = {f"{p}_{k:02d}" for k in (6, 7, 8) for p in ("early", "dp")}
    if (
        result["identity"] != ident
        or set(plans) != names
        or set(result["summaries"]) != names
        or len({p["config_hash"] for p in plans.values()}) != 6
    ):
        raise ValueError("Expected six distinct plans and matching score summaries")
    if plans != build_plans(REFERENCE / "quota_plans.csv"):
        raise ValueError("Masks, codec configuration or measured KV capacity changed")
    published = {
        (r["config_hash"], r["tokens_sha256"]): r
        for r in read_rows(REFERENCE / "document_nll.csv")
        if r["split"] == "validation"
    }
    docs = read(prior / "data_manifest.json")["windows"]["validation"]
    if (
        len(docs) != 8
        or len({d["file"] for d in docs}) != 8
        or len({d["document_sha256"] for d in docs}) != 8
        or len({d["tokens_sha256"] for d in docs}) != 8
    ):
        raise ValueError("Validation must consist of eight unique windows")
    for doc in docs:
        if (
            doc["tokens"] != 1024
            or doc["scored_tokens"] != 1023
            or digest(prior / "assets" / doc["file"]) != doc["tokens_sha256"]
        ):
            raise ValueError(f"Changed validation window: {doc['file']}")

    summaries, documents = {}, []
    for name in sorted(names):
        plan = plans[name]
        rows = []
        for doc in docs:
            filename = doc["file"]
            basename = Path(filename).with_suffix(".json")
            path = root / "nll" / "validation" / name / basename
            raw = read(path)
            expected = {
                "evaluation": ident,
                "file": filename,
                "stage": "validation",
                "tokens_sha256": doc["tokens_sha256"],
                "plan": plan["config_hash"],
                "scored_tokens": 1023,
            }
            if any(raw[k] != v for k, v in expected.items()):
                raise ValueError(f"Invalid score identity: {path}")
            loss = raw["nll_sum"]
            mean = loss / raw["scored_tokens"]
            if not math.isfinite(loss) or loss < 0 or mean != raw["mean_nll"]:
                raise ValueError(f"Invalid document NLL: {path}")
            previous = published.get((raw["plan"], raw["tokens_sha256"]))
            source = "new"
            if previous:
                old_path = prior / "nll/validation" / previous["plan_id"] / basename
                if old_path.is_file() and digest(old_path) == previous["raw_sha256"]:
                    source = "reused"
                    old = read(old_path)
                    expected_old = {
                        **expected,
                        "evaluation": old_identity,
                        "nll_sum": loss,
                    }
                    if (
                        any(old[k] != v for k, v in expected_old.items())
                        or raw.get("source_plan_id") != previous["plan_id"]
                        or raw.get("source_raw_sha256") != previous["raw_sha256"]
                    ):
                        raise ValueError(
                            f"Prior score fingerprint mismatch: {old_path}"
                        )
            if raw["source_kind"] != source:
                raise ValueError(f"Incorrect score provenance: {path}")
            rows.append(
                {
                    "plan_id": name,
                    "file": filename,
                    "title": doc["title"],
                    "config_hash": raw["plan"],
                    "tokens_sha256": raw["tokens_sha256"],
                    "nll_sum": loss,
                    "scored_tokens": raw["scored_tokens"],
                    "mean_nll": mean,
                    "ppl": math.exp(mean),
                    "source_kind": source,
                    "raw_sha256": digest(path),
                    "source_raw_sha256": raw.get("source_raw_sha256", ""),
                }
            )
        total = math.fsum(r["nll_sum"] for r in rows)
        tokens = sum(r["scored_tokens"] for r in rows)
        pooled = total / tokens
        reused = sum(r["source_kind"] == "reused" for r in rows)
        summary = {
            "nll_sum": total,
            "scored_tokens": tokens,
            "mean_nll": pooled,
            "ppl": math.exp(pooled),
            "reused": reused,
            "new": len(rows) - reused,
        }
        saved = result["summaries"][name]
        if (
            saved["documents"] != {r["file"]: r["mean_nll"] for r in rows}
            or any(saved[k] != v for k, v in summary.items())
            or tokens != 8184
        ):
            raise ValueError(f"Aggregate NLL/provenance differs: {name}")
        documents.extend(rows)
        summaries[name] = summary

    order = sorted(names, key=lambda n: (summaries[n]["mean_nll"], n != ANCHOR, n))
    winner = order[0]
    if (
        result["ranking"] != order
        or result["selected"] != winner
        or result["selected_config_hash"] != plans[winner]["config_hash"]
        or any(
            result[key] != sum(s[key] for s in summaries.values())
            for key in ("new", "reused")
        )
    ):
        raise ValueError("Winner, ranking or count differs from verified 48 rows")
    comparisons = []
    for rank, name in enumerate(order, start=1):
        p, s = plans[name], summaries[name]
        comparisons.append(
            {
                "rank": rank,
                "plan_id": name,
                "k8_count": len(p["k8_layers"]),
                "v4_count": len(p["v4_layers"]),
                "mean_k_bits": p["key_average_bits"],
                "mean_v_bits": p["value_average_bits"],
                **s,
                "nll_minus_early_07": s["mean_nll"] - summaries[ANCHOR]["mean_nll"],
                "allocated_cache_bytes": p["allocated_cache_bytes"],
                "config_hash": p["config_hash"],
            }
        )
    export_csv(out / "comparison.csv", comparisons)
    export_csv(out / "document_nll.csv", documents)
    _freeze_json(
        out / "selected_plan.json",
        {
            **plans[winner],
            "plan_id": winner,
            "selected_mean_nll": summaries[winner]["mean_nll"],
            "selected_ppl": summaries[winner]["ppl"],
            "scored_tokens": 8184,
            "evaluation_sha256": ident,
            "validation_result_sha256": digest(root / "validation_result.json"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    report(args.root, args.out, args.prior or args.root.parent / "relative_kv_20261124")
