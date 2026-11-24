# SPDX-License-Identifier: BSD-3-Clause
"""Verify the frozen validation cohort, W4A16 evaluator and reusable raw NLLs."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from compare_cross_model_budgets import digest, read, sources

from qai_hub_models.models.templates.llm.turboquant import (
    constants,
    reference,
    structured_k8,
)

TOOLS = Path(__file__).resolve().parent
PRIOR_REPORT = TOOLS / "experiments/relative_kv"
PRIOR_DOCUMENTS = PRIOR_REPORT / "document_nll.csv"
PRIOR_EVALUATION_SHA256 = (
    "cf5dfcaf8d5218e5884d760f0a884778eef12bb3e0b61aafe6e79fb40978458a"
)
PRIOR_DOCUMENTS_SHA256 = (
    "a16427cb6c623a7e47ef4b13c3eca87912920e4602a50d093984a8859e843132"
)
ALIASES = {"early_07": "early", "dp_07": "relative_07"}


def checked(path: Path, expected: str) -> str:
    if digest(path) != expected:
        raise ValueError(f"Prior SHA-256 mismatch: {path}")
    return expected


def verify_inputs(prior: Path, plans: dict | None = None) -> tuple[list[dict], dict]:
    evaluation = prior / "evaluation_identity.json"
    checked(evaluation, PRIOR_EVALUATION_SHA256)
    previous = read(evaluation)
    checked(PRIOR_REPORT / "protocol.json", previous["protocol_sha256"])
    checked(prior / "protocol.json", previous["protocol_sha256"])
    checked(prior / "data_manifest.json", previous["data_sha256"])
    checked(PRIOR_DOCUMENTS, PRIOR_DOCUMENTS_SHA256)
    docs = read(prior / "data_manifest.json")["windows"]["validation"]
    if len(docs) != 8 or len({doc["document_sha256"] for doc in docs}) != 8:
        raise ValueError("Expected the frozen eight validation articles")
    for i, doc in enumerate(docs):
        if (doc["file"], doc["tokens"], doc["scored_tokens"]) != (
            f"validation_{i:02}.bin",
            1024,
            1023,
        ):
            raise ValueError(f"Unexpected validation manifest row: {doc['file']}")
        token_file = prior / "assets" / doc["file"]
        if token_file.stat().st_size != 4096:
            raise ValueError(f"Unexpected int32 token count: {token_file}")
        checked(token_file, doc["tokens_sha256"])

    runtime = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
    }
    if any(previous[k] != value for k, value in runtime.items()):
        raise ValueError(f"Runtime differs from prior W4A16 evaluator: {runtime}")
    for name in ("allocation_offline_model.py", "evaluate_local_kv_scores.py"):
        checked(TOOLS / name, previous["software"][name])
    # Allocation replay delegates rotation and the LM/structured codebooks to
    # these modules. Pin their source bytes from the preceding study commit.
    codec_modules = {
        reference: "d25d667b64e04ec4ba87adb7438122a3c1385a4ab543699b6aaf825fbb41fb87",
        constants: "9df5ed87e38a8020b4d66a3e81128c01dd97b665fc5b4f96c342dfdbe4223460",
        structured_k8: "6e679696e64df6245e1ef0355e76f929c07611f87c9164064b92a41eda78a44f",
    }
    for module, expected in codec_modules.items():
        checked(Path(module.__file__), expected)

    historical = prior.parent / "local_kv_scores_20261123/evaluation_identity.json"
    checked(historical, previous["previous_w4a16_identity_sha256"])
    inventory = read(historical)["graphs_and_inputs"]
    bundle, assets, split = sources(prior.parent, "qwen3_1_7b")
    if not all(
        str(path) in inventory
        for path in (
            split / "split_manifest.json",
            assets / "assets.json",
            bundle / "prompt_ar128_cl1024_2_of_4.onnx",
        )
    ):
        raise ValueError("Runtime source directories do not match the frozen inventory")
    for filename, expected in inventory.items():
        checked(Path(filename), expected)

    if plans is not None:
        with (PRIOR_REPORT / "comparison.csv").open(newline="") as stream:
            prior_plans = {
                r["plan_id"]: r["config_hash"] for r in csv.DictReader(stream)
            }
        for name, source_name in ALIASES.items():
            if plans[name]["config_hash"] != prior_plans[source_name]:
                raise ValueError(f"Historical config hash mismatch for {name}")
    return docs, {
        "prior_evaluation_sha256": PRIOR_EVALUATION_SHA256,
        "data_manifest_sha256": previous["data_sha256"],
        "prior_document_nll_sha256": PRIOR_DOCUMENTS_SHA256,
        "previous_w4a16_identity_sha256": previous["previous_w4a16_identity_sha256"],
        "scorer_sha256": previous["software"]["allocation_offline_model.py"],
        "codec_sources_sha256": {m.__name__: sha for m, sha in codec_modules.items()},
        **runtime,
    }


def reusable(
    prior: Path, name: str, plan: dict, doc: dict, identity: dict
) -> dict | None:
    """Return a historical raw record only if its exact bytes were published."""
    source = ALIASES.get(name)
    if source is None:
        return None
    with PRIOR_DOCUMENTS.open(newline="") as stream:
        entry = next(
            (
                r
                for r in csv.DictReader(stream)
                if r["split"] == "validation"
                and r["plan_id"] == source
                and r["file"] == doc["file"]
            ),
            None,
        )
    if (
        entry is None
        or entry["config_hash"] != plan["config_hash"]
        or entry["tokens_sha256"] != doc["tokens_sha256"]
    ):
        return None
    path = prior / "nll/validation" / source / (Path(doc["file"]).stem + ".json")
    if not path.is_file() or digest(path) != entry["raw_sha256"]:
        return None
    row = read(path)
    fields = {
        "evaluation": identity["prior_evaluation_sha256"],
        "stage": "validation",
        "plan": plan["config_hash"],
        "file": doc["file"],
        "tokens_sha256": doc["tokens_sha256"],
        "scored_tokens": doc["scored_tokens"],
        "nll_sum": float(entry["nll_sum"]),
    }
    if not all(row[k] == value for k, value in fields.items()) or not math.isfinite(
        row["nll_sum"]
    ):
        return None
    return row
