# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Freeze article-disjoint WikiText windows for the local K/V score experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_cross_model_budgets import sources
from cross_model_plans import _freeze_json
from niah_data import DEFAULT_SOURCE, REVISION, digest, document_rows, text_digest
from refinement_data import (
    ARCHIVE,
    LENGTH,
    archive_inventory,
    canonical,
    inventory_paths,
    publish,
    spans,
)
from transformers import AutoTokenizer

SALT = "qwen3-1.7b-local-kv-scores-20261123/"
ROLES = ("calibration", "validation", "test")
WINDOWS_PER_SPLIT = 8


def freeze(root: Path, archive: Path) -> dict:
    manifest_path = root / "data_manifest.json"
    if manifest_path.exists():
        frozen = json.loads(manifest_path.read_bytes())
        manifests = list(map(Path, frozen["prior_manifest_sha256"]))
        files = set(map(Path, frozen["prior_token_files_sha256"]))
    else:
        manifests, files = inventory_paths(archive, root)
    inventory, titles, documents, forbidden = archive_inventory(
        archive, manifests, files
    )
    _, assets, _ = sources(archive, "qwen3_1_7b")
    checkpoint = Path(
        json.loads((assets / "assets.json").read_bytes())["checkpoint_dir"]
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer_files = {
        name: digest(checkpoint / name)
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "merges.txt",
            "vocab.json",
        )
        if (checkpoint / name).exists()
    }
    parquet = {}

    def read_articles(split: str) -> list[dict]:
        path = DEFAULT_SOURCE.with_name(f"{split}-00000-of-00001.parquet")
        parquet[split] = {"path": str(path), "sha256": digest(path)}
        return document_rows(pd.read_parquet(path, columns=["text"])["text"].tolist())

    pools: list[list[tuple[str, dict]]] = [[], [], []]
    for article in read_articles("train"):
        key = text_digest(SALT + article["text"])
        pools[int(key, 16) % len(ROLES)].append((key, article))

    def eligible(pool: list[tuple[str, dict]]) -> list[tuple[dict, np.ndarray]]:
        picked = []
        current_titles = set()
        current_documents = set()
        current_grams = set()
        for _, article in sorted(pool, key=lambda pair: pair[0]):
            title = canonical(article["title"])
            if (
                title in titles
                or title in current_titles
                or article["sha256"] in documents
                or article["sha256"] in current_documents
            ):
                continue
            ids = np.asarray(
                tokenizer.encode(article["text"], add_special_tokens=False), dtype="<i4"
            )
            if len(ids) < LENGTH or any(
                gram in forbidden or gram in current_grams for gram in spans(ids)
            ):
                continue
            picked.append((article, ids))
            current_titles.add(title)
            current_documents.add(article["sha256"])
            current_grams.update(spans(ids))
            if len(picked) == WINDOWS_PER_SPLIT:
                break
        return picked

    windows, payloads, official_coverage = {}, {}, {}
    for role, pool in zip(ROLES, pools, strict=True):
        split = role if role != "calibration" else "train"
        if split != "train":
            official = [
                (text_digest(SALT + a["text"]), a) for a in read_articles(split)
            ]
            picked = eligible(official)
            official_coverage[role] = len(picked)
            if len(picked) < WINDOWS_PER_SPLIT:
                split = "train"
        if split == "train":
            picked = eligible(pool)
        rows = windows[role] = []
        for article, ids in picked:
            title = canonical(article["title"])
            offset = min(256, len(ids) - LENGTH)
            name = f"{role}_{len(rows):02}.bin"
            payload = ids[offset : offset + LENGTH].tobytes()
            rows.append(
                {
                    "file": name,
                    "title": article["title"],
                    "canonical_title": title,
                    "document_sha256": article["sha256"],
                    "document_tokens": len(ids),
                    "official_split": split,
                    "offset": offset,
                    "tokens": LENGTH,
                    "scored_tokens": LENGTH - 1,
                    "tokens_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            payloads[name] = payload
            titles.add(title)
            documents.add(article["sha256"])
            forbidden.update(spans(ids))
        if len(rows) != WINDOWS_PER_SPLIT:
            raise ValueError(f"{role}: selected only {len(rows)} eligible documents")
    manifest = {
        "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
        "revision": REVISION,
        "parquet": parquet,
        "tokenizers": {
            "qwen3_1_7b": {
                "checkpoint": str(checkpoint),
                "files_sha256": tokenizer_files,
            }
        },
        "selection_salt": SALT,
        "selection": "Prefer the corresponding official split for validation/test when eight eligible articles remain. Partition official train by SHA256(salt + exact article text) as unsigned big-endian modulo 3 (0 calibration, 1 reserve validation, 2 reserve test). If an official split has insufficient eligible articles, use its entire reserve cohort. Sort candidates by the same hash, take the first eight eligible articles and offset min(256, article tokens - 1024).",
        "official_eligible_up_to_eight": official_coverage,
        "exclusion": "Exclude prior canonical titles, document hashes, or any 16-token span shared between the full candidate article and inventoried past token files or previously selected full articles across any role.",
        "scope": "Independence is with respect to recorded local study data.",
        "token_format": "Headerless little-endian int32 with no added special tokens",
        "overlap_16grams": 0,
        **inventory,
        "windows": windows,
    }
    for name, payload in payloads.items():
        publish(root / "assets" / name, payload)
    _freeze_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = freeze(root, args.archive.resolve())
    for role, rows in manifest["windows"].items():
        print(role, len(rows), flush=True)
    print("data_manifest_sha256", digest(root / "data_manifest.json"))
