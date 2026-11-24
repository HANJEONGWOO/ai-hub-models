# SPDX-License-Identifier: BSD-3-Clause
"""Freeze shared calibration and fresh validation/test windows for four-state allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from cross_model_plans import _freeze_json
from niah_data import digest, document_rows, text_digest
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

PREVIOUS = "local_kv_scores_20261123"
PREVIOUS_MANIFEST = "a9adf42b02130989167452614b1a96d7a275022f075fcc57179dd9efdc61fa4d"
SALT = "Qwen3-1.7B-relative-KV-20261124/"
COUNT = 8


def freeze(root: Path, archive: Path) -> dict:
    prior_dir = archive / PREVIOUS
    prior_manifest = prior_dir / "data_manifest.json"
    if digest(prior_manifest) != PREVIOUS_MANIFEST:
        raise ValueError("Previous calibration manifest identity mismatch")
    prior = json.loads(prior_manifest.read_bytes())
    target = root / "data_manifest.json"
    if target.exists():
        saved = json.loads(target.read_bytes())
        manifests = list(map(Path, saved["prior_manifest_sha256"]))
        old_tokens = set(map(Path, saved["prior_token_files_sha256"]))
    else:
        manifests, old_tokens = inventory_paths(archive, root)
    if prior_manifest not in manifests:
        raise ValueError("Missing original study in exclusion inventory")
    inventory, titles, hashes, forbidden = archive_inventory(
        archive, manifests, old_tokens
    )
    tokenizer_description = prior["tokenizers"]["qwen3_1_7b"]
    checkpoint = Path(tokenizer_description["checkpoint"])
    for name, expected in tokenizer_description["files_sha256"].items():
        if digest(checkpoint / name) != expected:
            raise ValueError(f"Tokenizer artifact changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)

    articles = {}
    for split, item in prior["parquet"].items():
        source = Path(item["path"])
        if digest(source) != item["sha256"]:
            raise ValueError(f"Dataset changed: {source}")
        articles[split] = document_rows(
            pd.read_parquet(source, columns=["text"])["text"].tolist()
        )

    def encode(article: dict) -> np.ndarray:
        return np.array(
            tokenizer.encode(article["text"], add_special_tokens=False), dtype="<i4"
        )

    prior_articles = {
        r["document_sha256"]: r for rows in prior["windows"].values() for r in rows
    }
    validated = set()
    for records in articles.values():
        for article in records:
            row = prior_articles.get(article["sha256"])
            if row is None:
                continue
            ids = encode(article)
            payload = ids[row["offset"] : row["offset"] + LENGTH].tobytes()
            original = prior_dir / "assets" / row["file"]
            if (
                hashlib.sha256(payload).hexdigest() != row["tokens_sha256"]
                or original.read_bytes() != payload
            ):
                raise ValueError(f"Original article/window changed: {original}")
            forbidden.update(spans(ids))
            validated.add(row["document_sha256"])
    if validated != set(prior_articles) or len(validated) != 24:
        raise ValueError("Incomplete original article coverage")

    windows = {"calibration": prior["windows"]["calibration"]}
    if len(windows["calibration"]) != COUNT:
        raise ValueError("Incomplete original calibration")
    payloads = {
        r["file"]: (prior_dir / "assets" / r["file"]).read_bytes()
        for r in windows["calibration"]
    }
    official_counts = {}

    def select(pool: list[tuple[str, dict]]) -> list[tuple[dict, np.ndarray]]:
        picked = []
        local_titles, local_hashes, local_spans = set(), set(), set()
        for _, article in sorted(pool, key=lambda p: p[0]):
            title, sha = canonical(article["title"]), article["sha256"]
            if (
                title in titles
                or sha in hashes
                or title in local_titles
                or sha in local_hashes
            ):
                continue
            ids = encode(article)
            if len(ids) < LENGTH or any(
                g in forbidden or g in local_spans for g in spans(ids)
            ):
                continue
            picked.append((article, ids))
            local_titles.add(title)
            local_hashes.add(sha)
            local_spans.update(spans(ids))
            if len(picked) == COUNT:
                break
        return picked

    reserve = [[], []]
    for article in articles["train"]:
        key = text_digest(SALT + article["text"])
        reserve[int(key, 16) % 2].append((key, article))
    for idx, role in enumerate(("validation", "test")):
        split = role
        cohort = select([(text_digest(SALT + a["text"]), a) for a in articles[split]])
        official_counts[role] = len(cohort)
        if len(cohort) < COUNT:
            split = "train"
            cohort = select(reserve[idx])
        if len(cohort) != COUNT:
            raise ValueError(f"Insufficient fresh articles for {role}")
        windows[role] = []
        for i, (article, ids) in enumerate(cohort):
            title = canonical(article["title"])
            offset = min(256, len(ids) - LENGTH)
            file = f"{role}_{i:02d}.bin"
            payloads[file] = ids[offset : offset + LENGTH].tobytes()
            windows[role].append(
                {
                    "file": file,
                    "title": article["title"],
                    "canonical_title": title,
                    "document_sha256": article["sha256"],
                    "document_tokens": len(ids),
                    "official_split": split,
                    "offset": offset,
                    "tokens": LENGTH,
                    "scored_tokens": LENGTH - 1,
                    "tokens_sha256": hashlib.sha256(payloads[file]).hexdigest(),
                }
            )
            titles.add(title)
            hashes.add(article["sha256"])
            forbidden.update(spans(ids))

    result = {
        **{
            k: prior[k]
            for k in ("dataset", "revision", "parquet", "tokenizers", "token_format")
        },
        **inventory,
        "calibration_manifest_sha256": PREVIOUS_MANIFEST,
        "official_eligible_up_to_eight": official_counts,
        "selection_salt": SALT,
        "selection": "Reuse all eight original calibration articles and exact token windows. For validation and test independently, prefer the official split if eight articles qualify; otherwise use a fresh train reserve. Reserve split = integer SHA256(salt+exact text) mod 2 (0 validation, 1 test); order by the same hash. Offset=min(256, length-1024).",
        "exclusion": "For new articles, exclude archived canonical titles and source document hashes, all inventoried prior 16-token sequences, all full-article 16-token sequences from the 24 prior-study articles and the newly selected articles. Independence refers to recorded local studies.",
        "windows": windows,
    }
    for name, content in payloads.items():
        publish(root / "assets" / name, content)
    _freeze_json(target, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    args = p.parse_args()
    frozen = freeze(args.root.resolve(), args.archive.resolve())
    for role, rows in frozen["windows"].items():
        print(role, len(rows), rows[0]["official_split"])
    print("Official eligible up to eight", frozen["official_eligible_up_to_eight"])
    print("data_manifest_sha256", digest(args.root / "data_manifest.json"))
