# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Freeze shared CL1024 allocation-calibration and held-out WikiText documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from compare_cross_model_budgets import sources
from cross_model_plans import MODELS, _freeze_json
from niah_data import DEFAULT_SOURCE, REVISION, digest, document_rows, text_digest
from transformers import AutoTokenizer

ARCHIVE = Path("/mnt/d/ai-hub-models/binaries/turboquant")
SALT = "Qwen3-PU-local-refinement-20261120:"
LENGTH = 1024
GRAM = 16
SPLITS = {"calibration": (4, ("train",)), "heldout": (8, ("test", "validation"))}


def canonical(title: str) -> str:
    return " ".join(title.strip().strip("=").strip().lower().split())


def spans(ids: np.ndarray) -> Iterator[bytes]:
    return (ids[i : i + GRAM].tobytes() for i in range(len(ids) - GRAM + 1))


def inventory_paths(archive: Path, root: Path) -> tuple[list[Path], set[Path]]:
    manifests, files = [], set()
    for directory, children, names in os.walk(archive):
        parent = Path(directory)
        children[:] = [name for name in children if parent / name != root]
        for name in names:
            path = parent / name
            if name in {"data_manifest.json", "prior_data_inventory.json"}:
                manifests.append(path)
            elif name == "tokens.npz" or (
                "assets" in parent.name
                and name.endswith(".bin")
                and name.startswith(
                    (
                        "wikitext_",
                        "calibration_",
                        "search_",
                        "validation_",
                        "test_",
                        "train_",
                        "heldout_",
                        "pilot_",
                        "smoke_",
                        "prompt_ids",
                        "boundary_prompt",
                    )
                )
            ):
                files.add(path)
    return manifests, files


def archive_inventory(
    archive: Path, manifests: list[Path], files: set[Path]
) -> tuple[dict, set, set, set]:
    titles, documents, prior_spans = set(), set(), set()
    expected = {}

    def visit(obj: object) -> None:
        if isinstance(obj, list):
            for item in obj:
                visit(item)
        if not isinstance(obj, dict):
            return
        for key in ("title", "canonical_title"):
            if key in obj:
                titles.add(canonical(obj[key]))
        for key in ("canonical_titles", "excluded_titles"):
            titles.update(map(canonical, obj.get(key, [])))
        for key in ("sha256", "document_sha256"):
            if "title" in obj and isinstance(obj.get(key), str):
                documents.add(obj[key])
        for key in ("token_files_sha256", "prior_token_files_sha256"):
            for p, sha in obj.get(key, {}).items():
                path = Path(p)
                if expected.setdefault(path, sha) != sha:
                    raise ValueError(f"Conflicting prior token identities for {path}")
                files.add(path)
        for value in obj.values():
            visit(value)

    for path in manifests:
        visit(json.loads(path.read_bytes()))
    historical = [
        archive / "device_assets_cl1024" / f"wikitext_w{i}.bin" for i in range(4)
    ]
    if not manifests or any(p not in files for p in historical):
        raise ValueError("Historical experiment data inventory is incomplete")
    hashes, unique = {}, set()
    for path in sorted(files):
        sha = digest(path)
        if path in expected and expected[path] != sha:
            raise ValueError(f"Prior token file differs from recorded hash: {path}")
        hashes[str(path)] = sha
        if sha in unique:
            continue
        unique.add(sha)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as saved:
                for data in saved.values():
                    for row in data.reshape(-1, data.shape[-1]):
                        prior_spans.update(spans(row.astype("<i4")))
        else:
            prior_spans.update(spans(np.fromfile(path, dtype="<i4")))
    return (
        {
            "prior_manifest_sha256": {str(p): digest(p) for p in sorted(manifests)},
            "prior_token_files_sha256": hashes,
            "prior_unique_token_payloads": len(unique),
            "prior_16gram_count": len(prior_spans),
            "excluded_canonical_titles": sorted(titles),
            "excluded_document_sha256": sorted(documents),
            "historical_windows_sha256": {p.name: hashes[str(p)] for p in historical},
        },
        titles,
        documents,
        prior_spans,
    )


def tokenizer_identity(archive: Path) -> tuple[dict, dict]:
    tokenizers, identity = {}, {}
    for model in MODELS:
        _, assets, _ = sources(archive, model)
        checkpoint = Path(
            json.loads((assets / "assets.json").read_bytes())["checkpoint_dir"]
        )
        tokenizers[model] = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True
        )
        identity[model] = {
            "checkpoint": str(checkpoint),
            "files_sha256": {
                p.name: digest(p)
                for name in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "added_tokens.json",
                    "merges.txt",
                    "vocab.json",
                )
                if (p := checkpoint / name).exists()
            },
        }
    return tokenizers, identity


def publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    with NamedTemporaryFile(dir=path.parent) as tmp:
        tmp.write(payload)
        tmp.flush()
        try:
            path.hardlink_to(tmp.name)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(
                    f"Existing frozen token payload differs: {path}"
                ) from None


def freeze(root: Path, archive: Path) -> dict:
    outfile = root / "data_manifest.json"
    tokenizers, tokenizer_meta = tokenizer_identity(archive)
    primary = tokenizers[MODELS[0]]
    if outfile.exists():
        # Later studies may introduce new documents. Reproduce this freeze using
        # exactly the prior-data inventory available when it was made.
        frozen = json.loads(outfile.read_bytes())
        manifests = [Path(p) for p in frozen["prior_manifest_sha256"]]
        files = {Path(p) for p in frozen["prior_token_files_sha256"]}
    else:
        manifests, files = inventory_paths(archive, root)
    inventory, titles, previous_docs, forbidden = archive_inventory(
        archive, manifests, files
    )
    inputs, windows, payloads = {}, {}, {}
    for role, (expected, splits) in SPLITS.items():
        candidates = []
        for split in splits:
            path = DEFAULT_SOURCE.with_name(f"{split}-00000-of-00001.parquet")
            inputs[split] = {"path": str(path), "sha256": digest(path)}
            for doc in document_rows(pd.read_parquet(path)["text"].tolist()):
                candidates.append(
                    (text_digest(SALT + role + ":" + doc["text"]), split, doc)
                )
        rows = windows[role] = []
        for _, split, doc in sorted(candidates, key=lambda c: (c[0], c[1])):
            title = canonical(doc["title"])
            if title in titles or doc["sha256"] in previous_docs:
                continue
            ids = np.asarray(
                primary.encode(doc["text"], add_special_tokens=False), dtype="<i4"
            )
            if len(ids) < LENGTH or any(span in forbidden for span in spans(ids)):
                continue
            for model in MODELS[1:]:
                other = tokenizers[model].encode(doc["text"], add_special_tokens=False)
                if not np.array_equal(ids, other):
                    raise ValueError(f"Token IDs differ for {model}: {title}")
            offset = min(256, len(ids) - LENGTH)
            name = f"{role}_{len(rows):02d}.bin"
            payload = ids[offset : offset + LENGTH].tobytes()
            rows.append(
                {
                    "title": doc["title"],
                    "canonical_title": title,
                    "document_sha256": doc["sha256"],
                    "document_tokens": len(ids),
                    "official_split": split,
                    "offset": offset,
                    "file": name,
                    "tokens": LENGTH,
                    "scored_tokens": LENGTH - 1,
                    "tokens_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            titles.add(title)
            previous_docs.add(doc["sha256"])
            forbidden.update(spans(ids))
            payloads[name] = payload
            if len(rows) == expected:
                break
        if len(rows) != expected:
            raise ValueError(f"Only {len(rows)}/{expected} eligible {role} documents")
    manifest = {
        "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
        "revision": REVISION,
        "parquet": inputs,
        "tokenizers": tokenizer_meta,
        "token_format": "Headerless little-endian int32; no added special tokens",
        "selection_salt": SALT,
        "selection": "For each role, ascending SHA256(salt + role + ':' + exact UTF-8 article). Calibration uses train. Heldout pools official test and validation. Retain first eligible distinct articles in hash order, with offset min(256, document_tokens - 1024).",
        "exclusion": "Reject previously recorded canonical titles or article hashes and any candidate article sharing a contiguous 16-token span with inventoried token files or another selected full article. Verify full-article IDs across all three checkpoint tokenizers.",
        "scope": "Independent from recorded local experiment data. Pretraining and original activation-calibration document membership are unknown.",
        "overlap_16grams": 0,
        **inventory,
        "windows": windows,
    }
    for name, payload in payloads.items():
        publish(root / "assets" / name, payload)
    _freeze_json(outfile, manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    args = p.parse_args()
    result = freeze(args.root.resolve(), args.archive.resolve())
    for split_name, rows in result["windows"].items():
        print(split_name, [(d["file"], d["title"]) for d in rows], flush=True)
