# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Freeze new documents and search rules before observing allocation NLLs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_asymmetric_once import ARTIFACTS, digest, read, save_new
from benchmark_nll_allocation import STATIC, symlink

from qai_hub_models.models.templates.llm.turboquant.config import (
    PROFILES,
    structured_allocation_plan,
)

LATEST = ARTIFACTS / "uniform_k8_baseline_20261020"
OLD = ARTIFACTS / "structured_k8_nll_allocation_20261013"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def freeze_weights(root: Path) -> None:
    split = ARTIFACTS / "qwen3_1_7b_w4a16_split/split_manifest.json"
    files = [split]
    for entry in read(split)["parts"].values():
        for suffix in ("*.onnx", "*.data", "*.encodings"):
            files.extend(Path(entry["bundle_dir"]).glob(suffix))
    save_new(
        root / "source_identity.json",
        {
            "files_sha256": {str(p): digest(p) for p in files},
            "purpose": "Unchanged exported W4A16 weights and activation grids",
            "performance_order": "Frozen rule in protocol.json; resolved physical aliases in device_order.json before test/performance",
        },
    )


def documents(path: Path) -> list[tuple[str, str]]:
    result, current, title = [], [], ""
    for row in pd.read_parquet(path)["text"]:
        if re.fullmatch(r"\s*= [^=]+ =\s*", row):
            if current:
                result.append((title, "".join(current)))
            title, current = row.strip(), [row]
        elif current:
            current.append(row)
    if current:
        result.append((title, "".join(current)))
    return result


def freeze_data(root: Path, assets: dict) -> dict:
    from benchmark_lm8_convergence import NLL_PREVIOUS, canonical
    from transformers import AutoTokenizer

    inventory = read(NLL_PREVIOUS / "prior_data_inventory.json")
    excluded = set(inventory["canonical_titles"])
    prior_files = {Path(p) for p in inventory["token_files_sha256"]}
    manifests = sorted(
        set(ARTIFACTS.glob("*/data_manifest.json"))
        | set(ARTIFACTS.glob("*/*/data_manifest.json"))
    )

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ("title", "canonical_title") and isinstance(v, str):
                    excluded.add(canonical(v))
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)

    for p in manifests:
        if root not in p.parents:
            visit(read(p))
    for p in ARTIFACTS.glob("*/assets*/*.bin"):
        if root not in p.parents and p.stat().st_size <= 65536 and "rope" not in p.name:
            prior_files.add(p)
    prior_hashes, spans = {}, set()
    for p in sorted(prior_files):
        if p.is_file():
            prior_hashes[str(p)] = digest(p)
            ids = np.fromfile(p, dtype="<i4")
            spans.update(ids[i : i + 16].tobytes() for i in range(len(ids) - 15))
    source = Path(
        read(NLL_PREVIOUS / "data_manifest.json")["datasets"]["validation"]["path"]
    ).parent
    tokenizer = AutoTokenizer.from_pretrained(
        assets["checkpoint_dir"], local_files_only=True
    )
    selected = {"search": [], "validation": [], "test": []}
    source_hashes = {}
    # Official train supplies search/selection calibration. Official validation
    # supplies this experiment's independent test, never model/mask selection.
    for split, count in (("train", 16), ("validation", 8)):
        p = source / f"{split}-00000-of-00001.parquet"
        source_hashes[str(p)] = digest(p)
        items = documents(p)
        salt = f"StructuredK8-search-v2-20261020:{split}:"
        items.sort(key=lambda x: hashlib.sha256((salt + x[1]).encode()).hexdigest())
        accepted = 0
        for title, text in items:
            if canonical(title) in excluded:
                continue
            ids = np.asarray(
                tokenizer.encode(text, add_special_tokens=False), dtype="<i4"
            )
            if len(ids) < 1024:
                continue
            grams = {ids[i : i + 16].tobytes() for i in range(len(ids) - 15)}
            if grams & spans:
                continue
            role = (
                "test"
                if split == "validation"
                else ("search" if accepted < 12 else "validation")
            )
            offset = min(256, len(ids) - 1024)
            name = f"{role}_{len(selected[role]):02d}.bin"
            with (root / "assets" / name).open("xb") as f:
                f.write(ids[offset : offset + 1024].tobytes())
            sha = digest(root / "assets" / name)
            assets["sha256"][name] = sha
            selected[role].append(
                {
                    "title": title,
                    "canonical_title": canonical(title),
                    "official_split": split,
                    "file": name,
                    "offset": offset,
                    "tokens": 1024,
                    "scored_tokens": 1023,
                    "tokens_sha256": sha,
                    "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
            excluded.add(canonical(title))
            spans.update(grams)
            accepted += 1
            if accepted == count:
                break
        if accepted != count:
            raise ValueError(
                f"Only {accepted}/{count} unused {split} documents; do not change protocol silently"
            )
    return {
        "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
        "revision": REVISION,
        "windows": selected,
        "parquet_sha256": source_hashes,
        "prior_token_files_sha256": prior_hashes,
        "prior_manifest_sha256": {
            str(p): digest(p) for p in manifests if root not in p.parents
        },
        "selection": "Salted content hash ascending; offset min(256,length-1024); no model outcome",
        "overlap_16grams": 0,
        "legacy_reference": {
            n: assets["sha256"][n] for n in assets["wikitext_windows"]
        },
        "limits": "Independent test for this allocation experiment uses unused documents from the dataset's official validation partition. Independence is from recorded allocation/rotation experiments, not from model pretraining or unknown original activation calibration.",
    }


def prepare(root: Path, manifest: Path | None) -> None:
    if (root / "protocol.json").exists():
        raise FileExistsError(root / "protocol.json")
    assert read(root / "replay_preflight_v2.json")["passed"]
    for sub in ("assets", "plans", "reports", "offline", "builds"):
        (root / sub).mkdir(exist_ok=False)
    (root / "logs").mkdir(exist_ok=True)
    assets = copy.deepcopy(read(ARTIFACTS / "device_assets_cl1024/assets.json"))
    for n, sha in assets["sha256"].items():
        p = ARTIFACTS / "device_assets_cl1024" / n
        assert digest(p) == sha
        symlink(p, root / "assets" / n)
    if manifest:
        data = read(manifest)
        for role, expected in (("search", 12), ("validation", 4), ("test", 8)):
            assert len(data["windows"][role]) == expected
            for row in data["windows"][role]:
                p = manifest.parent / "assets" / row["file"]
                assert digest(p) == row["tokens_sha256"] and p.stat().st_size == 4096
                symlink(p, root / "assets" / row["file"])
                assets["sha256"][row["file"]] = row["tokens_sha256"]
    else:
        data = freeze_data(root, assets)
    save_new(root / "assets/assets.json", assets)
    save_new(root / "data_manifest.json", data)
    for group, mask in STATIC.items():
        save_new(root / "plans" / f"{group}.json", structured_allocation_plan(mask))
    files = [
        root / "data_manifest.json",
        root / "assets/assets.json",
        LATEST / "native/manifest.json",
        OLD / "runner/qnn-llm-runner",
        OLD / "offline/prepared_w4a16.pt",
    ]
    files += [root / "assets" / n for n in assets["sha256"]]
    native = read(LATEST / "native/manifest.json")
    files += [Path(v["path"]) for v in native["libraries"].values()]
    for source in (LATEST / "structured", OLD / "early", OLD / "boundary"):
        files += [source / "convert_report.json", *source.glob("part*_of_4.bin")]
    protocol = {
        "branch": "exp/structured-k8-layer-search-v2",
        "base_branch": "exp/uniform-k8-baseline",
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": "Qwen3-1.7B W4A16",
        "context_length": 1024,
        "device": "S26 Ultra",
        "native": str(LATEST / "native"),
        "runner": str(OLD / "runner/qnn-llm-runner"),
        "split": str(ARTIFACTS / "qwen3_1_7b_w4a16_split"),
        "assets": str(root / "assets"),
        "assets_manifest": assets,
        "static_masks": STATIC,
        "previous_profile_hashes": {n: c.config_hash() for n, c in PROFILES.items()},
        "frozen_files": {str(p): digest(p) for p in files},
        "search": {
            "starts": list(STATIC),
            "objective": "sum full-model NLL / sum scored tokens",
            "search_documents": 12,
            "validation_documents": 4,
            "legacy": {"max_steps": 3, "shortlist_per_direction": 4},
            "multi_start_greedy": "Same legacy 3-step/4x4 rule independently at all six starts; Odd reused, not rerun",
            "beam": {
                "width": 2,
                "max_steps": 8,
                "swap_count_per_parent": 196,
                "screen_documents": [0, 1],
                "screen_tokens": 256,
                "full_shortlist_per_parent": 4,
                "retain_incumbent": True,
                "allow_nonmonotonic_frontier": True,
                "tie": "NLL then ascending layer-mask integer",
            },
            "epsilon": 1e-6,
            "budget": "Identical hyperparameters and document sets for each start within a method. Methods have unequal total budgets; report logical unique mask-document evaluations, scored tokens, physical misses, GPU time and incremental wall time, not an unqualified equal-compute claim.",
            "nominees": "Best full-search-12 NLL final mask per method: legacy, multi-start greedy, multi-start beam. Deduplicate; fill to two from other search finalists if available, never more than three.",
            "selection": "After nominees fixed, use actual HTP NLL on four validation documents to choose among nominees and Uniform-Odd; ties within 1e-6 retain Odd, otherwise mask integer. HTP search-12 scores are transfer diagnostics, not final selection.",
            "local_certificate": "After final mask freezes, evaluate all 196 swaps on full offline search-12. Report failure if any improves; do not silently optimize further or claim HTP local optimality.",
            "cache": "SQLite immutable rows keyed by evaluator identity, config/mask, document hash and length. Shared cache across starts/methods; independent logical evaluation counts prevent order-dependent cost claims.",
        },
        "evaluation": {
            "groups": [
                "uniform_odd",
                "early",
                "boundary",
                "legacy_greedy",
                "multi_greedy",
                "multi_beam",
                "k6_v2",
            ],
            "independent_test_documents": 8,
            "targets_per_document": 1023,
            "legacy_windows": "Four previous WikiText windows, reference only after selection",
            "performance_runs": 3,
            "prompt_tokens": 897,
            "generated_tokens": 128,
            "order": "Unique physical groups: forward, reverse, rotate-left-by-three; aliases reuse identical measurements",
            "statistics": "Mean/sample SD/min/max; no superiority claim from n=3 alone",
        },
        "fixed": "Structured K8x14, LM K4x14, all LM V2; arithmetic HVX K8, Dense QR K42/V542, norm correction, FP16 scale, current/past compressed, tile256; 30277632 KV bytes. Uniform K6/V2 is the explicitly different-bit reference.",
        "limits": "Offline surrogate is not HTP bit-exact. CUDA Graph must match shape-matched eager logits exactly; 256-token coarse scores are not full-1024 quality. No global-optimality claim. Original encoder scale failures remain tracked.",
    }
    save_new(root / "protocol.json", protocol)
    freeze_weights(root)
    print(
        "Frozen",
        {k: [d["title"] for d in v] for k, v in data["windows"].items()},
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-manifest", type=Path)
    args = p.parse_args()
    prepare(args.root.resolve(), args.data_manifest)
