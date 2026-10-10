# SPDX-License-Identifier: BSD-3-Clause
"""Frozen data, candidates and document-level statistics for NLL seed selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_key_rotation import read
from benchmark_rotation_followup import identity as followup_identity
from rotation_data import DATA_REVISION, digest, write_json
from transformers import AutoTokenizer

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.reference import DenseQRRotation
from qai_hub_models.models.templates.llm.turboquant.rotation_artifact import (
    matrix_digest,
    save_rotation,
    with_key_rotation,
)

BASE = Path("/mnt/d/ai-hub-models/binaries/turboquant")
FOLLOW = BASE / "k_rotation_followup_20261009"
PREVIOUS = BASE / "k_rotation_mse_nll_20261009"
SEEDS = list(range(42, 50))
PROTOCOL = {
    "model": "Qwen3-1.7B W4A16",
    "seeds": SEEDS,
    "change": "One shared K and corresponding Query Dense QR matrix only; V unchanged",
    "fixed": "LM tree/codebook, K4/V4, QJL-off, compressed current KV, norm correction/FP16 effective scale, Native LUT, Dense MatMul, tile256, weights/calibration",
    "context": 1024,
    "scored_tokens": 1023,
    "validation_documents": 16,
    "heldout_documents": 16,
    "data_seed": 2026100917,
    "data_revision": DATA_REVISION,
    "exclusion": "All prior manifest titles (canonicalized), plus reject whole articles sharing any contiguous 16-token sequence with saved old token inputs; known local experiments, not a pretraining-contamination guarantee",
    "selection": "Minimum sum(NLL_sum)/sum(scored_tokens) on all validation documents; exact numerical ties choose lowest seed; never MSE",
    "score_repeats": 1,
    "validation_order": "Rotate ascending seed list by document index modulo 8",
    "folds": 4,
    "fold_seed": 2026100918,
    "fold_rule": "Permute 16 documents once, split into four equal groups; select on other 12, evaluate excluded 4, no device reruns",
    "bootstrap_replicates": 20000,
    "bootstrap_seed": 2026100920,
    "bootstrap": "Paired document resampling, same sampled indices for candidate and baseline; percentile 2.5/97.5 linear quantiles; S fixed before heldout, no reselection inside bootstrap",
    "performance_repeats": 3,
    "performance": "CL1024, short35+128 and long897+128; unique A/B/S order, cyclic shifts each repeat; deduplicate aliases",
    "primary": "Heldout S-A mean NLL; CI containing zero means uncertain improvement",
    "secondary": "Heldout S-B and per-document consistency; no claim NLL selection universally beats earlier MSE selection",
    "restrictions": "No learning, per-layer selection, default promotion, candidate expansion, 4B, or post-heldout rule/data changes",
    "max_build_workers": 3,
}


def canonical_title(title: str) -> str:
    return " ".join(title.strip().strip("=").split()).casefold()


def grams(tokens: np.ndarray, width: int = 16) -> set[bytes]:
    raw = np.asarray(tokens, dtype="<i4").ravel().tobytes()
    return {raw[i : i + 4 * width] for i in range(0, len(raw) - 4 * width + 1, 4)}


def documents(path: Path) -> list[tuple[str, str]]:
    result, title, lines = [], None, []
    for line in pd.read_parquet(path)["text"]:
        if re.fullmatch(r"\s*= [^=]+ =\s*", line):
            if title is not None:
                result.append((title, "\n".join(lines)))
            title, lines = line.strip(), [line]
        elif title is not None:
            lines.append(line)
    if title is not None:
        result.append((title, "\n".join(lines)))
    return result


def prior_inputs(root: Path) -> tuple[dict, set[str], set[bytes]]:
    paths = [
        Path(p)
        for p in subprocess.check_output(
            ["rg", "--files", "--hidden", str(BASE)], text=True
        ).splitlines()
        if not Path(p).is_relative_to(root)
    ]
    titles, overlap, manifests, token_files = set(), set(), {}, {}

    def collect(value: object) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("title"), str):
                titles.add(canonical_title(value["title"]))
            for entry in value.values():
                collect(entry)
        elif isinstance(value, list):
            for entry in value:
                collect(entry)

    seen = set()
    for path in paths:
        if path.name not in ("data_manifest.json", "assets.json", "manifest.json"):
            continue
        data = read(path)
        collect(data)
        manifests[str(path)] = digest(path)
        if path.name == "assets.json":
            for name, expected in data.get("sha256", {}).items():
                if not name.endswith(".bin") or name.startswith("rope"):
                    continue
                token_path = path.parent / name
                if not token_path.is_file() or digest(token_path) != expected:
                    raise ValueError(f"Prior input missing/changed: {token_path}")
                token_files[str(token_path)] = expected
                if expected not in seen:
                    overlap.update(grams(np.fromfile(token_path, dtype="<i4")))
                    seen.add(expected)
    # Includes old 128 training + 16 validation + 16 test article windows even
    # when no device asset file was produced for a training sample.
    for path in paths:
        if path.name != "tokens.npz":
            continue
        checksum = digest(path)
        token_files[str(path)] = checksum
        if checksum in seen:
            continue
        with np.load(path) as data:
            for key in data.files:
                values = data[key]
                if np.issubdtype(values.dtype, np.integer):
                    for row in values.reshape(-1, values.shape[-1]):
                        overlap.update(grams(row))
        seen.add(checksum)
    return (
        {
            "manifest_sha256": manifests,
            "token_files_sha256": token_files,
            "canonical_titles": sorted(titles),
            "unique_token_payloads": len(seen),
            "old_16grams": len(overlap),
        },
        titles,
        overlap,
    )


def fold_assignment(count: int) -> list[int]:
    if count % 4 or count < 4:
        raise ValueError("Four equal document folds required")
    assignment = np.empty(count, dtype=int)
    order = np.random.default_rng(PROTOCOL["fold_seed"]).permutation(count)
    for fold, indices in enumerate(np.split(order, 4)):
        assignment[indices] = fold
    return assignment.tolist()


def freeze(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    for name in ("reports", "rotations", "builds", "runner"):
        (root / name).mkdir()
    write_json(root / "protocol.json", PROTOCOL)
    previous = read(FOLLOW / "reports/experiment.json")
    if followup_identity(FOLLOW) != previous:
        raise ValueError("Original A/B binaries/settings/device changed")
    inventory, excluded, overlap = prior_inputs(root)
    write_json(root / "prior_data_inventory.json", inventory)
    original = read(FOLLOW / "data_manifest.json")
    base_assets = read(FOLLOW / "assets/assets.json")
    tokenizer = AutoTokenizer.from_pretrained(
        base_assets["checkpoint_dir"], local_files_only=True
    )
    rng = np.random.default_rng(PROTOCOL["data_seed"])
    windows, statistics, datasets = {}, {}, {}
    for role, split in (("validation", "validation"), ("heldout", "test")):
        dataset = original["datasets"][split]
        path = Path(dataset["path"])
        if digest(path) != dataset["sha256"]:
            raise ValueError("Pinned dataset changed")
        datasets[role] = dataset
        articles = documents(path)
        eligible, reasons = [], {"old_title": 0, "too_short": 0, "old_token_overlap": 0}
        for title, text in articles:
            canonical = canonical_title(title)
            if canonical in excluded:
                reasons["old_title"] += 1
                continue
            ids = np.asarray(
                tokenizer(text, add_special_tokens=False).input_ids, dtype="<i4"
            )
            if len(ids) < 1024:
                reasons["too_short"] += 1
                continue
            if not grams(ids).isdisjoint(overlap):
                reasons["old_token_overlap"] += 1
                continue
            eligible.append((title, text, ids))
        count = PROTOCOL[f"{role}_documents"]
        if len(eligible) < count:
            raise ValueError(
                f"Only {len(eligible)} unused {role} documents; protocol adjustment needed before any measurements"
            )
        out = root / f"assets_{role}"
        out.mkdir()
        entries = []
        for index in rng.permutation(len(eligible))[:count]:
            title, text, ids = eligible[index]
            offset = int(rng.integers(0, len(ids) - 1024 + 1))
            name = f"{role}_{len(entries):02d}.bin"
            with (out / name).open("xb") as stream:
                stream.write(ids[offset : offset + 1024].tobytes())
            entries.append(
                {
                    "file": name,
                    "title": title,
                    "canonical_title": canonical_title(title),
                    "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "offset": offset,
                    "tokens": 1024,
                    "tokens_sha256": digest(out / name),
                }
            )
            excluded.add(canonical_title(title))
        (out / base_assets["rope"]).symlink_to(
            (FOLLOW / "assets" / base_assets["rope"]).resolve()
        )
        assets = dict(base_assets)
        assets.update(
            {
                "prompt_ids": entries[0]["file"],
                "prompt_tokens": 1024,
                "prompt": f"Frozen {role} scoring windows",
                "wikitext_windows": [w["file"] for w in entries],
                "sha256": {p.name: digest(p) for p in sorted(out.glob("*.bin"))},
            }
        )
        assets.pop("boundary_prompt", None)
        write_json(out / "assets.json", assets)
        windows[role] = entries
        statistics[role] = {
            "total_articles": len(articles),
            "eligible": len(eligible),
            "excluded": reasons,
        }
    # Known old performance prompts, isolated from both quality sets.
    perf = root / "assets_performance"
    perf.mkdir()
    perf_assets = dict(base_assets)
    names = [base_assets[k] for k in ("rope", "prompt_ids", "boundary_prompt")]
    for name in names:
        (perf / name).symlink_to((FOLLOW / "assets" / name).resolve())
    perf_assets["sha256"] = {n: digest(perf / n) for n in names}
    perf_assets["wikitext_windows"] = []
    write_json(perf / "assets.json", perf_assets)
    write_json(
        root / "data_manifest.json",
        {
            "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
            "revision": DATA_REVISION,
            "datasets": datasets,
            "windows": windows,
            "statistics": statistics,
            "validation_folds": fold_assignment(len(windows["validation"])),
            "exclusion_inventory_sha256": digest(root / "prior_data_inventory.json"),
            "heldout_status": "Never scored before selection; used for final evaluation only; after this experiment no longer independent for future tuning",
        },
    )
    candidates = {}
    for seed in SEEDS:
        path = root / f"rotations/seed{seed}.json"
        matrix = DenseQRRotation(seed, 128).matrix().astype("<f4")
        save_rotation(
            path,
            matrix,
            {
                "seed": seed,
                "numpy": np.__version__,
                "generation": "default_rng(seed).standard_normal((128,128)), FP64 np.linalg.qr, positive R diagonal, flip first Q column if det<0, cast FP32; row_vector @ R.T",
            },
        )
        config = (
            get_profile("k4_v4_scaled")
            if seed == 42
            else with_key_rotation(get_profile("k4_v4_scaled"), path)
        )
        source = FOLLOW / ("A" if seed == 42 else "B") if seed in (42, 48) else None
        if (
            source
            and read(source / "convert_report.json")["config_hash"]
            != config.config_hash()
        ):
            raise ValueError("Reused candidate matrix/config mismatch")
        candidates[str(seed)] = {
            "seed": seed,
            "matrix_f32_sha256": matrix_digest(matrix),
            "matrix_fp16_sha256": hashlib.sha256(
                matrix.astype("<f2").tobytes()
            ).hexdigest(),
            "orthogonality_spectral": float(
                np.linalg.norm(matrix.astype(float).T @ matrix - np.eye(128), 2)
            ),
            "artifact_sha256": digest(path),
            "config_hash": config.config_hash(),
            "source_bundle": str(source) if source else None,
        }
        if source:
            dest = root / f"seed{seed}"
            dest.mkdir()
            for item in source.iterdir():
                if (
                    item.is_file()
                    and item.name != "runtime_manifest.json"
                    and not item.name.startswith("device_push_")
                ):
                    (dest / item.name).symlink_to(item.resolve())
    write_json(root / "candidates.json", candidates)
    (root / "runner/qnn-llm-runner").symlink_to(
        (FOLLOW / "runner/qnn-llm-runner").resolve()
    )
    preserved = {}
    for old in (FOLLOW, PREVIOUS, BASE / "k_rotation_quality_20261006"):
        for pattern in ("*.json", "reports/*.json", "rotations/*.json"):
            for path in old.glob(pattern):
                preserved[str(path)] = digest(path)
    for group in ("A", "B"):
        for name in (
            "convert_report.json",
            *(f"part{i}_of_4.bin" for i in range(1, 5)),
        ):
            path = FOLLOW / group / name
            preserved[str(path)] = digest(path)
    write_json(
        root / "source_identity.json",
        {
            "sha256": preserved,
            "device": previous["device"],
            "runner_sha256": previous["runner_sha256"],
            "source_weights_calibration": read(FOLLOW / "capture_identity.json")[
                "source_files_sha256"
            ],
        },
    )
    hashes = {
        name: digest(root / name)
        for name in (
            "protocol.json",
            "prior_data_inventory.json",
            "data_manifest.json",
            "candidates.json",
            "source_identity.json",
            "assets_validation/assets.json",
            "assets_heldout/assets.json",
            "assets_performance/assets.json",
        )
    }
    write_json(root / "freeze_identity.json", hashes)
    print("FROZEN", statistics, "8 candidates, 16 validation + 16 heldout", flush=True)


def verify_frozen(root: Path, sources: bool = False) -> None:
    if read(root / "protocol.json") != PROTOCOL:
        raise ValueError("Predeclared protocol changed")
    for name, checksum in read(root / "freeze_identity.json").items():
        if digest(root / name) != checksum:
            raise ValueError(f"Frozen file changed: {name}")
    for seed, data in read(root / "candidates.json").items():
        if digest(root / f"rotations/seed{seed}.json") != data["artifact_sha256"]:
            raise ValueError("Candidate changed")
    for role in ("validation", "heldout", "performance"):
        out = root / f"assets_{role}"
        for name, checksum in read(out / "assets.json")["sha256"].items():
            if digest(out / name) != checksum:
                raise ValueError("Input changed")
    if sources:
        original = read(root / "source_identity.json")
        for entries in (original["sha256"], original["source_weights_calibration"]):
            for name, checksum in entries.items():
                if digest(name) != checksum:
                    raise ValueError(f"Original artifact changed: {name}")


def replay(root: Path, old: Path) -> None:
    """Explicit reproduction of a saved cohort, never a fresh independent test."""
    verify_frozen(old, sources=True)
    audit = read(old / "reports/graph_audit.json")
    if not audit["passed"]:
        raise ValueError("Replay requires previously audited candidates")
    root.mkdir(parents=True, exist_ok=False)
    for name in ("reports", "rotations", "builds", "runner"):
        (root / name).mkdir()
    for name in ("protocol.json", "prior_data_inventory.json"):
        (root / name).symlink_to((old / name).resolve())
    data = read(old / "data_manifest.json")
    data["heldout_status"] = (
        "Reproduction of previously observed documents; NOT a new independent test"
    )
    data["replayed_from_manifest_sha256"] = digest(old / "data_manifest.json")
    write_json(root / "data_manifest.json", data)
    for role in ("validation", "heldout", "performance"):
        dest = root / f"assets_{role}"
        dest.mkdir()
        for path in (old / f"assets_{role}").iterdir():
            if path.is_file():
                (dest / path.name).symlink_to(path.resolve())
    candidates = read(old / "candidates.json")
    sources = read(old / "source_identity.json")
    for seed in SEEDS:
        source = old / f"seed{seed}"
        for name, checksum in audit["candidates"][str(seed)]["bins_sha256"].items():
            if digest(source / name) != checksum:
                raise ValueError("Replay binary changed")
            sources["sha256"][str(source / name)] = checksum
        candidates[str(seed)]["source_bundle"] = str(source)
        dest = root / f"seed{seed}"
        dest.mkdir()
        for path in source.iterdir():
            if (
                path.is_file()
                and path.name != "runtime_manifest.json"
                and not path.name.startswith("device_push_")
            ):
                (dest / path.name).symlink_to(path.resolve())
        (root / f"rotations/seed{seed}.json").symlink_to(
            (old / f"rotations/seed{seed}.json").resolve()
        )
    (root / "runner/qnn-llm-runner").symlink_to(
        (old / "runner/qnn-llm-runner").resolve()
    )
    write_json(root / "candidates.json", candidates)
    write_json(root / "source_identity.json", sources)
    write_json(
        root / "freeze_identity.json",
        {
            name: digest(root / name)
            for name in (
                "protocol.json",
                "prior_data_inventory.json",
                "data_manifest.json",
                "candidates.json",
                "source_identity.json",
                "assets_validation/assets.json",
                "assets_heldout/assets.json",
                "assets_performance/assets.json",
            )
        },
    )
    write_json(
        root / "replay.json", {"source": str(old), "independent_new_test": False}
    )
    print(
        "REPLAY FROZEN: same saved cohort/candidates, not a new independent test",
        flush=True,
    )


def choose_seed(nll: np.ndarray, counts: np.ndarray, seeds: list[int]) -> int:
    nll, counts = np.asarray(nll, float), np.asarray(counts, float)
    if (
        nll.ndim != 2
        or nll.shape != (len(seeds), len(counts))
        or not len(counts)
        or not np.isfinite(nll).all()
        or not np.isfinite(counts).all()
        or np.any(nll < 0)
        or np.any(counts <= 0)
    ):
        raise ValueError("Invalid candidate/document NLL table")
    means = nll.sum(axis=1) / counts.sum()
    return min(zip(means.tolist(), seeds, strict=True))[1]


def fold_analysis(
    nll: np.ndarray, counts: np.ndarray, seeds: list[int], folds: list[int]
) -> dict:
    if sorted(set(folds)) != list(range(4)) or len(folds) != nll.shape[1]:
        raise ValueError("Invalid document folds")
    assignments = np.asarray(folds)
    result, selected, out_of_fold = [], [], np.empty(nll.shape[1])
    for fold in range(4):
        test = np.flatnonzero(assignments == fold)
        train = np.flatnonzero(assignments != fold)
        seed = choose_seed(nll[:, train], counts[train], seeds)
        selected.append(seed)
        values = nll[seeds.index(seed), test]
        out_of_fold[test] = values
        result.append(
            {
                "fold": fold,
                "train_documents": train.tolist(),
                "excluded_documents": test.tolist(),
                "selected_seed": seed,
                "mean_nll": float(values.sum() / counts[test].sum()),
                "delta_vs_A": float(
                    (values - nll[seeds.index(42), test]).sum() / counts[test].sum()
                ),
                "delta_vs_B": float(
                    (values - nll[seeds.index(48), test]).sum() / counts[test].sum()
                ),
            }
        )
    return {
        "folds": result,
        "selection_counts": {str(s): selected.count(s) for s in seeds},
        "out_of_fold_mean_nll": float(out_of_fold.sum() / counts.sum()),
        "out_of_fold_ppl": math.exp(float(out_of_fold.sum() / counts.sum())),
        "out_of_fold_delta_vs_A": float(
            (out_of_fold - nll[seeds.index(42)]).sum() / counts.sum()
        ),
        "out_of_fold_delta_vs_B": float(
            (out_of_fold - nll[seeds.index(48)]).sum() / counts.sum()
        ),
        "note": "Four different seed selections; not the heldout result of full-validation S",
    }


def paired_bootstrap(
    candidate: np.ndarray, baseline: np.ndarray, counts: np.ndarray
) -> dict:
    candidate, baseline, counts = (
        np.asarray(x, float) for x in (candidate, baseline, counts)
    )
    if (
        candidate.shape != baseline.shape
        or candidate.shape != counts.shape
        or candidate.ndim != 1
        or not len(counts)
        or np.any(counts <= 0)
        or not all(np.isfinite(x).all() for x in (candidate, baseline, counts))
        or np.any(candidate < 0)
        or np.any(baseline < 0)
    ):
        raise ValueError("Invalid paired documents")
    indices = np.random.default_rng(PROTOCOL["bootstrap_seed"]).integers(
        0, len(counts), size=(PROTOCOL["bootstrap_replicates"], len(counts))
    )
    difference = candidate - baseline
    resampled = difference[indices].sum(axis=1) / counts[indices].sum(axis=1)
    lower, upper = np.quantile(resampled, [0.025, 0.975], method="linear")
    delta = float(difference.sum() / counts.sum())
    per_document = difference / counts
    gains = np.maximum(-difference, 0)
    return {
        "mean_nll_delta": delta,
        "ppl_relative_change": math.expm1(delta),
        "ci95": [float(lower), float(upper)],
        "ci_contains_zero": bool(lower <= 0 <= upper),
        "improved_documents": int(np.sum(difference < 0)),
        "worse_documents": int(np.sum(difference > 0)),
        "equal_documents": int(np.sum(difference == 0)),
        "document_mean_nll_deltas": per_document.tolist(),
        "largest_improvement_document": int(np.argmin(per_document))
        if np.any(difference < 0)
        else None,
        "largest_regression_document": int(np.argmax(per_document))
        if np.any(difference > 0)
        else None,
        "top3_share_of_positive_nll_gains": float(
            np.sort(gains)[-3:].sum() / gains.sum()
        )
        if gains.sum()
        else None,
        "concentration_denominator": "Sum of positive NLL reductions only, not net benefit",
        "point_estimate_improves": delta < 0,
        "ci_entirely_below_zero": bool(upper < 0),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "verify", "replay"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replay-source", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.root.resolve())
    elif args.stage == "replay":
        if args.replay_source is None:
            parser.error("replay requires --replay-source")
        replay(args.root.resolve(), args.replay_source.resolve())
    else:
        verify_frozen(args.root.resolve(), sources=True)


if __name__ == "__main__":
    main()
