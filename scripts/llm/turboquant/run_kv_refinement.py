# SPDX-License-Identifier: BSD-3-Clause
"""Score fixed local equal-byte KV exchanges and freeze the calibration winner."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from allocation_offline_model import Model
from compare_cross_model_budgets import digest, read, sources
from cross_model_plans import MODELS, _freeze_json

from qai_hub_models.models.templates.llm.turboquant.config import get_profile


def load_plans(root: Path, model_id: str) -> list[dict]:
    manifest = read(root / "selection_protocol.json")
    plans = []
    for name, expected in sorted(manifest["plan_files_sha256"].items()):
        path = root / name
        if path.parent.name != model_id:
            continue
        if digest(path) != expected:
            raise ValueError(f"Plan changed: {path}")
        plan = read(path)
        cfg = get_profile(plan["profile"], layer_plan=plan)
        if plan["k8_layers"] != [
            i for i, (k, _) in enumerate(cfg.layer_bits) if k == 8
        ] or plan["v4_layers"] != [
            i for i, (_, v) in enumerate(cfg.layer_bits) if v == 4
        ]:
            raise ValueError(f"Scoring masks disagree with frozen byte layout: {path}")
        plans.append(plan)
    if len({p["config_hash"] for p in plans}) != len(plans) or "anchor" not in {
        p["plan_id"] for p in plans
    }:
        raise ValueError(f"Duplicate masks or missing anchor for {model_id}")
    return plans


def validate_data(root: Path) -> dict:
    windows = read(root / "data_manifest.json")["windows"]
    if len(windows["calibration"]) != 4 or len(windows["heldout"]) != 8:
        raise ValueError(
            "Expected four calibration and eight independent heldout windows"
        )
    rows = windows["calibration"] + windows["heldout"]
    for field in ("document_sha256", "tokens_sha256", "file"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"Overlapping documents/windows: {field}")
    for row in rows:
        path = root / "assets" / row["file"]
        if path.stat().st_size != 4096 or digest(path) != row["tokens_sha256"]:
            raise ValueError(f"Changed token file: {path}")
    return windows


def score_model(
    root: Path, archive: Path, model_id: str, protocol_hash: str, windows: dict
) -> dict:
    plans = load_plans(root, model_id)
    bundle, assets, split = sources(archive, model_id)
    prior_study = archive / "cross_model_pu_budget_20261120"
    parent = prior_study / "scorer" / (model_id + ".json")
    previous = read(parent)
    if (
        previous["scorer"]
        != digest(Path(__file__).with_name("allocation_offline_model.py"))
        or previous["split_manifest_sha256"] != digest(split / "split_manifest.json")
        or previous["input_assets"] != digest(assets / "assets.json")
    ):
        raise ValueError("W4A16 source identity differs from preceding study")
    for name, expected in previous["graphs_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Graph changed: {name}")
    asset_manifest = read(assets / "assets.json")
    rope = asset_manifest["rope"]
    if digest(assets / rope) != asset_manifest["sha256"][rope]:
        raise ValueError("Rotary table hash mismatch")
    anchor = next(p for p in plans if p["plan_id"] == "anchor")
    original_id = "k6_v2_early" if model_id == "qwen3_8b" else "k5_v3_early"
    original_plan = read(prior_study / "plans" / model_id / (original_id + ".json"))
    if (
        original_plan["bits"] != anchor["bits"]
        or original_plan["config_hash"] != anchor["config_hash"]
    ):
        raise ValueError(f"{model_id} anchor differs from preceding comparison")

    identity = _freeze_json(
        root / "scorer" / (model_id + ".json"),
        {
            "model": model_id,
            "previous_scorer_sha256": digest(parent),
            "protocol_sha256": protocol_hash,
            "backend": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        },
    )
    model = None

    def evaluate(plan: dict, split_name: str) -> dict:
        nonlocal model
        documents = {}
        total_loss = 0.0
        for row in windows[split_name]:
            dest = (
                root
                / "scores"
                / model_id
                / split_name
                / plan["plan_id"]
                / (Path(row["file"]).stem + ".json")
            )
            key = dict(
                config_hash=plan["config_hash"],
                scorer_sha256=identity,
                tokens_sha256=row["tokens_sha256"],
                window=row["file"],
                split=split_name,
            )
            if dest.exists():
                result = read(dest)
            else:
                if model is None:
                    model = Model(
                        bundle,
                        assets,
                        device="cuda",
                        split_dir=split,
                        stream_parts=True,
                        stream_weights=True,
                    )
                    if sorted(i for p in model.parts for i in p.macros) != list(
                        range(len(plan["bits"]))
                    ):
                        raise ValueError("Replay graph layer count mismatch")
                    for part in model.parts:
                        part.to("cuda")
                    model.stream_parts = False
                    control = read(
                        prior_study
                        / "scores"
                        / model_id
                        / original_id
                        / "wikitext_w0.json"
                    )
                    control_file = assets / control["window"]
                    if (
                        digest(control_file) != control["token_hash"]
                        or control["config_hash"] != anchor["config_hash"]
                    ):
                        raise ValueError("Historical control input or mask mismatch")
                    repeat, logits = model.score(
                        np.fromfile(control_file, dtype="<i4"),
                        anchor["k8_layers"],
                        v4_layers=anchor["v4_layers"],
                    )
                    del logits
                    delta = repeat["mean_nll"] - control["mean_nll"]
                    if abs(delta) > 1e-5:
                        raise ValueError(f"Historical anchor NLL differs by {delta}")
                    _freeze_json(
                        root / "controls" / (model_id + ".json"),
                        {
                            "config_hash": anchor["config_hash"],
                            "tokens_sha256": control["token_hash"],
                            "reference_mean_nll": control["mean_nll"],
                            "delta_mean_nll": delta,
                        },
                    )
                tokens = np.fromfile(root / "assets" / row["file"], dtype="<i4")
                stats, logits = model.score(
                    tokens, plan["k8_layers"], v4_layers=plan["v4_layers"]
                )
                del logits
                result = {**key, **stats}
                _freeze_json(dest, result)
            if (
                any(result[k] != v for k, v in key.items())
                or result["scored_tokens"] != 1023
                or not math.isfinite(result["nll_sum"])
            ):
                raise ValueError(f"Invalid raw NLL: {dest}")
            loss = result["nll_sum"]
            documents[row["file"]] = loss / 1023
            total_loss += loss
        mean = total_loss / (len(documents) * 1023)
        print(model_id, split_name, plan["plan_id"], f"NLL={mean:.7f}", flush=True)
        return {
            **{
                k: plan[k]
                for k in (
                    "plan_id",
                    "category",
                    "k8_layers",
                    "v4_layers",
                    "config_hash",
                )
            },
            "nll": mean,
            "ppl": math.exp(mean),
            "documents": documents,
        }

    candidates = [evaluate(p, "calibration") for p in plans]
    selected = min(
        candidates, key=lambda r: (r["nll"], r["plan_id"] != "anchor", r["plan_id"])
    )["plan_id"]
    controls = {
        category: min(
            (r for r in candidates if r["category"] == category),
            key=lambda r: (r["nll"], r["plan_id"]),
        )["plan_id"]
        for category in ("budget_only", "placement_only")
    }
    _freeze_json(
        root / "selection" / (model_id + ".json"),
        {
            "scorer_sha256": identity,
            "anchor": "anchor",
            "selected": selected,
            "category_controls": controls,
            "calibration": candidates,
        },
    )
    final_plans = [
        p for p in plans if p["plan_id"] in {"anchor", selected, *controls.values()}
    ]
    return {
        "anchor": "anchor",
        "candidates": candidates,
        "selected": selected,
        "category_controls": controls,
        "heldout": [evaluate(p, "heldout") for p in final_plans],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument(
        "--archive", type=Path, default=Path("/mnt/d/ai-hub-models/binaries/turboquant")
    )
    args = ap.parse_args()
    windows = validate_data(args.root)
    protocol = _freeze_json(
        args.root / "evaluation_protocol.json",
        {
            "data_sha256": digest(args.root / "data_manifest.json"),
            "plans_sha256": digest(args.root / "selection_protocol.json"),
            "runner_sha256": digest(Path(__file__)),
            "model_ids": list(MODELS),
            "selection": "Within each model minimize summed next-token NLL / scored tokens over 4 calibration documents. Exact ties prefer anchor then lexicographic candidate ID.",
            "test": "Primary comparison is frozen calibration winner versus anchor on eight heldout documents. Pre-inference amendment to the plan manifest: also evaluate the best budget_only and best placement_only calibration rows as secondary controls (same tie rule), deduplicating masks. Freeze all finalists before test. FP32 W4A16 replay with explicit FP16 codec rounding, CL1024.",
            "deployment": "Offline allocation study. No default promotion.",
        },
    )
    models = {}
    for model_id in MODELS:
        models[model_id] = score_model(
            args.root, args.archive, model_id, protocol, windows
        )
        _freeze_json(args.root / "completed" / (model_id + ".json"), models[model_id])
    _freeze_json(
        args.root / "summary.json", {"protocol_sha256": protocol, "models": models}
    )


if __name__ == "__main__":
    main()
