# SPDX-License-Identifier: BSD-3-Clause
"""Fixed-cohort offline W4A16 replay on the four historical WikiText windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from allocation_offline_model import Model
from cross_model_plans import MODELS, _freeze_json, freeze_plans

from qai_hub_models.models.templates.llm.turboquant.config import get_profile


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sources(archive: Path, model_id: str) -> tuple[Path, Path, Path]:
    if model_id == "qwen3_1_7b":
        previous = archive / "structured_k8_layer_search_v2_20261020"
        bundle = previous / "uniform_odd"
    else:
        previous = archive / "structured_k8_model_scaling_20261026" / model_id
        bundle = previous / "offline/graphs"
    return bundle, previous / "assets", Path(read(previous / "protocol.json")["split"])


def score_model(root: Path, archive: Path, model_id: str, device: str) -> None:
    bundle, assets, split = sources(archive, model_id)
    manifest = read(assets / "assets.json")
    names = manifest["wikitext_windows"]
    if names != [f"wikitext_w{i}.bin" for i in range(4)]:
        raise ValueError(f"Unexpected scored documents for {model_id}")
    reference_assets = read(sources(archive, MODELS[0])[1] / "assets.json")
    for name in [*names, manifest["rope"]]:
        if digest(assets / name) != manifest["sha256"][name]:
            raise ValueError(f"Input hash mismatch: {name}")
    for name in names:
        if manifest["sha256"][name] != reference_assets["sha256"][name]:
            raise ValueError("Models do not share the same token IDs")
    protocol = read(root / "selection_protocol.json")
    plans = []
    for plan_name, expected in sorted(protocol["plan_files_sha256"].items()):
        path = root / plan_name
        if path.parent.name != model_id:
            continue
        plan = read(path)
        if digest(path) != expected:
            raise ValueError(f"Frozen plan changed: {path}")
        cfg = get_profile(plan["profile"], layer_plan=plan)
        if cfg.config_hash() != plan["config_hash"]:
            raise ValueError("Wrong codec configuration")
        plans.append(plan)
    if len(plans) != 11:
        raise ValueError("Incomplete comparison cohort")

    parts = read(split / "split_manifest.json")["parts"]
    graphs = []
    for i in range(1, len(parts) + 1):
        entry = parts[f"part{i}_of_{len(parts)}"]
        graph = bundle / f"prompt_ar128_cl1024_{i}_of_{len(parts)}.onnx"
        if i == 1 and not graph.exists():
            graph = Path(entry["bundle_dir"]) / (entry["class"] + ".onnx")
        graphs.extend([graph, graph.with_suffix(".encodings")])
    identity = {
        "model_id": model_id,
        "scorer": digest(Path(__file__).with_name("allocation_offline_model.py")),
        "plans_sha256": digest(root / "selection_protocol.json"),
        "input_assets": digest(assets / "assets.json"),
        "split_manifest_sha256": digest(split / "split_manifest.json"),
        "graphs_sha256": {str(p): digest(p) for p in graphs},
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "description": "Offline W4A16 replay, FP32 arithmetic with explicit quantization and FP16 rounding. CL1024, 1023 scored targets/window.",
    }
    identity_sha = _freeze_json(root / "scorer" / (model_id + ".json"), identity)
    model = None
    for plan in plans:
        for name in names:
            file = (
                root
                / "scores"
                / model_id
                / plan["plan_id"]
                / (Path(name).stem + ".json")
            )
            request = dict(
                model_id=model_id,
                plan_id=plan["plan_id"],
                config_hash=plan["config_hash"],
                window=name,
                token_hash=manifest["sha256"][name],
                scorer_sha256=identity_sha,
            )
            if file.exists():
                row = read(file)
                if (
                    any(row[k] != v for k, v in request.items())
                    or row["scored_tokens"] != 1023
                    or not math.isfinite(row["nll_sum"])
                ):
                    raise ValueError(f"Invalid saved result: {file}")
                continue
            if model is None:
                model = Model(
                    bundle,
                    assets,
                    device=device,
                    split_dir=split,
                    stream_parts=device == "cuda",
                    stream_weights=device == "cuda",
                )
                layers = sorted(layer for part in model.parts for layer in part.macros)
                if layers != list(range(plan["num_layers"])):
                    raise ValueError(f"Wrong replay layers: {layers}")
            tokens = np.fromfile(assets / name, dtype="<i4")
            stats, logits = model.score(
                tokens, plan["k8_layers"], v4_layers=plan["v4_layers"]
            )
            del logits
            if stats["scored_tokens"] != 1023 or not math.isfinite(stats["nll_sum"]):
                raise ValueError("Invalid scored result")
            _freeze_json(file, {**request, **stats})
            print(model_id, plan["plan_id"], name, stats["mean_nll"], flush=True)


def summarize(root: Path) -> None:
    results = []
    for plan_path in sorted(
        read(root / "selection_protocol.json")["plan_files_sha256"]
    ):
        plan = read(root / plan_path)
        directory = root / "scores" / plan["model_id"] / plan["plan_id"]
        rows = [read(directory / f"wikitext_w{i}.json") for i in range(4)]
        n = sum(row["scored_tokens"] for row in rows)
        if n != 4092:
            raise ValueError("Wrong number of targets")
        nll = sum(row["nll_sum"] for row in rows) / n
        results.append(
            dict(
                model_id=plan["model_id"],
                plan_id=plan["plan_id"],
                config_hash=plan["config_hash"],
                scored_tokens=n,
                mean_nll=nll,
                ppl=math.exp(nll),
                cache_format_bytes_cl1024=plan["cache_bytes_per_token"]["total"] * 1024,
            )
        )
    _freeze_json(root / "offline_comparison.json", {"rows": results})
    with (root / "offline_comparison.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--archive", type=Path, default=Path("/mnt/d/ai-hub-models/binaries/turboquant")
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    freeze_plans(args.root)
    for model_id in MODELS:
        score_model(args.root, args.archive, model_id, args.device)
    summarize(args.root)


if __name__ == "__main__":
    main()
