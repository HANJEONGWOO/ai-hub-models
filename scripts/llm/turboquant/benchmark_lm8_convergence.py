# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Immutable three-way LM8 convergence experiment; one performance run/group."""

from __future__ import annotations

import argparse
import copy
import hashlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
from benchmark_asymmetric_once import ARTIFACTS, digest, execute, read, save_new
from benchmark_mixed_once import measure
from benchmark_nll_allocation import symlink
from transformers import AutoTokenizer
from verify_kv_boundary import parse_dlcinfo

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.lloyd_max_k8 import (
    codebook,
    metadata,
)
from qai_hub_models.models.templates.llm.turboquant.reference import load_codebook

GROUPS = {
    "lm8_100": "mixed_k6_v2_uniform",
    "lm8_converged": "mixed_k6_v2_uniform_lm8_converged",
    "structured": "mixed_k6_v2_uniform_structured_k8",
}
PREVIOUS = ARTIFACTS / "structured_k8_decoder_20261013"
LM_SOURCE = ARTIFACTS / "hardware_aligned_mixed_kv_20261010/mixed_k6_v2_uniform"
STRUCTURED_SOURCE = PREVIOUS / "structured_arithmetic"
NLL_PREVIOUS = ARTIFACTS / "k_rotation_nll_selection_20261009"


def canonical(title: str) -> str:
    return " ".join(title.strip().strip("=").strip().lower().split())


def heldout(
    root: Path, assets: dict, *, selection_salt: str = "LM8-heldout-20261020:"
) -> dict:
    """Freeze four new documents by content hash order, never by model outcome."""
    inventory = read(NLL_PREVIOUS / "prior_data_inventory.json")
    excluded = set(inventory["canonical_titles"])
    prior_files = {Path(p) for p in inventory["token_files_sha256"]}
    manifests = sorted(
        set(ARTIFACTS.glob("*/data_manifest.json"))
        | set(ARTIFACTS.glob("*/*/data_manifest.json"))
    )

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("title", "canonical_title") and isinstance(item, str):
                    excluded.add(canonical(item))
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for path in manifests:
        if root not in path.parents:
            visit(read(path))
    for path in ARTIFACTS.glob("*/assets*/*.bin"):
        if path.stat().st_size <= 65536 and "rope" not in path.name:
            prior_files.add(path)
    old_spans = set()
    old_hashes = {}
    for path in sorted(prior_files):
        if not path.is_file():
            continue
        ids = np.fromfile(path, dtype="<i4")
        old_hashes[str(path)] = digest(path)
        old_spans.update(ids[i : i + 16].tobytes() for i in range(len(ids) - 15))
    source = Path(
        read(NLL_PREVIOUS / "data_manifest.json")["datasets"]["heldout"]["path"]
    )
    rows = pd.read_parquet(source)["text"].tolist()
    documents = []
    current = []
    title = ""
    for row in rows:
        if re.fullmatch(r"\s*= [^=]+ =\s*", row):
            if current:
                documents.append((title, "".join(current)))
            title, current = row.strip(), [row]
        elif current:
            current.append(row)
    if current:
        documents.append((title, "".join(current)))
    documents.sort(
        key=lambda item: hashlib.sha256((selection_salt + item[1]).encode()).hexdigest()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        assets["checkpoint_dir"], local_files_only=True
    )
    selected = []
    for title, document in documents:
        if canonical(title) in excluded:
            continue
        tokens = np.array(
            tokenizer.encode(document, add_special_tokens=False), dtype="<i4"
        )
        if len(tokens) < 1024:
            continue
        # Reject any prior 16-token overlap anywhere in the source document,
        # not merely at the selected window. New evaluations never enter this set.
        if any(
            tokens[i : i + 16].tobytes() in old_spans for i in range(len(tokens) - 15)
        ):
            continue
        offset = min(256, len(tokens) - 1024)
        window = tokens[offset : offset + 1024]
        name = f"heldout_{len(selected)}.bin"
        with (root / "assets" / name).open("xb") as stream:
            stream.write(window.tobytes())
        assets["sha256"][name] = digest(root / "assets" / name)
        selected.append(
            {
                "title": title,
                "canonical_title": canonical(title),
                "offset": offset,
                "file": name,
                "tokens": 1024,
                "scored_tokens": 1023,
                "document_sha256": hashlib.sha256(document.encode()).hexdigest(),
                "tokens_sha256": assets["sha256"][name],
            }
        )
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError(
            "Insufficient unused heldout documents; do not silently change data protocol"
        )
    return {
        "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
        "split": "test",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "parquet": str(source),
        "parquet_sha256": digest(source),
        "selection": f"content-hash ascending salted {selection_salt.rstrip(':')}; offset min(256,length-1024); four eligible docs",
        "heldout": selected,
        "legacy": {k: assets["sha256"][k] for k in assets["wikitext_windows"]},
        "excluded_titles": sorted(excluded),
        "prior_token_files_sha256": old_hashes,
        "prior_manifest_sha256": {
            str(p): digest(p) for p in manifests if root not in p.parents
        },
        "overlap_16grams": 0,
        "limits": "Unused relative to recorded prior experiment documents/tokens, not a claim of exclusion from original pretrained model training.",
    }


def prepare(root: Path, data_manifest: Path | None = None) -> None:
    for sub in ("assets", "reports", "logs", "builds"):
        (root / sub).mkdir(parents=True, exist_ok=False)
    optimized = read(root / "codebook/optimization.json")
    assert optimized["centroid_f64_sha256"] == metadata()["codebook_sha256"]
    assets_dir = ARTIFACTS / "device_assets_cl1024"
    assets = copy.deepcopy(read(assets_dir / "assets.json"))
    for name, sha in assets["sha256"].items():
        assert digest(assets_dir / name) == sha
        symlink(assets_dir / name, root / "assets" / name)
    if data_manifest is None:
        data = heldout(root, assets)
    else:
        # Reproduction must reuse the frozen windows, not reselect after this
        # experiment's heldout documents have entered the prior-data inventory.
        data = read(data_manifest)
        if (
            len(data.get("heldout", [])) != 4
            or len({d["file"] for d in data["heldout"]}) != 4
        ):
            raise ValueError("Reproduction requires four distinct heldout windows")
        if data["legacy"] != {
            name: assets["sha256"][name] for name in assets["wikitext_windows"]
        }:
            raise ValueError("Reproduction legacy dataset differs")
        for document in data["heldout"]:
            source = data_manifest.parent / "assets" / document["file"]
            if (
                digest(source) != document["tokens_sha256"]
                or source.stat().st_size != 4096
            ):
                raise ValueError("Frozen heldout token file changed")
            symlink(source, root / "assets" / document["file"])
            assets["sha256"][document["file"]] = document["tokens_sha256"]
    save_new(root / "data_manifest.json", data)
    save_new(root / "assets/assets.json", assets)
    native = read(PREVIOUS / "native/manifest.json")
    frozen = [
        root / "codebook/optimization.json",
        root / "data_manifest.json",
        PREVIOUS / "native/manifest.json",
    ]
    frozen += [Path(v["path"]) for v in native["libraries"].values()]
    for source in (LM_SOURCE, STRUCTURED_SOURCE):
        frozen += [source / "convert_report.json", *source.glob("part*_of_4.bin")]
        frozen += (
            list(source.glob("*.onnx"))
            + list(source.glob("*.encodings"))
            + list(source.glob("*.dlc"))
        )
    runner = ARTIFACTS / "structured_k8_nll_allocation_20261013/runner/qnn-llm-runner"
    frozen += [runner] + [root / "assets" / name for name in assets["sha256"]]
    profiles = {
        g: {
            "profile": p,
            "config": get_profile(p).to_dict(),
            "config_hash": get_profile(p).config_hash(),
        }
        for g, p in GROUPS.items()
    }
    for group, source in (("lm8_100", LM_SOURCE), ("structured", STRUCTURED_SOURCE)):
        target = root / group
        target.mkdir()
        previous = read(source / "convert_report.json")
        assert previous["config_hash"] == profiles[group]["config_hash"]
        for path in source.iterdir():
            if not path.is_file() or path.name == "convert_report.json":
                continue
            if (
                (
                    group == "structured"
                    and re.fullmatch(r"part\d+_of_\d+\.bin", path.name)
                )
                or path.suffix in (".onnx", ".dlc", ".encodings", ".data")
                or path.name.endswith((".kv_edits.json", ".dlcinfo.txt"))
            ):
                symlink(path, target / path.name)
        previous = copy.deepcopy(previous)
        if group == "lm8_100":
            previous["native_decoder"] = native
        previous["convergence_experiment_source"] = str(source)
        save_new(target / "convert_report.json", previous)
    save_new(
        root / "protocol.json",
        {
            "branch": "exp/lloyd-max-k8-convergence",
            "base_branch": "exp/structured-k8-nll-allocation",
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "model": "Qwen3-1.7B W4A16",
            "context_length": 1024,
            "groups": list(GROUPS),
            "profiles": profiles,
            "assets": str(root / "assets"),
            "assets_manifest": assets,
            "runner": str(runner),
            "split": str(ARTIFACTS / "qwen3_1_7b_w4a16_split"),
            "native": str(PREVIOUS / "native"),
            "frozen_files": {str(p): digest(p) for p in frozen},
            "performance": {
                "sessions_per_group": 1,
                "prompt": 897,
                "generated": 128,
                "profiled": False,
            },
            "quality": {
                "legacy_windows": 4,
                "new_heldout_windows": 4,
                "scored_targets_per_window": 1023,
                "selection": "Codebook selected only by Gaussian MSE; no real-data fitting or PPL feedback",
            },
            "fixed": "Odd14 K8/even14 K4, V2 all, Dense K42/V542, current+past compressed, norm correction FP16 scales, tile256, weights/calibration; 30277632 KV bytes",
            "attribution": "LM100 to converged: constants only. Structured vs LM changes BOTH quantizer and decoder; isolate codec probes and prior Structured same-codebook LUT/arithmetic evidence, never attribute full-model delta to one factor.",
            "limits": "Single performance observation, no significance/dispersion claim. Encoder historical numerical failures remain separately reported, not waived.",
        },
    )
    print("HELDOUT", [d["title"] for d in data["heldout"]], flush=True)


def validate(root: Path) -> dict:
    protocol = read(root / "protocol.json")
    for name, sha in protocol["frozen_files"].items():
        if digest(Path(name)) != sha:
            raise ValueError("Frozen input changed: " + name)
    for group, profile in GROUPS.items():
        assert (
            get_profile(profile).config_hash()
            == protocol["profiles"][group]["config_hash"]
        )
    return protocol


def freeze_sources(root: Path, evaluation_order: list[str] | None = None) -> None:
    """Pin the actual exported model weights/grids, not only deployment contexts."""
    split = ARTIFACTS / "qwen3_1_7b_w4a16_split/split_manifest.json"
    files = [split]
    for entry in read(split)["parts"].values():
        directory = Path(entry["bundle_dir"])
        for suffix in ("*.onnx", "*.data", "*.encodings"):
            files.extend(directory.glob(suffix))
    save_new(
        root / "source_identity.json",
        {
            "files_sha256": {str(p): digest(p) for p in files},
            "purpose": "Unchanged exported W4A16 weights/activation grids; no training or recalibration",
            "actual_evaluation_order": evaluation_order
            or ["structured", "lm8_100", "lm8_converged"],
            "order_reason": "Predeclared repeated order; see protocol"
            if evaluation_order
            else "Reuse Structured while new LM contexts build; single run per group, no counterbalancing claim",
        },
    )
    import onnxruntime
    import scipy
    import torch

    checkpoint = Path(read(root / "assets/assets.json")["checkpoint_dir"])
    save_new(
        root / "environment.json",
        {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "tokenizer_sha256": {
                str(p): digest(p)
                for p in (
                    checkpoint / "tokenizer.json",
                    checkpoint / "tokenizer_config.json",
                )
            },
            "note": "Model weights/calibration in source_identity.json; runtime Native libraries and assets in protocol.json. No random fitting samples.",
        },
    )


def build(root: Path, protocol: dict) -> None:
    def job(number: int) -> None:
        group = "lm8_100" if number == 0 else "lm8_converged"
        target = root / group if number == 0 else root / "builds" / f"part{number}"
        options = [
            "--split-dir",
            protocol["split"],
            "--out",
            str(target),
            "--profile",
            GROUPS[group],
            "--context-length",
            "1024",
            "--sequence-lengths",
            "1",
            "128",
            "--native-decoder-package",
            protocol["native"],
        ]
        options += ["--context-only"] if number == 0 else ["--parts", str(number)]
        execute("convert_parts.py", options, root / "logs" / f"build_{number}.log")

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(job, (2, 3, 4, 0)))
    assemble(root)


def assemble(root: Path) -> None:
    target = root / "lm8_converged"
    target.mkdir(exist_ok=False)
    combined = copy.deepcopy(read(root / "builds/part2/convert_report.json"))
    combined["parts"] = {}
    provenance = {}
    for number in (1, 2, 3, 4):
        source = root / "lm8_100" if number == 1 else root / "builds" / f"part{number}"
        report = read(source / "convert_report.json")
        name = f"part{number}_of_4"
        entry = report["parts"][name]
        assert (source / f"{name}.bin").is_file() and "context_s" in entry
        if number == 1:
            assert all(
                not g.get("surgery", {}).get("codec_io")
                for g in entry["graphs"].values()
            )
        else:
            assert report["config_hash"] == combined["config_hash"]
        combined["parts"][name] = entry
        symlink(source / f"{name}.bin", target / f"{name}.bin")
        for graph in entry["graphs"]:
            for path in source.glob(graph + ".*"):
                if path.is_file():
                    symlink(path, target / path.name)
        for path in source.glob("*.data"):
            symlink(path, target / path.name)
        provenance[name] = {
            "source": str(source),
            "binary_sha256": digest(source / f"{name}.bin"),
            "report_sha256": digest(source / "convert_report.json"),
        }
    combined["convergence_parts"] = provenance
    save_new(target / "convert_report.json", combined)


def audit(root: Path) -> None:
    rows = {}
    for path in sorted((root / "lm8_converged").glob("*.kv_edits.json")):
        name = path.name.removesuffix(".kv_edits.json")
        graphs = [
            onnx.load(root / g / (name + ".onnx"), load_external_data=False).graph
            for g in ("lm8_100", "lm8_converged")
        ]
        normalized = []
        for graph in graphs:
            items = []
            for node in graph.node:
                n = copy.deepcopy(node)
                for i, value in enumerate(n.input):
                    if value == "tq_native_converged_k8_centroids_fp16":
                        n.input[i] = "tq_native_b8_centroids_fp16"
                items.append(n.SerializeToString())
            normalized.append(items)
        constants = [{t.name: t for t in graph.initializer} for graph in graphs]
        shared = constants[0].keys() & constants[1].keys()
        changed = [
            k
            for k in shared
            if constants[0][k].SerializeToString()
            != constants[1][k].SerializeToString()
        ]
        lut = onnx.numpy_helper.to_array(
            constants[1]["tq_native_converged_k8_centroids_fp16"]
        )
        checks = {
            "nodes_identical_except_lut_name": normalized[0] == normalized[1],
            "only_k8_constants_changed": bool(changed)
            and all(
                k.startswith(("tq_scalar_b8_", "tq_centroid_b8_")) for k in changed
            ),
            "io_identical": all(
                [v.SerializeToString() for v in getattr(graphs[0], f)]
                == [v.SerializeToString() for v in getattr(graphs[1], f)]
                for f in ("input", "output")
            ),
            "lut_fp16_exact": np.array_equal(
                lut.ravel(), codebook().astype(np.float16)
            ),
            "legacy_lut_fp16_exact": np.array_equal(
                onnx.numpy_helper.to_array(
                    constants[0]["tq_native_b8_centroids_fp16"]
                ).ravel(),
                load_codebook(8, 128).astype(np.float16),
            ),
            "only_lut_constant_renamed": constants[0].keys() - constants[1].keys()
            == {"tq_native_b8_centroids_fp16"}
            and constants[1].keys() - constants[0].keys()
            == {"tq_native_converged_k8_centroids_fp16"},
            "all_255_boundaries_exact": all(
                float(
                    onnx.numpy_helper.to_array(
                        constants[1][f"tq_scalar_b8_d128_threshold_{i}"]
                    )
                )
                == float(value)
                for i, value in enumerate(
                    ((codebook()[:-1] + codebook()[1:]) / 2).astype(np.float32)
                )
            ),
            "encodings_identical": digest(root / "lm8_100" / (name + ".encodings"))
            == digest(root / "lm8_converged" / (name + ".encodings")),
        }
        compiled = [
            parse_dlcinfo(root / group / (name + ".dlcinfo.txt"))
            for group in ("lm8_100", "lm8_converged")
        ]
        signatures = []
        for info in compiled:
            signature = {}
            for op in info.ops:
                value = asdict(op)
                value.pop("ident")
                if op.op_type == "RmsNorm" and re.fullmatch(
                    r"rms_norm_node___\d+(?:__\d+)*", op.name
                ):
                    value["name"] = "SDK_RMS_AUTONAME"
                for tensor in value["inputs"]:
                    tensor["name"] = tensor["name"].replace(
                        "tq_native_converged_k8_centroids_fp16",
                        "tq_native_b8_centroids_fp16",
                    )
                # Static numerical parameters change with the intended K8
                # constants; connectivity, shape and dtype still must agree.
                if re.match(r"tq_key_\d+_enc_(scalar_|scale_)", op.name):
                    value.pop("params")
                signature[tuple(t.name for t in op.outputs)] = value
            signatures.append(signature)
        checks["compiled_op_count_identical"] = len(compiled[0].ops) == len(
            compiled[1].ops
        )
        checks["compiled_io_identical"] = compiled[0].io_tables == compiled[1].io_tables
        checks["compiled_matmul_contract_identical"] = {
            k: v
            for k, v in signatures[0].items()
            if v["op_type"] in ("MatMul", "FullyConnected")
        } == {
            k: v
            for k, v in signatures[1].items()
            if v["op_type"] in ("MatMul", "FullyConnected")
        }
        differing = [
            list(k)
            for k in signatures[0].keys() | signatures[1].keys()
            if signatures[0].get(k) != signatures[1].get(k)
        ]
        checks["compiled_contracts_identical_except_codebook_constants"] = not differing
        rows[name] = {
            "checks": checks,
            "changed_constants": sorted(changed),
            "compiled_signature_differences": differing,
        }
    passed = len(rows) == 6 and all(
        all(row["checks"].values()) for row in rows.values()
    )
    save_new(
        root / "reports/codebook_only_audit.json", {"graphs": rows, "passed": passed}
    )
    if not passed:
        raise ValueError("Codebook-only source audit failed")


def evaluate(root: Path, group: str, protocol: dict) -> None:
    if (root / group / "runtime_manifest.json").is_symlink():
        raise ValueError(
            "Runtime manifests must be local experiment files, never inherited symlinks"
        )
    execute(
        "verify_rotated_attention.py",
        [
            "--bundle",
            str(root / group),
            "--report",
            str(root / "reports" / f"audit_{group}.json"),
        ],
        root / "logs" / f"audit_{group}.log",
    )
    execute(
        "run_device_llm.py",
        [
            "push",
            "--bundle-dir",
            str(root / group),
            "--runner",
            protocol["runner"],
            "--name",
            root.name + "_" + group,
        ],
        root / "logs" / f"push_{group}.log",
    )
    reset = measure(
        root,
        group,
        "reset",
        ["--mode", "generate", "--n-gen", "8", "--sessions", "2"],
        protocol,
    )
    assert reset["sessions"][0]["generated"] == reset["sessions"][1]["generated"]
    measure(
        root,
        group,
        "perf_long",
        [
            "--mode",
            "generate",
            "--n-gen",
            "128",
            "--sessions",
            "1",
            "--tokens",
            "boundary_prompt_cl1024.bin",
        ],
        protocol,
    )
    for split, prefix in (("legacy", "wikitext_w"), ("heldout", "heldout_")):
        for i in range(4):
            report = measure(
                root,
                group,
                f"score_{split}_{i}",
                ["--mode", "score", "--tokens", f"{prefix}{i}.bin"],
                protocol,
            )
            assert report["scored_tokens"] == 1023


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=("prepare", "sources", "build", "assemble", "audit", "evaluate"),
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--data-manifest",
        type=Path,
        help="Reuse a frozen experiment's heldout windows for exact reproduction",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        choices=list(GROUPS),
        default=["structured", "lm8_100", "lm8_converged"],
    )
    args = p.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        prepare(root, args.data_manifest)
        freeze_sources(root)
        return
    if args.stage == "sources":
        freeze_sources(root)
        return
    protocol = validate(root)
    if args.stage == "build":
        build(root, protocol)
    elif args.stage == "assemble":
        assemble(root)
    elif args.stage == "audit":
        audit(root)
    else:
        for group in args.groups:
            evaluate(root, group, protocol)


if __name__ == "__main__":
    main()
