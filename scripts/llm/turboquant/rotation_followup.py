# SPDX-License-Identifier: BSD-3-Clause
"""Frozen layer-selection and independent shared-rotation update diagnostics.

No device model is rebuilt by these offline stages. Existing artifacts are read
only; new outputs use exclusive creation. Hard forwards and the STE are imported
unchanged from the original experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from learn_key_rotation import Samples, attention, codec, rounded
from rotation_data import DATA_REVISION, digest, write_json
from transformers import AutoTokenizer

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.packing import (
    pack_indices,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.reference import DenseQRRotation
from qai_hub_models.models.templates.llm.turboquant.rotation_artifact import (
    load_rotation,
    matrix_digest,
    save_layer_rotations,
    save_rotation,
    with_key_rotation,
)

LAYERS = [0, 4, 8, 12, 16, 20, 24, 27]
POSITIONS = [31, 159, 287, 415, 543, 671, 799, 927]
PROTOCOL = {
    "version": 1,
    "seed": 20261009,
    "model": "Qwen3-1.7B W4A16",
    "context": 1024,
    "A_seed": 42,
    "B_seed": 48,
    "selection_seeds": list(range(42, 50)),
    "P_selection": "argmin original validation absolute layer_mse, ties lowest seed; no reselection",
    "diagnostic_layers": LAYERS,
    "diagnostic_windows": [0, 1],
    "diagnostic_positions": POSITIONS,
    "diagnostic_train": "original train windows 0,1; fixed layers/queries/all heads",
    "diagnostic_validation": "original validation windows 0,1; same fixed layers/queries/all heads",
    "learning_rates": [1e-3, 1e-4, 1e-5],
    "gradient_optimizer_projection": "unchanged identity STE + Adam defaults + FP64 SVD/polar after each update",
    "one_step": "independent optimizer and B matrix for each learning rate",
    "short_extension": "Only if at least one first step reduces fixed train hard MSE. Among those choose lowest validation MSE, tie smaller lr; continue that same optimizer to total 20 steps, evaluate every step. No other tuning.",
    "D_selection": "minimum fixed validation hard MSE, B step0 eligible; promote only if fixed train AND validation, full original validation AND extra validation each improve over B",
    "extra_validation": "4 unused title-disjoint official validation documents; all 28 layers, positions31,63,...1023",
    "heldout": "4 unused title-disjoint official test documents; PPL only after candidates frozen; never selection",
    "CPU_reference_gate": "relative L2 < 0.02 on extra validation",
    "P_gate": "extra-validation aggregate absolute Attention MSE strictly below B",
    "HTP_gate": "isolated FP16 Attention relative L2 <0.03; original encoder/unconditioned errors separately retained",
    "HTP_probe": "real extra-validation doc0, every distinct accepted K matrix at its first layer; AR1/128, current/past, 2 KV heads x GQA2",
    "performance_order_no_D": ["ABP", "BPA", "PAB"],
    "performance_order_with_D": ["ABPD", "BPDA", "PDAB"],
    "performance_conditions": ["35+128", "897+128"],
    "performance_repeats": 3,
    "quality": "one score run per new heldout document per accepted group; aggregate total NLL / total tokens",
    "invariants": "LM tree/codebook, K4/V4, QJL off, compressed current KV, norm correction/FP16 scale, V seed542, W4A16 weights/calibration unchanged",
    "no_combination": "P and shared learned D never combined",
    "stop": "No expanded data/search/training after failed gate; preserve every attempt",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def source(root: Path) -> Path:
    record = read(root / "source_identity.json")
    old = Path(record["root"])
    for name, checksum in record["sha256"].items():
        if digest(old / name) != checksum:
            raise ValueError(f"Original experiment changed: {name}")
    if read(root / "protocol.json") != PROTOCOL:
        raise ValueError("Frozen followup protocol changed")
    return old


def freeze(root: Path, old: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    (root / "reports").mkdir()
    (root / "rotations").mkdir()
    names = [
        "selection.json",
        "training.json",
        "protocol.json",
        "data_manifest.json",
        "capture_identity.json",
        "samples/manifest.json",
        "samples/encodings.json",
        "rotations/A.json",
        "rotations/B.json",
        "rotations/C.json",
        "reports/experiment.json",
    ]
    old_identity = read(old / "reports/experiment.json")
    for group in "AB":
        for name, checksum in old_identity["groups"][group]["bins_sha256"].items():
            if digest(old / group / name) != checksum:
                raise ValueError("Original binary changed")
            names.append(f"{group}/{name}")
        names.append(f"{group}/convert_report.json")
    write_json(
        root / "source_identity.json",
        {
            "root": str(old.resolve()),
            "sha256": {name: digest(old / name) for name in names},
        },
    )
    write_json(root / "protocol.json", PROTOCOL)
    for g in "AB":
        shutil.copy2(old / f"rotations/{g}.json", root / f"rotations/{g}.json")
    # New writable directory, immutable shared payloads. push writes runtime_manifest.
    for name in ("capture_bundle", "runner", "A", "B"):
        dest = root / name
        dest.mkdir()
        for item in (old / name).iterdir():
            if item.name not in ("runtime_manifest.json", "push_manifest.json"):
                (dest / item.name).symlink_to(item.resolve())
    prepare_data(root, old)
    print("FROZEN", root, digest(root / "protocol.json"), flush=True)


def prepare_data(root: Path, old: Path) -> None:
    previous = read(old / "data_manifest.json")
    used = {w["document_id"] for rows in previous["windows"].values() for w in rows}
    base = read(old / "assets/assets.json")
    tokenizer = AutoTokenizer.from_pretrained(
        base["checkpoint_dir"], local_files_only=True
    )
    rng = np.random.default_rng(PROTOCOL["seed"])
    out = root / "assets"
    out.mkdir()
    windows, datasets = {}, {}
    for split in ("validation", "test"):
        path = Path(previous["datasets"][split]["path"])
        if digest(path) != previous["datasets"][split]["sha256"]:
            raise ValueError("Pinned source dataset changed")
        datasets[split] = {"path": str(path), "sha256": digest(path)}
        documents, title, lines = [], None, []
        for line in pd.read_parquet(path)["text"]:
            if re.fullmatch(r"\s*= [^=]+ =\s*", line):
                if title is not None:
                    documents.append((title, "\n".join(lines)))
                title, lines = line.strip(), [line]
            elif title is not None:
                lines.append(line)
        if title is not None:
            documents.append((title, "\n".join(lines)))
        chosen = []
        for index in rng.permutation(len(documents)):
            title, text = documents[index]
            title_hash = hashlib.sha256(title.encode()).hexdigest()
            if title_hash in used:
                continue
            tokens = tokenizer(text, add_special_tokens=False).input_ids
            if len(tokens) < 1024:
                continue
            offset = int(rng.integers(len(tokens) - 1024 + 1))
            name = f"{split}_{len(chosen)}.bin"
            np.asarray(tokens[offset : offset + 1024], dtype="<i4").tofile(out / name)
            used.add(title_hash)
            chosen.append(
                {
                    "file": name,
                    "title": title,
                    "document_id": title_hash,
                    "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "offset": offset,
                    "tokens_sha256": digest(out / name),
                }
            )
            if len(chosen) == 4:
                break
        if len(chosen) != 4:
            raise ValueError("Insufficient unused documents")
        windows[split] = chosen
    for name in (base["rope"], base["prompt_ids"], base["boundary_prompt"]):
        shutil.copy2(old / "assets" / name, out / name)
    base["wikitext_windows"] = [w["file"] for w in windows["test"]]
    base["sha256"] = {p.name: digest(p) for p in sorted(out.glob("*.bin"))}
    write_json(out / "assets.json", base)
    write_json(
        root / "data_manifest.json",
        {
            "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
            "revision": DATA_REVISION,
            "seed": PROTOCOL["seed"],
            "context": 1024,
            "chunk": 128,
            "datasets": datasets,
            "windows": windows,
            "excluded_original_document_ids": sorted(
                {
                    w["document_id"]
                    for rows in previous["windows"].values()
                    for w in rows
                }
            ),
            "selection": "4 new validation, 4 heldout test; no test capture or loss during selection",
        },
    )


def matrix(seed: int) -> torch.Tensor:
    return torch.tensor(DenseQRRotation(seed, 128).matrix().astype(np.float32))


@torch.no_grad()
def evaluate_policy(
    samples: Samples,
    matrices: list[torch.Tensor],
    split: str,
    windows: int,
    reference: bool = False,
) -> dict:
    positions = torch.arange(31, 1024, 32)
    entries, squared, refsq, count = {}, 0.0, 0.0, 0
    for layer in range(28):
        sq, target_sq, n = 0.0, 0.0, 0
        for window in range(windows):
            sample = samples.get(split, window, layer)
            output = attention(sample, matrices[layer], positions, not reference)
            target = sample["o"][:, positions]
            sq += float((output - target).double().square().sum())
            target_sq += float(target.double().square().sum())
            n += target.numel()
        entries[str(layer)] = {
            "mse": sq / n,
            "reference_mean_square": target_sq / n,
            "relative_l2": (sq / target_sq) ** 0.5,
            "elements": n,
        }
        squared, refsq, count = squared + sq, refsq + target_sq, count + n
    return {
        "mse": squared / count,
        "relative_l2": (squared / refsq) ** 0.5,
        "elements": count,
        "layers": entries,
    }


def analyze(root: Path) -> None:
    old = source(root)
    candidates = read(old / "selection.json")["candidates"]
    if [c["seed"] for c in candidates] != PROTOCOL["selection_seeds"]:
        raise ValueError("Unexpected original candidate set")
    values = np.asarray(
        [[c["layer_mse"][str(l)] for c in candidates] for l in range(28)]
    )
    selected = values.argmin(1)
    seeds = [candidates[i]["seed"] for i in selected]
    for group, seed in (("A", 42), ("B", 48)):
        if (
            matrix_digest(matrix(seed).numpy())
            != read(root / f"rotations/{group}.json")["matrix_f32_sha256"]
        ):
            raise ValueError(
                "Current QR implementation no longer reproduces original A/B matrices"
            )
    save_layer_rotations(
        root / "rotations/P.json",
        seeds,
        {s: matrix(s).numpy() for s in set(seeds)},
        {
            "selection_sha256": digest(old / "selection.json"),
            "protocol_sha256": digest(root / "protocol.json"),
        },
    )
    samples = Samples(old)
    refs = []
    for layer in range(28):
        targets = [
            samples.get("validation", w, layer)["o"][:, 31::32].double()
            for w in range(2)
        ]
        refs.append(
            sum(float(x.square().sum()) for x in targets)
            / sum(x.numel() for x in targets)
        )
    p = values[np.arange(28), selected]
    records = []
    for l in range(28):
        row = {"layer": l, "selected_seed": seeds[l], "reference_mean_square": refs[l]}
        for i, candidate in enumerate(candidates):
            row[f"seed{candidate['seed']}_mse"] = values[l, i]
            row[f"seed{candidate['seed']}_relative_l2"] = (
                values[l, i] / refs[l]
            ) ** 0.5
        for label, idx in (("A", 0), ("B", 6)):
            delta = values[l, idx] - p[l]
            row[f"P_vs_{label}_relative_mse_change"] = p[l] / values[l, idx] - 1
            row[f"P_vs_{label}_aggregate_mse_reduction_contribution"] = delta / 28
            row[f"P_vs_{label}_reduction_share"] = delta / (values[:, idx] - p).sum()
        records.append(row)
    with (root / "reports/layer_seed_table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(
        root / "layer_selection.json",
        {
            "source_selection_sha256": digest(old / "selection.json"),
            "policy": PROTOCOL["P_selection"],
            "layer_seeds": seeds,
            "layers": records,
            "original_validation": {
                "A_mse": float(values[:, 0].mean()),
                "B_mse": float(values[:, 6].mean()),
                "P_mse": float(p.mean()),
                "P_relative_l2": float((p.sum() / sum(refs)) ** 0.5),
            },
            "not_a_full_model_PPL_result": True,
        },
    )
    heatmap(root)
    print("SELECTED P", seeds, "MSE", float(p.mean()), flush=True)


def heatmap(root: Path) -> None:
    """Render frozen selection results without reselecting or overwriting them."""
    selection = read(root / "layer_selection.json")
    values = np.asarray(
        [[row[f"seed{s}_mse"] for s in range(42, 50)] for row in selection["layers"]]
    )
    selected = np.asarray(selection["layer_seeds"]) - 42
    for suffix in ("png", "svg"):
        if (root / f"reports/layer_seed_heatmap.{suffix}").exists():
            raise FileExistsError("Heatmap already exists")
    # Data-driven scientific figure, not an AI-generated illustration.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 12), constrained_layout=True)
    for ax, data, title in zip(
        axes,
        (np.log10(values), values / values[:, 6:7] - 1),
        ("log10 absolute Attention MSE", "Relative MSE change vs shared seed48"),
        strict=True,
    ):
        im = ax.imshow(
            data, aspect="auto", cmap="viridis" if ax is axes[0] else "coolwarm"
        )
        ax.set_xticks(range(8), range(42, 50))
        ax.set_yticks(range(28))
        ax.set_xlabel("K rotation seed")
        ax.set_ylabel("Layer (0-based)")
        ax.set_title(title)
        ax.scatter(
            selected, range(28), marker="s", facecolors="none", edgecolors="black"
        )
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(root / "reports/layer_seed_heatmap.png", dpi=170)
    fig.savefig(root / "reports/layer_seed_heatmap.svg")
    plt.close(fig)


def fixed_items(samples: Samples, split: str) -> list[tuple[dict, torch.Tensor, str]]:
    positions = torch.tensor(POSITIONS)
    return [
        (samples.get(split, w, l), positions, f"{split}_{w}_layer{l}")
        for w in (0, 1)
        for l in LAYERS
    ]


def fixed_loss(items: list, rotation: torch.Tensor, backward: bool = False) -> dict:
    loss_sum, by_sample = 0.0, {}
    for sample, positions, name in items:
        output = attention(sample, rotation, positions)
        loss = (output - sample["o"][:, positions]).square().mean()
        if backward:
            (loss / len(items)).backward()
        by_sample[name] = float(loss.detach())
        loss_sum += float(loss.detach())
    return {"mse": loss_sum / len(items), "sample_mse": by_sample}


@torch.no_grad()
def index_change(items: list, before: torch.Tensor, after: torch.Tensor) -> dict:
    changed, total, by_sample = 0, 0, {}
    for sample, _, name in items:
        old, new = codec(sample["k"], before)[1], codec(sample["k"], after)[1]
        n = int((old != new).sum())
        changed += n
        total += old.numel()
        by_sample[name] = n / old.numel()
    return {
        "fraction": changed / total,
        "changed": changed,
        "total": total,
        "sample_fraction": by_sample,
    }


def orthogonal_error(rotation: torch.Tensor) -> float:
    return float(
        torch.linalg.matrix_norm(
            rotation.double().T @ rotation.double() - torch.eye(128), ord=2
        )
    )


def update(
    parameter: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    train: list,
    validation: list,
    initial: torch.Tensor,
    step: int,
) -> dict:
    before = parameter.detach().clone()
    optimizer.zero_grad()
    train_before = fixed_loss(train, parameter, backward=True)
    gradient = parameter.grad.detach().clone()
    if not torch.isfinite(gradient).all():
        raise ValueError("Nonfinite STE gradient")
    optimizer.step()
    pre_projection = parameter.detach().clone()
    with torch.no_grad():
        u, _, vh = torch.linalg.svd(parameter.double(), full_matrices=False)
        parameter.copy_((u @ vh).float())
        after = parameter.detach().clone()
        hbefore, hafter = rounded(before), rounded(after)
        result = {
            "step": step,
            "gradient_l2": float(gradient.norm()),
            "gradient_max_abs": float(gradient.abs().max()),
            "update_pre_projection_fro": float((pre_projection - before).norm()),
            "update_post_projection_fro": float((after - before).norm()),
            "projection_correction_fro": float((after - pre_projection).norm()),
            "update_from_B_fro": float((after - initial).norm()),
            "fp16_update_fro": float((hafter - hbefore).norm()),
            "fp16_changed_fraction": float((hafter != hbefore).float().mean()),
            "orthogonality_before": orthogonal_error(before),
            "orthogonality_pre_projection": orthogonal_error(pre_projection),
            "orthogonality_after": orthogonal_error(after),
            "orthogonality_fp16": orthogonal_error(hafter),
            "train_before": train_before,
            "train_after": fixed_loss(train, after),
            "validation_after": fixed_loss(validation, after),
            "index_change_train": index_change(train, before, after),
            "index_change_validation": index_change(validation, before, after),
            "matrix_f32_sha256": matrix_digest(after.numpy()),
        }
    print(
        "UPDATE",
        step,
        optimizer.param_groups[0]["lr"],
        "train",
        result["train_after"]["mse"],
        "val",
        result["validation_after"]["mse"],
        "grad",
        result["gradient_l2"],
        flush=True,
    )
    return result


def diagnose(root: Path) -> None:
    old = source(root)
    work = root / "diagnostics"
    work.mkdir(exist_ok=False)
    torch.manual_seed(PROTOCOL["seed"])
    torch.set_num_threads(8)
    samples = Samples(old)
    train, validation = (
        fixed_items(samples, "train"),
        fixed_items(samples, "validation"),
    )
    initial = torch.tensor(load_rotation(root / "rotations/B.json")).reshape(128, 128)
    with torch.no_grad():
        baseline = {
            "step": 0,
            "train": fixed_loss(train, initial),
            "validation": fixed_loss(validation, initial),
        }
    write_json(work / "step0.json", baseline)
    trials = []
    for lr in PROTOCOL["learning_rates"]:
        parameter = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([parameter], lr=lr)
        record = update(parameter, optimizer, train, validation, initial, 1)
        write_json(work / f"lr{lr}_step01.json", record)
        save_rotation(
            work / f"lr{lr}_step01_matrix.json",
            parameter.detach().numpy(),
            {"lr": lr, "step": 1},
        )
        trials.append((lr, parameter, optimizer, record))
    eligible = [
        t for t in trials if t[3]["train_after"]["mse"] < baseline["train"]["mse"]
    ]
    best_matrix, best_step, best_val, best_train, selected_lr = (
        initial.clone(),
        0,
        baseline["validation"]["mse"],
        baseline["train"]["mse"],
        None,
    )
    history = []
    if eligible:
        lr, parameter, optimizer, first = min(
            eligible, key=lambda t: (t[3]["validation_after"]["mse"], t[0])
        )
        selected_lr = lr
        history.append(first)
        for step in range(1, 21):
            record = (
                first
                if step == 1
                else update(parameter, optimizer, train, validation, initial, step)
            )
            if step > 1:
                write_json(work / f"lr{lr}_step{step:02d}.json", record)
                save_rotation(
                    work / f"lr{lr}_step{step:02d}_matrix.json",
                    parameter.detach().numpy(),
                    {"lr": lr, "step": step},
                )
                history.append(record)
            if record["validation_after"]["mse"] < best_val:
                best_matrix, best_step = parameter.detach().clone(), step
                best_val, best_train = (
                    record["validation_after"]["mse"],
                    record["train_after"]["mse"],
                )
    save_rotation(
        root / "rotations/D.json",
        best_matrix.numpy(),
        {
            "method": "independent_shared_B_diagnostic",
            "selected_step": best_step,
            "learning_rate": selected_lr,
            "protocol_sha256": digest(root / "protocol.json"),
        },
    )
    write_json(
        root / "diagnostics.json",
        {
            "baseline": baseline,
            "first_steps": [{"lr": t[0], **t[3]} for t in trials],
            "extended_lr": selected_lr,
            "history": history,
            "selected_step": best_step,
            "selected_train_mse": best_train,
            "selected_validation_mse": best_val,
            "fixed_subset_pass": best_step > 0
            and best_train < baseline["train"]["mse"]
            and best_val < baseline["validation"]["mse"],
            "no_extended_search": True,
        },
    )


def validate(root: Path) -> None:
    old = source(root)
    torch.set_num_threads(8)
    samples = Samples(root)
    seeds = read(root / "layer_selection.json")["layer_seeds"]
    matrices = {
        "A": [matrix(42)] * 28,
        "B": [matrix(48)] * 28,
        "P": [matrix(s) for s in seeds],
    }
    report = {
        "groups": {},
        "cpu_reference": evaluate_policy(
            samples, matrices["A"], "validation", 4, reference=True
        ),
    }
    for group, values in matrices.items():
        report["groups"][group] = evaluate_policy(samples, values, "validation", 4)
        print("EXTRA VALIDATION", group, report["groups"][group]["mse"], flush=True)
    diagnostics = read(root / "diagnostics.json")
    report["D_fixed_subset_pass"] = diagnostics["fixed_subset_pass"]
    if diagnostics["fixed_subset_pass"]:
        d = torch.tensor(load_rotation(root / "rotations/D.json")).reshape(128, 128)
        original = Samples(old)
        report["D_original_validation"] = evaluate_policy(
            original, [d] * 28, "validation", 2
        )
        report["groups"]["D"] = evaluate_policy(samples, [d] * 28, "validation", 4)
    report["reference_pass"] = report["cpu_reference"]["relative_l2"] < 0.02
    b = report["groups"]["B"]["mse"]
    report["P_pass"] = report["reference_pass"] and report["groups"]["P"]["mse"] < b
    original_b = next(
        c["mse"] for c in read(old / "selection.json")["candidates"] if c["seed"] == 48
    )
    report["D_pass"] = (
        report["reference_pass"]
        and diagnostics["fixed_subset_pass"]
        and report["D_original_validation"]["mse"] < original_b
        and report["groups"]["D"]["mse"] < b
    )
    report["accepted_groups"] = (
        ["A", "B"]
        + (["P"] if report["P_pass"] else [])
        + (["D"] if report["D_pass"] else [])
    )
    report["not_full_model_PPL"] = True
    write_json(root / "cpu_quality.json", report)
    cpu_tables(root)
    print("CPU GATES", report["accepted_groups"], flush=True)


def cpu_tables(root: Path) -> None:
    """Derived tables only; no new sample, loss evaluation, or reselection."""
    quality = read(root / "cpu_quality.json")
    seeds = read(root / "layer_selection.json")["layer_seeds"]
    rows = []
    for layer in range(28):
        row = {"layer": layer, "P_seed": seeds[layer]}
        for group, result in quality["groups"].items():
            item = result["layers"][str(layer)]
            row[f"{group}_mse"] = item["mse"]
            row[f"{group}_relative_l2"] = item["relative_l2"]
        for baseline in "AB":
            delta = row[f"{baseline}_mse"] - row["P_mse"]
            total = quality["groups"][baseline]["mse"] - quality["groups"]["P"]["mse"]
            row[f"P_vs_{baseline}_relative_mse_change"] = (
                row["P_mse"] / row[f"{baseline}_mse"] - 1
            )
            row[f"P_vs_{baseline}_aggregate_mse_reduction_contribution"] = delta / 28
            row[f"P_vs_{baseline}_reduction_share"] = (
                delta / (28 * total) if total else None
            )
        rows.append(row)
    with (root / "reports/extra_validation_layers.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    diagnostics = read(root / "diagnostics.json")
    layers = {}
    for split in ("train", "validation"):
        values = diagnostics["baseline"][split]
        entries = []
        for layer in LAYERS:
            contribution = sum(
                v / 16
                for k, v in values["sample_mse"].items()
                if k.endswith(f"layer{layer}")
            )
            entries.append(
                {
                    "layer": layer,
                    "mse_contribution": contribution,
                    "loss_share": contribution / values["mse"],
                }
            )
        layers[split] = entries
    write_json(root / "reports/diagnostic_loss_shares.json", layers)


def invariants(root: Path) -> None:
    """Actual captured vectors: exact unquantized QK, FP16 drift and nibble ABI."""
    torch.set_num_threads(8)
    samples = Samples(root)
    config = with_key_rotation(get_profile("k4_v4_scaled"), root / "rotations/P.json")
    config.validate_for_model(28, 8, 128)
    result = {
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "layers": {},
    }
    for group, seed in (("A", 42), ("B", 48)):
        if (
            matrix_digest(matrix(seed).numpy())
            != read(root / f"rotations/{group}.json")["matrix_f32_sha256"]
        ):
            raise ValueError("Original A/B QR matrix changed")
    with torch.no_grad():
        for layer in range(28):
            rotation = torch.tensor(config.key_for_layer(layer).dense_matrix).reshape(
                128, 128
            )
            sample = samples.get("validation", 0, layer)
            q, k = sample["q"][0, :32], sample["k"][0, :128]
            direct = q.double() @ k.double().T
            rotated = (q.double() @ rotation.double().T) @ (
                k.double() @ rotation.double().T
            ).T
            half_direct = rounded(q @ k.T)
            half_rotated = rounded(
                rounded(q @ rounded(rotation).T) @ rounded(k @ rounded(rotation).T).T
            )
            _, indices, scale = codec(sample["k"], rotation)
            np.testing.assert_array_equal(
                unpack_indices(pack_indices(indices.numpy(), 4), 4, 128),
                indices.numpy(),
            )
            entry = {
                "seed": config.key_for_layer(layer).seed,
                "matrix_f32_sha256": matrix_digest(rotation.numpy()),
                "orthogonality_f32": orthogonal_error(rotation),
                "orthogonality_fp16": orthogonal_error(rounded(rotation)),
                "unquantized_qk_f64_max_abs": float((rotated - direct).abs().max()),
                "unquantized_qk_fp16_relative_l2": float(
                    (half_rotated - half_direct).double().norm()
                    / half_direct.double().norm()
                ),
                "packing_roundtrip": True,
                "scale_finite": bool(torch.isfinite(scale).all()),
            }
            if entry["unquantized_qk_f64_max_abs"] > 1e-4 or not entry["scale_finite"]:
                raise ValueError(f"Rotation/packing invariant failed: {layer}")
            result["layers"][str(layer)] = entry
    write_json(root / "rotation_validation.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "freeze",
            "analyze",
            "heatmap",
            "diagnose",
            "validate",
            "invariants",
            "cpu_tables",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/mnt/d/ai-hub-models/binaries/turboquant/k_rotation_quality_20261006"
        ),
    )
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.root, args.source)
    else:
        globals()[args.stage](args.root)


if __name__ == "__main__":
    main()
