# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Offline full-model NLL sensitivity and bounded greedy swap selection."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from allocation_offline_model import Model
from benchmark_asymmetric_once import digest, read, save_new
from benchmark_nll_allocation import STATIC, validate

from qai_hub_models.models.templates.llm.turboquant.config import (
    structured_allocation_plan,
)


def mask_key(layers: list[int]) -> str:
    return f"{sum(1 << i for i in layers):07x}"


def greedy(
    score: Callable[[list[int]], float],
    initial: list[int],
    iterations: int = 3,
    shortlist: int = 4,
    epsilon: float = 1e-6,
    checkpoint: Callable[[dict], None] | None = None,
) -> dict:
    current = sorted(initial)
    history = []
    for step in range(iterations):
        baseline = score(current)
        promotions, demotions = [], []
        for layer in range(28):
            candidate = sorted(set(current) ^ {layer})
            value = score(candidate)
            (demotions if layer in current else promotions).append(
                {"layer": layer, "mean_nll": value, "delta_nll": value - baseline}
            )
        promotion = sorted(promotions, key=lambda x: (x["mean_nll"], x["layer"]))[
            :shortlist
        ]
        demotion = sorted(demotions, key=lambda x: (x["mean_nll"], x["layer"]))[
            :shortlist
        ]
        swaps = []
        for p in promotion:
            for d in demotion:
                mask = sorted((set(current) - {d["layer"]}) | {p["layer"]})
                assert len(mask) == 14
                swaps.append(
                    {
                        "promote": p["layer"],
                        "demote": d["layer"],
                        "k8_layers": mask,
                        "mean_nll": score(mask),
                    }
                )
        best = min(swaps, key=lambda x: (x["mean_nll"], x["promote"], x["demote"]))
        accepted = best["mean_nll"] < baseline - epsilon
        row = {
            "iteration": step + 1,
            "before_layers": current,
            "before_nll": baseline,
            "promotions": promotions,
            "demotions": demotions,
            "shortlist_promotions": promotion,
            "shortlist_demotions": demotion,
            "swaps": swaps,
            "best_swap": best,
            "accepted": accepted,
        }
        history.append(row)
        if checkpoint:
            checkpoint(row)
        if not accepted:
            break
        current = best["k8_layers"]
    return {
        "initial_layers": initial,
        "final_layers": current,
        "initial_nll": score(initial),
        "final_nll": score(current),
        "history": history,
    }


def kl_stats(reference: np.ndarray | torch.Tensor, candidate: torch.Tensor) -> dict:
    if not isinstance(reference, torch.Tensor):
        reference = torch.as_tensor(
            np.asarray(reference).copy(), device=candidate.device
        )
    # Chunk rows to avoid materializing several full-vocabulary distributions.
    values = []
    for start in range(0, len(candidate), 32):
        left = torch.log_softmax(reference[start : start + 32].double(), -1)
        right = torch.log_softmax(candidate[start : start + 32].double(), -1)
        values.extend((left.exp() * (left - right)).sum(-1).cpu().tolist())
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "per_token": values,
    }


class Scorer:
    def __init__(self, root: Path, model: Model) -> None:
        self.root, self.model = root, model
        self.tokens = [
            np.fromfile(root / f"assets/calibration_{i}.bin", dtype="<i4")
            for i in range(4)
        ]
        self.identity = read(root / "offline/identity.json")
        self.identity_hash = digest(root / "offline/identity.json")

    def __call__(self, layers: list[int]) -> float:
        layers = sorted(layers)
        directory = self.root / "offline/evaluations" / mask_key(layers)
        directory.mkdir(parents=True, exist_ok=True)
        sums = []
        for i, tokens in enumerate(self.tokens):
            path = directory / f"calibration_{i}.json"
            if path.exists():
                row = read(path)
                assert (
                    row["k8_layers"] == layers
                    and row["identity_sha256"] == self.identity_hash
                )
            else:
                row, logits = self.model.score(tokens, layers)
                del logits
                row.update(
                    k8_layers=layers,
                    identity_sha256=self.identity_hash,
                    input_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),
                )
                save_new(path, row)
                print(
                    "NLL",
                    mask_key(layers),
                    "doc",
                    i,
                    row["mean_nll"],
                    "seconds",
                    row["elapsed_s"],
                    flush=True,
                )
            sums.append(row["nll_sum"])
        return sum(sums) / (4 * 1023)


def load_model(root: Path) -> Model:
    path = root / "offline/prepared_w4a16.pt"
    if path.exists():
        # Trusted local artifact generated by this experiment, never untrusted input.
        model = torch.load(path, weights_only=False, map_location="cuda")
    else:
        model = Model(root / "uniform_odd", root / "assets")
        torch.save(model, path)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    return model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["preflight", "search", "static-offline"])
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    protocol = validate(root)
    model = load_model(root)
    if args.stage == "preflight":
        directory = root / "offline/preflight"
        directory.mkdir(exist_ok=False)
        for i in range(4):
            tokens = np.fromfile(root / f"assets/calibration_{i}.bin", dtype="<i4")
            value, logits = model.score(tokens, STATIC["uniform_odd"])
            report = read(root / f"reports/uniform_odd_calibration_{i}.json")
            reference = np.memmap(
                root / f"reports/uniform_odd_calibration_{i}.logits.bin",
                mode="r",
                dtype="<f4",
                shape=tuple(logits.shape),
            )
            kl = kl_stats(reference, logits)
            value.update(
                htp_mean_nll=report["nll_sum"] / 1023,
                delta_nll=value["mean_nll"] - report["nll_sum"] / 1023,
                kl_htp_to_offline=kl,
                k8_layers=STATIC["uniform_odd"],
                input_sha256=digest(root / f"assets/calibration_{i}.bin"),
            )
            save_new(directory / f"calibration_{i}.json", value)
            print(
                "PREFLIGHT",
                i,
                value["mean_nll"],
                value["htp_mean_nll"],
                "KL",
                kl["mean"],
                flush=True,
            )
        save_new(
            directory / "identity.json",
            {
                "model_sha256": digest(root / "offline/prepared_w4a16.pt"),
                "emulator_sha256": digest(
                    Path(__file__).with_name("allocation_offline_model.py")
                ),
                "protocol_sha256": digest(root / "protocol.json"),
                "parts": [part.identity for part in model.parts],
            },
        )
        return
    source = {
        str(path): digest(path)
        for path in [
            Path(__file__),
            Path(__file__).with_name("allocation_offline_model.py"),
        ]
    }
    identity = {
        "sources_sha256": source,
        "protocol_sha256": digest(root / "protocol.json"),
        "prepared_model_sha256": digest(root / "offline/prepared_w4a16.pt"),
        "preflight_identity_sha256": digest(root / "offline/preflight/identity.json"),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "tf32": False,
        "cudnn_deterministic": True,
        "limits": "Offline exported-weight W4A16 fake-quantized replay; AR128-aligned attention with explicit FP16 codec/tile boundaries, FP32 GEMM/reductions and affine-grid emulation, not bit-exact HTP",
    }
    if (root / "offline/identity.json").exists():
        assert identity == read(root / "offline/identity.json"), (
            "Offline scorer changed during selection"
        )
    else:
        save_new(root / "offline/identity.json", identity)
    scorer = Scorer(root, model)
    if args.stage == "search":
        search = protocol["search"]
        result = greedy(
            scorer,
            STATIC["uniform_odd"],
            search["max_iterations"],
            search["shortlist_per_direction"],
            search["accept_min_nll_decrease"],
            lambda row: save_new(
                root / "offline" / f"iteration_{row['iteration']}.json", row
            ),
        )
        result["identity_sha256"] = digest(root / "offline/identity.json")
        save_new(root / "offline/search.json", result)
        save_new(
            root / "plans/nll_candidate.json",
            structured_allocation_plan(result["final_layers"]),
        )
        print(
            "FINAL OFFLINE CANDIDATE",
            result["final_layers"],
            result["initial_nll"],
            result["final_nll"],
            flush=True,
        )
    else:
        result = {group: scorer(layers) for group, layers in STATIC.items()}
        old = read(
            root.parent
            / "hardware_aligned_mixed_kv_20261010/selection/mixed_k6_v2_mse.json"
        )
        mse = [i for i, pair in enumerate(old["bits"]) if pair[0] == 8]
        result["prior_attention_mse_mask_with_structured_k8"] = {
            "k8_layers": mse,
            "mean_nll": scorer(mse),
            "note": "Same prior mask, updated Structured K8 codec; offline comparison only, not an eighth HTP benchmark",
        }
        save_new(root / "offline/static_calibration.json", result)


if __name__ == "__main__":
    main()
