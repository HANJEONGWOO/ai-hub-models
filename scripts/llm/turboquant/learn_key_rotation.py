# SPDX-License-Identifier: BSD-3-Clause
"""K-only Attention-MSE selection / projected STE training on actual W4A16 captures.

CPU is an offline training/validation tool, never an inference fallback. Every
forward uses hard LM bins. FP16 operation boundaries are emulated, not claimed
bit-exact to HTP reduction / transcendental kernels. The reference is captured
FP16-KV attention from the same W4A16 model, not an all-FP16 language model.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rotation_data import digest, write_json

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.reference import (
    DenseQRRotation,
    load_boundaries,
    load_codebook,
)
from qai_hub_models.models.templates.llm.turboquant.rotation_artifact import (
    load_rotation,
    save_rotation,
    with_key_rotation,
)

PROTOCOL = {
    "seed": 20261006,
    "model": "Qwen3-1.7B W4A16",
    "context": 1024,
    "candidate_seeds": [42, 43, 44, 45, 46, 47, 48, 49],
    "sharing": "one K rotation across all 28 layers and 8 KV heads; V seed 542 fixed",
    "selection": "validation mean squared Attention output error, hard forward",
    "optimizer": "Adam, lr=0.001, followed by FP64 polar/SVD orthogonal retraction every step",
    "gradient": "identity STE for hard centroid selection, FP16 rounding and affine quantization; differentiable norm correction",
    "steps": 240,
    "validation_every": 20,
    "patience": 4,
    "train_batch": "one uniformly sampled training window and layer, one AR128 chunk, 16 query positions, all heads",
    "evaluation_queries": "positions 31,63,...,1023; all 28 layers and all heads",
    "loss": "absolute per-element Attention output MSE against captured W4A16 FP16-KV output; V fixed TQ4",
    "performance_order": ["ABC", "BCA", "CAB"],
    "performance_repeats": 3,
    "performance_conditions": [
        "35 prompt + 128 generated",
        "897 prompt + 128 generated",
    ],
    "quality": "four heldout document windows; no test-based selection or tuning",
    "limitations": "CPU rounds operation outputs to FP16 with FP32 reductions; HTP accumulation and fused quantization may differ",
}


class HardForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, source: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
        return hard

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient, None


def rounded(x: torch.Tensor) -> torch.Tensor:
    """FP16 hard forward, identity straight-through gradient."""
    y = x.to(torch.float16).to(torch.float32)
    return HardForward.apply(x, y.detach()) if x.requires_grad else y


def affine(x: torch.Tensor, grid: tuple | None) -> torch.Tensor:
    if grid is None:
        return x
    scale, offset, bits = grid
    scale = scale[:, None, None]
    offset = offset[:, None, None]
    hard = (torch.round(x / scale - offset).clamp(0, 2**bits - 1) + offset) * scale
    return HardForward.apply(x, hard.detach()) if x.requires_grad else hard


CENTROIDS = torch.tensor(load_codebook(4, 128).astype(np.float16).astype(np.float32))
BOUNDARIES = torch.tensor(
    load_boundaries(4, 128).astype(np.float32).astype(np.float16).astype(np.float32)
)
VR = torch.tensor(DenseQRRotation(542, 128).matrix().astype(np.float32))


def encoder_centroids() -> torch.Tensor:
    """Existing encoder's affine-pair LUT has its own FP16 multiply/add rounds."""
    c = load_codebook(4, 128).astype(np.float32)
    values = []
    for i in range(16):
        magnitude = abs(i - 7.5) - 0.5
        offset = 2 * (int(magnitude) // 2)
        slope = np.float16(c[9 + offset] - c[8 + offset])
        base = np.float16(c[8 + offset] - offset * (c[9 + offset] - c[8 + offset]))
        value = np.float16(np.float16(magnitude * slope) + base)
        values.append(float(value) if i > 7 else -float(value))
    return torch.tensor(values)


ENCODER_CENTROIDS = encoder_centroids()


def codec(
    x: torch.Tensor, matrix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Actual hard 4-bit LM bins; robust norm, correction and native half product."""
    magnitude = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    large = magnitude > 256
    pre = torch.where(large, 1 / 256, 1.0)
    post = torch.where(large, 256.0, 1.0)
    xpre, maxpre = rounded(x * pre), rounded(magnitude * pre)
    scaled = rounded(xpre / torch.where(maxpre > 0, maxpre, 1.0))
    length = rounded(
        torch.sqrt(rounded(rounded(scaled * scaled).sum(-1, keepdim=True)))
    )
    unit = rounded(scaled / torch.where(length > 0, length, 1.0))
    norm = rounded(rounded(maxpre * length) * post)
    y = rounded(unit @ rounded(matrix).T)
    indices = torch.bucketize(y.detach().contiguous(), BOUNDARIES, right=False)
    chosen = CENTROIDS[indices]
    # Identity STE changes gradients only. Forward remains the selected LUT value.
    chosen = HardForward.apply(y, chosen) if y.requires_grad else chosen
    norm_chosen = ENCODER_CENTROIDS[indices]
    norm_chosen = HardForward.apply(y, norm_chosen) if y.requires_grad else norm_chosen
    normc = rounded(
        torch.sqrt(rounded(rounded(norm_chosen * norm_chosen).sum(-1, keepdim=True)))
    )
    scale = rounded(norm / torch.where(normc > 0, normc, 1.0))
    restored = rounded(chosen * scale)
    if not torch.isfinite(restored).all():
        raise ValueError("FP16 codec overflow")
    return restored, indices, scale


class Samples:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = json.loads((root / "samples/manifest.json").read_text())
        self.encodings = json.loads((root / "samples/encodings.json").read_text())
        if digest(root / "samples/encodings.json") != self.manifest["encodings_sha256"]:
            raise ValueError("Encoding checksum mismatch")
        for name, checksum in self.manifest["sha256"].items():
            if digest(root / "samples" / name) != checksum:
                raise ValueError(f"Changed capture {name}")

    @lru_cache(maxsize=280)
    def get(self, split: str, window: int, layer: int) -> dict:
        with np.load(
            self.root / "samples" / f"{split}_{window}_layer{layer:02d}.npz"
        ) as data:
            result = {
                k: torch.from_numpy(data[k].astype(np.float32)) for k in data.files
            }
        result["v_tq"] = codec(result["v"], VR)[0].repeat_interleave(2, dim=0)
        result["grids"] = {}
        for kind in ("score", "masked", "prob"):
            grids = [item[kind] for item in self.encodings[str(layer)]]
            if any(g["dtype"] != "INT" or g["bw"] != 16 for g in grids):
                raise ValueError("Unexpected attention boundary grid")
            result["grids"][kind] = (
                torch.tensor([g["scale"][0] for g in grids]),
                torch.tensor([g["offset"][0] for g in grids]),
                16,
            )
        return result


def attention(
    sample: dict, matrix: torch.Tensor, positions: torch.Tensor, quantized: bool = True
) -> torch.Tensor:
    """Preserve global mask/softmax, GQA, current quantization and four AV tiles."""
    if quantized:
        k = codec(sample["k"], matrix)[0].repeat_interleave(2, 0)
        q = rounded(sample["q"][:, positions] @ rounded(matrix).T)
        v = sample["v_tq"]
    else:
        k = sample["k"].repeat_interleave(2, 0)
        q = sample["q"][:, positions]
        v = sample["v"].repeat_interleave(2, 0)
    outputs = []
    keypos = torch.arange(1024)
    # positions sorted; each AR128 chunk has its own right-aligned cache layout.
    for chunk in torch.unique(positions // 128, sorted=True):
        chosen = positions // 128 == chunk
        pos = positions[chosen]
        end = (int(chunk) + 1) * 128
        scores = rounded(q[:, chosen] @ k[:, :end].transpose(-1, -2))
        scores = affine(scores, sample["grids"]["score"])
        mask = torch.where(keypos[:end][None, :] <= pos[:, None], 0.0, -100.0)
        masked = affine(scores + mask, sample["grids"]["masked"])
        prob = rounded(affine(torch.softmax(masked, dim=-1), sample["grids"]["prob"]))
        if quantized:
            total = None
            for stop in (256, 512, 768, 1024):
                start = max(0, stop - 256 - (1024 - end))
                stop = max(0, stop - (1024 - end))
                if start == stop:
                    partial = torch.zeros((*prob.shape[:-1], 128))
                else:
                    partial = rounded(prob[:, :, start:stop] @ v[:, start:stop])
                total = partial if total is None else rounded(total + partial)
            output = rounded(total @ rounded(VR))
        else:
            output = rounded(prob @ v[:, :end])
        outputs.append(output)
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def evaluate(
    samples: Samples,
    matrix: torch.Tensor,
    split: str,
    windows: int,
    reference_check: bool = False,
) -> dict:
    positions = torch.arange(31, 1024, 32)
    squared, reference_squared, count, max_abs = 0.0, 0.0, 0, 0.0
    by_layer = {}
    for layer in range(28):
        layer_sq, layer_count = 0.0, 0
        for window in range(windows):
            sample = samples.get(split, window, layer)
            output = attention(sample, matrix, positions, not reference_check)
            target = sample["o"][:, positions]
            error = (output - target).double()
            total = float(error.square().sum())
            squared += total
            layer_sq += total
            count += error.numel()
            layer_count += error.numel()
            reference_squared += float(target.double().square().sum())
            max_abs = max(max_abs, float(error.abs().max()))
        by_layer[str(layer)] = layer_sq / layer_count
    return {
        "mse": squared / count,
        "relative_l2": (squared / reference_squared) ** 0.5,
        "max_abs": max_abs,
        "elements": count,
        "layer_mse": by_layer,
    }


def train(root: Path) -> None:
    if (root / "rotations").exists():
        raise FileExistsError("Refusing to refit/overwrite selected rotations")
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol != PROTOCOL:
        raise ValueError("Frozen protocol differs")
    torch.manual_seed(PROTOCOL["seed"])
    torch.set_num_threads(8)
    samples = Samples(root)
    fidelity = evaluate(samples, torch.eye(128), "validation", 2, reference_check=True)
    write_json(root / "cpu_reference_validation.json", fidelity)
    print("REFERENCE FIDELITY", fidelity["relative_l2"], flush=True)
    if fidelity["relative_l2"] > 0.02:
        raise ValueError(
            "CPU FP16 attention differs excessively from captured HTP; fix the reference before training"
        )
    records = []
    for seed in PROTOCOL["candidate_seeds"]:
        matrix = torch.tensor(DenseQRRotation(seed, 128).matrix().astype(np.float32))
        stats = evaluate(samples, matrix, "validation", 2)
        records.append({"seed": seed, **stats})
        print("CANDIDATE", seed, stats["mse"], flush=True)
    best_seed = min(records, key=lambda x: x["mse"])["seed"]
    baseline = DenseQRRotation(42, 128).matrix().astype(np.float32)
    initial = DenseQRRotation(best_seed, 128).matrix().astype(np.float32)
    common = {
        "protocol_sha256": digest(root / "protocol.json"),
        "sample_manifest_sha256": digest(root / "samples/manifest.json"),
    }
    save_rotation(
        root / "rotations/A.json",
        baseline,
        {**common, "method": "baseline", "seed": 42},
    )
    save_rotation(
        root / "rotations/B.json",
        initial,
        {**common, "method": "validation_random_selection", "seed": best_seed},
    )
    write_json(
        root / "selection.json", {"candidates": records, "selected_seed": best_seed}
    )
    parameter = torch.nn.Parameter(torch.tensor(initial))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    best_value = min(x["mse"] for x in records)
    best_matrix = initial.copy()
    history = [{"step": 0, "validation_mse": best_value}]
    rng = np.random.default_rng(PROTOCOL["seed"])
    stale = 0
    start = time.monotonic()
    for step in range(1, PROTOCOL["steps"] + 1):
        window, layer, chunk = (
            int(rng.integers(4)),
            int(rng.integers(28)),
            int(rng.integers(8)),
        )
        positions = torch.tensor(
            np.sort(rng.choice(128, 16, replace=False)) + chunk * 128
        )
        sample = samples.get("train", window, layer)
        optimizer.zero_grad()
        output = attention(sample, parameter, positions)
        loss = (output - sample["o"][:, positions]).square().mean()
        loss.backward()
        if not torch.isfinite(parameter.grad).all():
            raise ValueError("Nonfinite surrogate gradient")
        optimizer.step()
        with torch.no_grad():
            u, _, vh = torch.linalg.svd(parameter.double(), full_matrices=False)
            parameter.copy_((u @ vh).float())
        if step % PROTOCOL["validation_every"] == 0:
            stats = evaluate(samples, parameter.detach(), "validation", 2)
            history.append(
                {
                    "step": step,
                    "train_batch_mse": float(loss.detach()),
                    "validation_mse": stats["mse"],
                    "elapsed_s": time.monotonic() - start,
                }
            )
            print("TRAIN", history[-1], flush=True)
            if stats["mse"] < best_value:
                best_value, best_matrix, stale = (
                    stats["mse"],
                    parameter.detach().numpy().copy(),
                    0,
                )
            else:
                stale += 1
            if stale >= PROTOCOL["patience"]:
                break
    save_rotation(
        root / "rotations/C.json",
        best_matrix,
        {
            **common,
            "method": "hard_forward_STE_projected_Adam",
            "initial_seed": best_seed,
            "selected_step": min(history, key=lambda x: x["validation_mse"])["step"],
        },
    )
    write_json(
        root / "training.json",
        {
            "history": history,
            "best_validation_mse": best_value,
            "early_stopped": step < PROTOCOL["steps"],
            "no_global_optimum_claim": True,
        },
    )


def heldout(root: Path) -> None:
    torch.set_num_threads(8)
    samples = Samples(root)
    report = {"protocol_sha256": digest(root / "protocol.json"), "groups": {}}
    rotation_checks = {}
    for group in "ABC":
        matrix = torch.tensor(load_rotation(root / f"rotations/{group}.json")).reshape(
            128, 128
        )
        report["groups"][group] = evaluate(samples, matrix, "test", 4)
        sample = samples.get("test", 0, 0)
        q, k = sample["q"][0, :32].double(), sample["k"][0, :128].double()
        exact = q @ k.T
        rotated = (q @ matrix.double().T) @ (k @ matrix.double().T).T
        half_rotated = rounded(
            rounded(q.float() @ rounded(matrix).T)
            @ rounded(k.float() @ rounded(matrix).T).T
        )
        half_direct = rounded(q.float() @ k.float().T)
        config = with_key_rotation(
            get_profile("k4_v4_scaled"),
            None if group == "A" else root / f"rotations/{group}.json",
        )
        rotation_checks[group] = {
            "config_hash": config.config_hash(),
            "matrix_f32_sha256": config.to_dict()["key"]["rotation_f32_sha256"],
            "fp32_orthogonality_spectral": float(
                torch.linalg.matrix_norm(
                    matrix.double().T @ matrix.double() - torch.eye(128), ord=2
                )
            ),
            "fp16_orthogonality_spectral": float(
                torch.linalg.matrix_norm(
                    rounded(matrix).double().T @ rounded(matrix).double()
                    - torch.eye(128),
                    ord=2,
                )
            ),
            "unquantized_qk_f64_max_abs": float((rotated - exact).abs().max()),
            "unquantized_qk_fp16_max_abs": float(
                (half_rotated - half_direct).abs().max()
            ),
            "unquantized_qk_fp16_relative_l2": float(
                torch.linalg.vector_norm((half_rotated - half_direct).double())
                / torch.linalg.vector_norm(half_direct.double())
            ),
            "sharing": "one global 128x128 K matrix; V unchanged",
        }
        print("HELDOUT", group, report["groups"][group]["mse"], flush=True)
    report["fp16_cpu_vs_captured_htp"] = evaluate(
        samples, matrix, "test", 4, reference_check=True
    )
    report["c_improves_b"] = report["groups"]["C"]["mse"] < report["groups"]["B"]["mse"]
    write_json(root / "cpu_quality.json", report)
    write_json(root / "rotation_validation.json", rotation_checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "train", "heldout"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        write_json(args.root / "protocol.json", PROTOCOL)
    else:
        globals()[args.stage](args.root)


if __name__ == "__main__":
    main()
