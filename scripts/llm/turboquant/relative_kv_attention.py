# SPDX-License-Identifier: BSD-3-Clause
"""Measure pooled four-state attention distortion on the shared FP16-KV trajectory."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from allocation_offline_model import Attention, Model, codec, rotation, rounded
from compare_cross_model_budgets import digest, read, sources
from cross_model_plans import _freeze_json
from local_kv_attention_scores import (
    ARCHIVE,
    ReferenceAttention,
    attention_specs,
    historical_checks,
)
from select_mixed_layers import OPTIONS

NUM_LAYERS = 28
ELEMENTS_PER_LAYER = 16 * 1024 * 128
EPSILON = 1e-12


class FourStateAttention(ReferenceAttention):
    """Replay both AV precisions with the same probabilities for each K precision."""

    def local_av(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> dict[tuple[int, int], torch.Tensor]:
        q_rot = rounded(q @ rotation(42, self.device).T)
        groups = q.shape[0] // k.shape[0]
        keys = {
            bits: codec(k, bits, 42, structured=bits == 8).repeat_interleave(groups, 0)
            for bits in (4, 8)
        }
        values = {
            bits: codec(v, bits, 542).repeat_interleave(groups, 0) for bits in (2, 4)
        }
        outputs: dict[tuple[int, int], list[torch.Tensor]] = {
            state: [] for state in OPTIONS
        }
        for start in range(0, 1024, 128):
            end = start + 128
            pad = 1024 - end
            for kb, key in keys.items():
                prob = self.probabilities(q_rot, key, start, end)
                for vb, val in values.items():
                    total = None
                    for off in range(0, 1024, 256):
                        first, last = max(0, off - pad), max(0, off + 256 - pad)
                        partial = (
                            rounded(
                                prob[:, :, pad + first : pad + last]
                                @ val[:, first:last]
                            )
                            if first < last
                            else torch.zeros_like(q[:, start:end])
                        )
                        total = partial if total is None else rounded(total + partial)
                    outputs[kb, vb].append(rounded(total @ rotation(542, self.device)))
        return {state: torch.cat(chunks, 1) for state, chunks in outputs.items()}

    def __call__(
        self,
        tensors: list[torch.Tensor],
        _layers: set[int],
        _v4_layers: set[int] | None = None,
    ) -> list[torch.Tensor]:
        q = torch.cat(tensors[:-2], 0).squeeze(1).float()
        k = tensors[-2].squeeze(1).transpose(1, 2).float()
        v = tensors[-1].squeeze(1).float()
        if (
            q.shape != (16, 1024, 128)
            or k.shape != (8, 1024, 128)
            or v.shape != k.shape
        ):
            raise ValueError(f"Invalid QKV at layer {self.layer}")
        target = self.reference_av(
            q, k.repeat_interleave(2, 0), v.repeat_interleave(2, 0)
        )
        approx = self.local_av(q, k, v)
        target64 = target.double()
        self.statistics = torch.stack(
            [
                target64.square().sum(),
                *((approx[s].double() - target64).square().sum() for s in OPTIONS),
            ]
        )
        self.calls += 1
        return list(target.unsqueeze(1).split(1, 0))


def verify_control(archive: Path, macro: FourStateAttention) -> dict:
    checks = historical_checks(archive, macro)
    with np.load(checks["archive"]) as arrays:
        q, k, v = [
            torch.tensor(arrays[n], device=macro.device).float()
            for n in ("q", "k", "v")
        ]
    rotated = rounded(q @ rotation(42, macro.device).T).unsqueeze(1)
    deployed = Attention.__call__(
        macro, [*rotated.split(1), k.unsqueeze(1), v.unsqueeze(1)], {0}, {0}
    )
    diff = float(
        (macro.local_av(q, k, v)[8, 4] - torch.cat(deployed, 0).squeeze(1)).abs().max()
    )
    if diff:
        raise ValueError(
            f"Shared K8/V4 probabilities produce a different AV than canonical replay: {diff}"
        )
    checks["local_vs_nll_macro_max_abs"]["E84"] = diff
    return checks


def validate_source(root: Path, prior: Path, archive: Path) -> tuple[list[dict], str]:
    if tuple(OPTIONS) != ((4, 2), (4, 4), (8, 2), (8, 4)):
        raise ValueError(
            "Relative error column order differs from four-state allocation solver"
        )
    previous = read(prior / "local_scoring_identity.json")
    old_summary = read(prior / "local_scores.json")
    if (
        digest(prior / "local_scoring_identity.json")
        != old_summary["source_identity_sha256"]
    ):
        raise ValueError("Prior FP16 graph identity mismatch")
    if digest(prior / "data_manifest.json") != previous["data_manifest_sha256"]:
        raise ValueError("Prior calibration data manifest mismatch")
    docs = read(root / "data_manifest.json")["windows"]["calibration"]
    prior_docs = read(prior / "data_manifest.json")["windows"]["calibration"]
    if len(docs) != 8 or [r["tokens_sha256"] for r in docs] != [
        r["tokens_sha256"] for r in prior_docs
    ]:
        raise ValueError("Expected the original eight calibration windows, in order")
    for name, expected in previous["source_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Changed calibration model artifact: {name}")
    inputs = {}
    for rec in docs:
        path = root / "assets" / rec["file"]
        if path.stat().st_size != 4096 or digest(path) != rec["tokens_sha256"]:
            raise ValueError(f"Invalid document: {path}")
        old = prior / "local_scores" / (Path(rec["file"]).stem + ".json")
        if read(old)["source_identity_sha256"] != old_summary["source_identity_sha256"]:
            raise ValueError(f"Changed prior MSE inputs or scorer: {old}")
        inputs[old.name] = digest(old)
    return docs, _freeze_json(
        root / "relative_scoring_identity.json",
        {
            "protocol_sha256": digest(root / "protocol.json"),
            "data_manifest_sha256": digest(root / "data_manifest.json"),
            "source_sha256": digest(Path(__file__)),
            "prior_calibration_identity_sha256": old_summary["source_identity_sha256"],
            "prior_raw_sha256": inputs,
            "archive": str(archive),
            "states": OPTIONS,
            "epsilon": EPSILON,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
        },
    )


def load_reference(
    archive: Path, scratch: Path, device: str
) -> tuple[Model, list[FourStateAttention]]:
    bundle, assets, split = sources(archive, "qwen3_1_7b")
    reference = archive / "qwen3_1_7b_fp16_attention_20260929/bundle"
    specs = {}
    for i in (2, 3, 4):
        filename = f"prompt_ar128_cl1024_{i}_of_4.onnx"
        specs[i] = attention_specs(
            reference / filename, bundle / filename, scratch / filename
        )
    model = Model(
        scratch, assets, device, split_dir=split, stream_parts=True, stream_weights=True
    )
    macros = []
    for i, part in enumerate(model.parts, start=1):
        part.to(device)
        for spec in specs.get(i, []):
            attn = FourStateAttention(*spec, device)
            attn.calls = 0
            part.macros[attn.layer] = attn
            macros.append(attn)
    model.stream_parts = False
    if [m.layer for m in macros] != list(range(NUM_LAYERS)):
        raise ValueError("Reference branch must visit all layers once")
    return model, macros


@torch.inference_mode()
def collect(root: Path, prior: Path, archive: Path) -> None:
    started = time.monotonic()
    docs, identity = validate_source(root, prior, archive)
    raw = []
    with TemporaryDirectory(prefix="relative-kv-attn-") as directory:
        model, macros = load_reference(archive, Path(directory), "cuda")
        setup = time.monotonic() - started
        control_start = time.monotonic()
        control = verify_control(archive, macros[0])
        control_s = time.monotonic() - control_start
        for doc in docs:
            stem = Path(doc["file"]).stem + ".json"
            dest = root / "calibration" / stem
            old = read(prior / "local_scores" / stem)
            if old["token_sha256"] != doc["tokens_sha256"] or old["mse_columns"] != [
                "E42",
                "E82",
                "E44",
            ]:
                raise ValueError(f"Unexpected archived three-state score for {stem}")
            if not dest.exists():
                begin = time.monotonic()
                before = [m.calls for m in macros]
                tokens = np.fromfile(root / "assets" / doc["file"], dtype="<i4")
                logits = model.logits(tokens, [])
                if (
                    not torch.isfinite(logits).all().item()
                    or [m.calls - n for m, n in zip(macros, before, strict=True)]
                    != [1] * NUM_LAYERS
                ):
                    raise ValueError(f"Invalid full-window FP16 replay: {stem}")
                del logits
                stats = torch.stack([m.statistics for m in macros]).cpu().numpy()
                # All paths round to FP16. The subtraction is now evaluated in
                # FP64; the historical implementation subtracted in FP32 first.
                if (
                    not np.isfinite(stats).all()
                    or (stats[:, 0] <= 0).any()
                    or (stats[:, 1:] < 0).any()
                ):
                    raise ValueError(f"Invalid squared sums: {stem}")
                if not np.allclose(
                    stats[:, [1, 3, 2]] / ELEMENTS_PER_LAYER,
                    np.asarray(old["mse"]),
                    rtol=1e-7,
                    atol=1e-14,
                ):
                    raise ValueError(
                        f"Three-state absolute MSE differs from prior calibration: {stem}"
                    )
                _freeze_json(
                    dest,
                    {
                        "identity": identity,
                        "file": doc["file"],
                        "tokens_sha256": doc["tokens_sha256"],
                        "states": OPTIONS,
                        "elements_per_layer": ELEMENTS_PER_LAYER,
                        "reference_energy": stats[:, 0].tolist(),
                        "error_sums": stats[:, 1:].tolist(),
                        "reference_passes": 1,
                        "elapsed_s": time.monotonic() - begin,
                    },
                )
            saved = read(dest)
            if (
                saved["identity"] != identity
                or saved["tokens_sha256"] != doc["tokens_sha256"]
            ):
                raise ValueError(f"Stale score: {dest}")
            raw.append(saved)
            print(stem, f"{saved['elapsed_s']:.2f}s", flush=True)
    energy = np.sum([r["reference_energy"] for r in raw], axis=0, dtype=np.float64)
    sums = np.sum([r["error_sums"] for r in raw], axis=0, dtype=np.float64)
    costs = sums / (energy[:, None] + EPSILON)
    previous = np.asarray(read(prior / "local_scores.json")["mean_mse"])
    mean_mse = sums / (len(raw) * ELEMENTS_PER_LAYER)
    max_diff = float(np.abs(mean_mse[:, [0, 2, 1]] - previous).max())
    if not np.allclose(mean_mse[:, [0, 2, 1]], previous, rtol=1e-7, atol=1e-14):
        raise ValueError("Aggregate MSE mismatch")
    summary = {
        "scoring_identity": identity,
        "states": OPTIONS,
        "relative_errors": costs.tolist(),
        "reference_energy": energy.tolist(),
        "error_sums": sums.tolist(),
        "epsilon": EPSILON,
        "reference_passes": len(raw),
        "elements_per_layer": len(raw) * ELEMENTS_PER_LAYER,
        "prior_absolute_mse_max_diff": max_diff,
        "historical_control": control,
        "documents_sha256": {
            r["file"]: digest(root / "calibration" / (Path(r["file"]).stem + ".json"))
            for r in raw
        },
    }
    out = root / "relative_scores.json"
    if out.exists():
        old_summary = read(out)
        if any(
            old_summary[k] != summary[k]
            for k in (
                "scoring_identity",
                "relative_errors",
                "reference_energy",
                "error_sums",
                "documents_sha256",
            )
        ):
            raise ValueError("Archived relative result conflicts")
    else:
        summary["timing"] = {
            "setup_s": setup,
            "historical_control_s": control_s,
            "calibration_forward_s": sum(r["elapsed_s"] for r in raw),
            "invocation_wall_s": time.monotonic() - started,
        }
        _freeze_json(out, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    collect(
        args.root, args.prior or args.archive / "local_kv_scores_20261123", args.archive
    )
