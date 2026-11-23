# SPDX-License-Identifier: BSD-3-Clause
"""Stream full-window local KV errors from the unchanged offline W4A16 FP16-KV branch.

The reference and candidate paths consume identical post-RoPE Q and current K/V
at each layer; only the reference AV is propagated through the checkpoint.
FP32 reductions and explicit FP16/affine boundaries approximate the HTP engine.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import torch
from allocation_offline_model import Attention, Model, codec, rotation, rounded
from compare_cross_model_budgets import digest, read, sources
from cross_model_plans import _freeze_json

ARCHIVE = Path("/mnt/d/ai-hub-models/binaries/turboquant")
OPTIONS = ((4, 2), (8, 2), (4, 4))
NAMES = ("E42", "E82", "E44")


class ReferenceAttention(Attention):
    """AR128 reference AV and aligned local TQ AVs, before the output projection."""

    def probabilities(
        self, q: torch.Tensor, key: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        scores = self.affine(rounded(q[:, start:end] @ key[:, :end].transpose(1, 2)), 0)
        positions = torch.arange(end, device=q.device)
        mask = torch.where(
            positions[None, :] <= positions[start:end, None], 0.0, -100.0
        )
        masked = self.affine(scores + mask, 1)
        if end < 1024:
            padding = masked.new_full((*masked.shape[:2], 1024 - end), -100.0)
            masked = torch.cat((self.affine(padding, 1), masked), dim=-1)
        return rounded(self.affine(torch.softmax(masked, dim=-1), 2))

    def reference_av(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        parts = []
        for start in range(0, 1024, 128):
            end = start + 128
            prob = self.probabilities(q, k, start, end)
            parts.append(rounded(prob[:, :, 1024 - end :] @ v[:, :end]))
        return torch.cat(parts, 1)

    def local_av(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> dict[tuple[int, int], torch.Tensor]:
        groups = q.shape[0] // k.shape[0]
        qr = rounded(q @ rotation(42, self.device).T)
        keys = {
            b: codec(k, b, 42, structured=b == 8).repeat_interleave(groups, 0)
            for b in (4, 8)
        }
        values = {b: codec(v, b, 542).repeat_interleave(groups, 0) for b in (2, 4)}
        outputs = {state: [] for state in OPTIONS}
        for start in range(0, 1024, 128):
            end = start + 128
            pad = 1024 - end
            probs = {
                b: self.probabilities(qr, key, start, end) for b, key in keys.items()
            }
            for state, blocks in outputs.items():
                kb, vb = state
                total = None
                for edge in (256, 512, 768, 1024):
                    first, last = max(0, edge - 256 - pad), max(0, edge - pad)
                    partial = (
                        rounded(
                            probs[kb][:, :, pad + first : pad + last]
                            @ values[vb][:, first:last]
                        )
                        if first < last
                        else torch.zeros_like(qr[:, start:end])
                    )
                    total = partial if total is None else rounded(total + partial)
                blocks.append(rounded(total @ rotation(542, self.device)))
        return {state: torch.cat(blocks, 1) for state, blocks in outputs.items()}

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
            raise ValueError(
                f"Unexpected reference tensor shapes at layer {self.layer}"
            )
        ref = self.reference_av(q, k.repeat_interleave(2, 0), v.repeat_interleave(2, 0))
        local = self.local_av(q, k, v)
        self.errors = torch.stack(
            [(local[s] - ref).double().square().mean() for s in OPTIONS]
        )
        self.calls += 1
        return list(ref.unsqueeze(1).split(1, 0))


def attention_specs(source: Path, tq_source: Path, destination: Path) -> list:
    """Keep the FP16 export's QKV producers and exact AV consumers/encodings.

    Replace only the verified per-head QK -> mask -> softmax -> AV pattern
    with an offline full-document macro; Part's dependency closure removes the
    unused external past-KV and the old score branches. All data blobs are shared.
    """
    model = onnx.load(source, load_external_data=False)
    graph = model.graph
    producer = {out: node for node in graph.node for out in node.output}
    tq_graph = onnx.load(tq_source, load_external_data=False).graph
    tq_producer = {out: node for node in tq_graph.node for out in node.output}
    tq_acts = {
        a["name"]: a
        for a in read(tq_source.with_suffix(".encodings"))["activation_encodings"]
    }
    acts = {
        a["name"]: a
        for a in read(source.with_suffix(".encodings"))["activation_encodings"]
    }
    if not np.isclose(acts["attention_mask"]["scale"][0] * 65535, 100):
        raise ValueError("Unsupported graph attention mask")
    layers = defaultdict(list)
    for item in read(source.with_suffix(".kv_edits.json"))["fp16_attention"]:
        layers[item["layer"]].append(item)
    specs, added, replaced = [], [], set()
    for layer, heads in sorted(layers.items()):
        inputs, outputs, grids = [], [], []
        for a in sorted(heads, key=lambda h: (h["head"], h["group"])):
            head, group, score, av = a["head"], a["group"], a["qk"], a["av"]
            if 2 * head + group != len(inputs):
                raise ValueError(f"Unsupported GQA mapping at layer {layer}")
            inputs.append(score["lhs"])
            outputs.append(av["output"])
            prob = producer[av["lhs"]].input[0]
            masked = producer[prob].input[0]
            pattern = [
                producer[n].op_type
                for n in (
                    score["lhs"],
                    score["output"],
                    score["boundary"],
                    masked,
                    prob,
                    av["lhs"],
                    av["output"],
                )
            ]
            if (
                pattern
                != ["Cast", "MatMul", "Cast", "Add", "Softmax", "Cast", "MatMul"]
                or list(producer[masked].input) != [score["boundary"], "attention_mask"]
                or not score["rhs"].endswith("_key_cat")
                or not av["rhs"].endswith("_value_cat")
            ):
                raise ValueError(f"Unexpected FP16 attention graph at layer {layer}")
            tq_prefix = f"tq_attn_{layer}_tile0_head{head}_q{group}_"
            tq_prob = tq_producer[tq_prefix + "prob"].input[0]
            grid = [acts[n] for n in (score["boundary"], masked, prob)]
            tq_grid = [
                tq_acts[n]
                for n in (tq_prefix + "score", tq_producer[tq_prob].input[0], tq_prob)
            ]
            if any(
                ({**f, "name": ""} != {**t, "name": ""})
                for f, t in zip(grid, tq_grid, strict=True)
            ):
                raise ValueError(
                    f"FP16 and deployed TQ score grids differ at {layer}/{head}/{group}"
                )
            grids.append(grid)
        if len(inputs) != 16:
            raise ValueError("Expected 16 query heads")
        inputs.extend([f"past_key_{layer}_out", f"past_value_{layer}_out"])
        added.append(
            onnx.helper.make_node("OfflineAttention", inputs, outputs, layer=layer)
        )
        replaced.update(outputs)
        specs.append((layer, inputs, outputs, grids))
    keep = [n for n in graph.node if not replaced.intersection(n.output)]
    del graph.node[:]
    graph.node.extend(keep + added)
    destination.write_bytes(model.SerializeToString())
    destination.with_suffix(".encodings").symlink_to(source.with_suffix(".encodings"))
    externals = {
        e.value
        for ten in graph.initializer
        for e in ten.external_data
        if e.key == "location"
    }
    for name in externals:
        (destination.parent / name).symlink_to(source.parent / name)
    return specs


def historical_checks(archive: Path, macro: ReferenceAttention) -> dict:
    """Compare every query with archived HTP outputs; also compare the fused local reuse to the NLL scorer."""
    source = archive / "k_rotation_quality_20261006/samples/train_0_layer00.npz"
    with np.load(source) as arrays:
        q, k, v, observed = [
            torch.tensor(arrays[n], device=macro.device).float()
            for n in ("q", "k", "v", "o")
        ]
    ref = macro.reference_av(q, k.repeat_interleave(2, 0), v.repeat_interleave(2, 0))
    rel = float(
        torch.linalg.vector_norm(ref - observed) / torch.linalg.vector_norm(observed)
    )
    local = macro.local_av(q, k, v)
    qr = rounded(q @ rotation(42, macro.device).T).unsqueeze(1)
    inputs = [*qr.split(1), k.unsqueeze(1), v.unsqueeze(1)]
    errors = {}
    for state, name in zip(OPTIONS, NAMES, strict=True):
        out = Attention.__call__(
            macro,
            inputs,
            {0} if state[0] == 8 else set(),
            {0} if state[1] == 4 else set(),
        )
        err = float((torch.cat(out, 0).squeeze(1) - local[state]).abs().max())
        if err != 0:
            raise ValueError(
                f"Shared local path differs from full NLL macro: {name} {err}"
            )
        errors[name] = err
    # An earlier offline-to-HTP control used a 0.02 relative L2 tolerance.
    if not np.isfinite(rel) or rel > 0.02:
        raise ValueError(f"Archived FP16 HTP control mismatch, relative L2={rel}")
    return dict(
        archive=str(source),
        sha256=digest(source),
        layer=macro.layer,
        queries=1024,
        fp16_relative_l2=rel,
        local_vs_nll_macro_max_abs=errors,
    )


@torch.inference_mode()
def collect(root: Path, archive: Path, device: str) -> None:
    overall = time.monotonic()
    protocol = digest(root / "protocol.json")
    dataset = root / "data_manifest.json"
    docs = read(dataset)["windows"]["calibration"]
    if len(docs) != 8:
        raise ValueError("Need eight calibration documents")
    tq_bundle, assets, split = sources(archive, "qwen3_1_7b")
    fp_bundle = archive / "qwen3_1_7b_fp16_attention_20260929/bundle"
    if (
        read(fp_bundle / "convert_report.json")["profile"]
        != "baseline_fp16_kv_fp16_attn"
    ):
        raise ValueError("Incorrect FP16 reference bundle")
    src_hashes = {}
    for idx in (2, 3, 4):
        stem = f"prompt_ar128_cl1024_{idx}_of_4"
        for bundle, endings in (
            (fp_bundle, (".onnx", ".encodings", ".kv_edits.json")),
            (tq_bundle, (".onnx", ".encodings")),
        ):
            for extension in endings:
                item = bundle / (stem + extension)
                src_hashes[str(item)] = digest(item)
    for f in (
        split / "split_manifest.json",
        assets / "assets.json",
        fp_bundle / "convert_report.json",
        Path(__file__),
        Path(__file__).with_name("allocation_offline_model.py"),
    ):
        src_hashes[str(f.resolve())] = digest(f)
    packed = read(assets / "assets.json")
    rope = assets / packed["rope"]
    if digest(rope) != packed["sha256"][rope.name]:
        raise ValueError("RoPE hash mismatch")
    src_hashes[str(rope)] = packed["sha256"][rope.name]
    weights_identity = read(
        archive / "k_rotation_quality_20261006/capture_identity.json"
    )["source_files_sha256"]
    for idx in range(1, 5):
        part = read(split / "split_manifest.json")["parts"][f"part{idx}_of_4"]
        base = Path(part["bundle_dir"]) / part["class"]
        weight = base.with_suffix(".data")
        canonical = str(weight.resolve())
        hashed = digest(weight)
        if hashed != weights_identity[canonical]:
            raise ValueError(f"FP16 reference weight mismatch: {weight}")
        src_hashes[canonical] = hashed
        if idx == 1:
            for ext in (".onnx", ".encodings"):
                path = base.with_suffix(ext)
                src_hashes[str(path.resolve())] = digest(path)
    provenance = {
        "protocol_sha256": protocol,
        "data_manifest_sha256": digest(dataset),
        "source_sha256": src_hashes,
        "codec": "K4/V2/V4 LM, K8 PU, same codec() as full NLL",
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "torch": str(torch.__version__),
        "reference": "W4A16 pre-recalibration FP16-KV ONNX, original encoded weights, GPU affine/FP16 boundary replay of AR128, not bit-exact HTP",
        "aggregation": "mean squared error over 16 query heads x 1024 causal Q positions x 128 components, equal document weight",
    }
    identity = _freeze_json(root / "local_scoring_identity.json", provenance)
    with TemporaryDirectory(prefix="local-kv-attention-") as temp:
        temp_path = Path(temp)
        specs = {}
        for idx in (2, 3, 4):
            n = f"prompt_ar128_cl1024_{idx}_of_4.onnx"
            specs[idx] = attention_specs(fp_bundle / n, tq_bundle / n, temp_path / n)
        model = Model(
            temp_path,
            assets,
            device,
            split_dir=split,
            stream_parts=True,
            stream_weights=True,
        )
        for idx, part in enumerate(model.parts, start=1):
            part.to(device)
            for spec in specs.get(idx, []):
                macro = ReferenceAttention(*spec, device)
                macro.calls = 0
                part.macros[macro.layer] = macro
        model.stream_parts = False
        macros = [m for part in model.parts for _, m in sorted(part.macros.items())]
        if [m.layer for m in macros] != list(range(28)):
            raise ValueError("Expected exactly 28 layers")
        setup = time.monotonic() - overall
        checkstart = time.monotonic()
        control = historical_checks(archive, macros[0])
        control_elapsed = time.monotonic() - checkstart
        raw = []
        elapsed = []
        for rec in docs:
            file = rec["file"]
            sha = digest(root / "assets" / file)
            if sha != rec["tokens_sha256"]:
                raise ValueError(f"Token hash mismatch for {file}")
            out = root / "local_scores" / (Path(file).stem + ".json")
            if out.exists():
                saved = read(out)
                if (
                    saved["source_identity_sha256"] != identity
                    or saved["token_sha256"] != sha
                ):
                    raise ValueError(f"Stale cached score: {out}")
                raw.append(saved["mse"])
                elapsed.append(saved["elapsed_s"])
                continue
            started = time.monotonic()
            tokens = np.fromfile(root / "assets" / file, dtype="<i4")
            before = [m.calls for m in macros]
            logits = model.logits(tokens, [])
            if (
                not torch.isfinite(logits).all().item()
                or [m.calls - n for m, n in zip(macros, before, strict=True)]
                != [1] * 28
            ):
                raise ValueError("Invalid reference output or repeated/missing layer")
            del logits
            row = torch.stack([m.errors for m in macros]).cpu().numpy()
            if not np.isfinite(row).all() or np.min(row) < 0:
                raise ValueError("Invalid local MSE")
            dt = time.monotonic() - started
            saved = {
                "source_identity_sha256": identity,
                "file": file,
                "token_sha256": sha,
                "elapsed_s": dt,
                "reference_passes": 1,
                "layers": 28,
                "queries_per_layer": 1024,
                "mse_columns": NAMES,
                "mse": row.tolist(),
            }
            _freeze_json(out, saved)
            raw.append(row)
            elapsed.append(dt)
            print(file, f"{dt:.2f}s", flush=True)
    error = np.mean(raw, axis=0)
    summary = {
        "source_identity_sha256": identity,
        "mse_columns": NAMES,
        "mean_mse": error.tolist(),
        "key_gains": (error[:, 0] - error[:, 1]).tolist(),
        "value_gains": (error[:, 0] - error[:, 2]).tolist(),
        "documents": len(raw),
        "reference_passes": len(raw),
        "layers": 28,
        "queries_per_layer_per_document": 1024,
        "timing": {
            "setup_s": setup,
            "historical_control_s": control_elapsed,
            "reference_plus_local_per_document_s": elapsed,
            "reference_plus_local_sum_s": sum(elapsed),
            "setup_plus_scores_s": setup + sum(elapsed),
            "current_invocation_wall_s": time.monotonic() - overall,
        },
        "historical_control": control,
    }
    if (root / "local_scores.json").exists():
        saved = read(root / "local_scores.json")
        for k in ("source_identity_sha256", "key_gains", "value_gains"):
            if saved[k] != summary[k]:
                raise ValueError(f"Existing local summary conflicts: {k}")
    else:
        _freeze_json(root / "local_scores.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    collect(args.root, args.archive, args.device)
