# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""PC reference evaluation of TurboQuant KV profiles on a real Qwen3 checkpoint.

Everything here runs on the host (HF float model + float64 codec oracle). It
produces numerical reference values and KV snapshots for device validation; it
is NOT an on-device or NPU result.

Contract emulated (same as the delta-cache graphs): the prompt is fed in
``--chunk``-token pieces; attention sees decoded packed KV for earlier chunks
and exact KV for the current chunk, and only the new chunk is encoded.
Profiles whose K is BASELINE keep K in float here (the int8 KV of the deployed
graph needs the quantized model), so ``k8_*`` rows are "float K" PC proxies.

Usage:

    HF_HUB_OFFLINE=1 PYTHONPATH=src python scripts/llm/turboquant/evaluate_qwen3_kv.py \
        --model Qwen/Qwen3-1.7B --num-windows 4 \
        --report /tmp/claude/turboquant/qwen3_1_7b_eval.json \
        --snapshot ~/.qaihm/tmp/turboquant/qwen3_1_7b_kv_snapshot.npz
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicLayer

from qai_hub_models.models.templates.llm.turboquant.config import (
    Rotation,
    TurboQuantConfig,
    get_profile,
)
from qai_hub_models.models.templates.llm.turboquant.numerics import (
    HTP_FP16,
    reconstruction_stats,
)
from qai_hub_models.models.templates.llm.turboquant.packing import (
    FLOAT16_MAX,
    to_storage_norms,
)
from qai_hub_models.models.templates.llm.turboquant.reference import (
    PolarQuantReference,
)

SNAPSHOT_LAYERS = (0, 13, 27)
SNAPSHOT_TOKENS = 256


def codec_roundtrip(
    states: torch.Tensor, codec: PolarQuantReference | None, norm_dtype: str
) -> torch.Tensor:
    """Encode + decode ``(batch, heads, seq, head_dim)`` with stored-norm rounding."""
    if codec is None:
        return states
    x = states.detach().to("cpu", torch.float64).numpy()
    indices, norms = codec.encode(x)
    stored = to_storage_norms(norms, norm_dtype).astype(np.float64)
    decoded = codec.decode(indices, stored)
    return torch.from_numpy(decoded).to(states.device, states.dtype)


class PackedPastLayer(DynamicLayer):
    """Attention gets exact current KV; the stored past is the codec round trip."""

    def __init__(
        self,
        key_codec: PolarQuantReference | None,
        value_codec: PolarQuantReference | None,
        norm_dtype: str,
    ) -> None:
        super().__init__()
        self.key_codec = key_codec
        self.value_codec = value_codec
        self.norm_dtype = norm_dtype

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        attend_k = torch.cat([self.keys, key_states], dim=-2)
        attend_v = torch.cat([self.values, value_states], dim=-2)
        self.keys = torch.cat(
            [self.keys, codec_roundtrip(key_states, self.key_codec, self.norm_dtype)],
            dim=-2,
        )
        self.values = torch.cat(
            [
                self.values,
                codec_roundtrip(value_states, self.value_codec, self.norm_dtype),
            ],
            dim=-2,
        )
        return attend_k, attend_v


class PackedPastCache(Cache):
    def __init__(self, config: TurboQuantConfig, num_layers: int) -> None:
        def make(spec_name: str) -> PolarQuantReference | None:
            spec = getattr(config, spec_name)
            if not spec.is_polar:
                return None
            return PolarQuantReference(
                spec,
                config.block_size,
                config.rotation,
                config.norm_correction,
                config.precomputed_norm,
            )

        super().__init__(
            layers=[
                PackedPastLayer(make("key"), make("value"), config.norm_dtype)
                for _ in range(num_layers)
            ]
        )


def load_wikitext_tokens(tokenizer: Any) -> list[int]:
    path = hf_hub_download(
        repo_id="Salesforce/wikitext",
        repo_type="dataset",
        filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
    )
    text = "\n\n".join(pd.read_parquet(path)["text"].tolist())
    return tokenizer(text, add_special_tokens=False).input_ids


@torch.no_grad()
def run_window(
    model: Any, tokens: torch.Tensor, chunk: int, cache: Cache
) -> torch.Tensor:
    """Log-probs ``(seq, vocab)`` from chunked teacher-forced prefill."""
    log_probs = []
    for start in range(0, tokens.shape[1], chunk):
        out = model(
            input_ids=tokens[:, start : start + chunk],
            past_key_values=cache,
            use_cache=True,
        )
        log_probs.append(torch.log_softmax(out.logits[0].float(), dim=-1))
    return torch.cat(log_probs)


def kv_statistics(cache: Cache, config: TurboQuantConfig) -> dict[str, Any]:
    """Codec quality and float16 risk on the exact KV of one window."""
    stats: dict[str, Any] = {}
    for name in ("key", "value"):
        spec = getattr(config, name)
        codec = PolarQuantReference(
            spec,
            config.block_size,
            config.rotation,
            config.norm_correction,
            config.precomputed_norm,
        )
        per_layer = []
        max_norm = 0.0
        near_boundary = 0
        total = 0
        attr = "keys" if name == "key" else "values"
        for layer_idx, layer in enumerate(cache.layers):
            states = getattr(layer, attr)
            x = states.to("cpu", torch.float64).numpy()
            y, norms = codec.rotate_normalized(x)
            indices = np.searchsorted(codec.boundaries, y, side="left").astype(np.uint8)
            scalars = codec.encode(x)[1] if config.precomputed_norm else norms
            fits = float(scalars.max()) <= FLOAT16_MAX
            stored = to_storage_norms(scalars, config.norm_dtype if fits else "float32")
            row = reconstruction_stats(x, codec.decode(indices, stored))
            row["layer"] = layer_idx
            row["max_norm"] = float(norms.max())
            row["min_norm"] = float(norms.min())
            per_layer.append(row)
            max_norm = max(max_norm, row["max_norm"])
            distance = np.min(np.abs(y[..., None] - codec.boundaries), axis=-1)
            near_boundary += int(np.sum(distance <= HTP_FP16.boundary_distance))
            total += distance.size
        stats[name] = {
            "bits": spec.bits,
            "per_layer": per_layer,
            "rel_mse_mean": float(np.mean([r["rel_mse_mean"] for r in per_layer])),
            "cosine_mean": float(np.mean([r["cosine_mean"] for r in per_layer])),
            "cosine_min": float(np.min([r["cosine_min"] for r in per_layer])),
            "max_norm": max_norm,
            "max_norm_fits_float16": max_norm <= FLOAT16_MAX,
            "fraction_within_htp_boundary_tolerance": near_boundary / total,
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--profiles", nargs="+", default=["k4_v4", "k8_v3", "k4_v3"])
    parser.add_argument("--rotation", choices=[r.value for r in Rotation])
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    rotation = Rotation(args.rotation) if args.rotation else None
    configs = {
        name: get_profile(name, rotation) for name in {*args.profiles, "k4_v4", "k4_v3"}
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager"
    ).to(device)
    model.eval()
    num_layers = model.config.num_hidden_layers
    for profile in args.profiles:
        configs[profile].validate_for_model(
            num_layers, model.config.num_key_value_heads, model.config.head_dim
        )

    all_tokens = load_wikitext_tokens(tokenizer)
    report: dict[str, Any] = {
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "dtype": args.dtype,
        "window": args.window,
        "chunk": args.chunk,
        "num_windows": args.num_windows,
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test, non-overlapping windows",
        "scope": "PC reference (HF float model + float64 codec oracle); not a device result",
        "profiles": {},
    }
    sums = {name: {"nll": 0.0, "kl": 0.0, "top1": 0, "n": 0} for name in args.profiles}
    base_nll = 0.0
    base_n = 0

    for w in range(args.num_windows):
        window = all_tokens[w * args.window : (w + 1) * args.window]
        tokens = torch.tensor([window], device=device)
        targets = tokens[0, 1:]

        base_cache = PackedPastCache(get_profile("baseline_int8"), num_layers)
        base_lp = run_window(model, tokens, args.chunk, base_cache)[:-1]
        base_nll += float(-base_lp.gather(1, targets[:, None]).sum())
        base_n += targets.numel()

        if w == 0:
            report["kv_statistics"] = {
                profile: kv_statistics(base_cache, configs[profile])
                for profile in ("k4_v4", "k4_v3")
            }
            args.snapshot.expanduser().parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.snapshot.expanduser(),
                **{
                    f"layer{i}_{kind}": getattr(base_cache.layers[i], kind)[
                        0, :, :SNAPSHOT_TOKENS
                    ]
                    .to("cpu", torch.float32)
                    .numpy()
                    for i in SNAPSHOT_LAYERS
                    for kind in ("keys", "values")
                },
            )
        del base_cache

        for profile in args.profiles:
            cache = PackedPastCache(configs[profile], num_layers)
            lp = run_window(model, tokens, args.chunk, cache)[:-1]
            s = sums[profile]
            s["nll"] += float(-lp.gather(1, targets[:, None]).sum())
            s["kl"] += float(torch.sum(base_lp.exp() * (base_lp - lp)))
            s["top1"] += int(torch.sum(base_lp.argmax(-1) == lp.argmax(-1)))
            s["n"] += targets.numel()
            del cache
        print(f"window {w + 1}/{args.num_windows} done", flush=True)

    report["baseline_float_kv_ppl"] = math.exp(base_nll / base_n)
    for profile, s in sums.items():
        ppl = math.exp(s["nll"] / s["n"])
        report["profiles"][profile] = {
            "config_hash": configs[profile].config_hash(),
            "config": configs[profile].to_dict(),
            "ppl": ppl,
            "ppl_delta_pct_vs_float_kv": 100
            * (ppl / report["baseline_float_kv_ppl"] - 1),
            "kl_mean": s["kl"] / s["n"],
            "top1_agreement": s["top1"] / s["n"],
            "tokens_scored": s["n"],
        }
    report["snapshot"] = str(args.snapshot.expanduser())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: v for k, v in report.items() if k != "kv_statistics"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
