# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Freeze matched, fixed-budget PU-K8/LM placements across Qwen3 model sizes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from model_identity import QWEN_SHAPES

from qai_hub_models.models.templates.llm.turboquant.config import (
    mixed_kv_allocation_plan,
)

MODELS = ("qwen3_1_7b", "qwen3_4b", "qwen3_8b")
PLACEMENTS = {
    "early": "First q layers",
    "late": "Last q layers",
    "middle": "q consecutive layers starting at floor((L-q)/2)",
    "boundary": "First ceil(q/2) and last floor(q/2) layers",
    "interleaved": "Last layer of each consecutive group of L/q layers",
}


def _high_layers(num_layers: int, quota: int, placement: str) -> set[int]:
    """Absolute zero-based layer indices for the higher precision of one KV kind."""
    if quota == 0:
        return set()
    if placement == "early":
        return set(range(quota))
    if placement == "late":
        return set(range(num_layers - quota, num_layers))
    if placement == "middle":
        first = (num_layers - quota) // 2
        return set(range(first, first + quota))
    if placement == "boundary":
        return set(range((quota + 1) // 2)) | set(
            range(num_layers - quota // 2, num_layers)
        )
    period = num_layers // quota
    return set(range(period - 1, num_layers, period))


def layer_bits(
    num_layers: int, key_average_bits: int, placement: str
) -> list[list[int]]:
    """Build a joint K/V mask for equal-size KV layers at mean K+V = 8 bits."""
    if type(num_layers) is not int or num_layers <= 0 or num_layers % 4:
        raise ValueError("The layer count must be a positive multiple of four")
    if type(key_average_bits) is not int or key_average_bits not in (4, 5, 6):
        raise ValueError("Expected an average key precision of 4, 5, or 6")
    if placement not in PLACEMENTS:
        raise ValueError(f"Unknown placement {placement!r}")
    k8 = _high_layers(num_layers, num_layers * (key_average_bits - 4) // 4, placement)
    v4 = _high_layers(num_layers, num_layers * (6 - key_average_bits) // 2, placement)
    return [[8 if i in k8 else 4, 4 if i in v4 else 2] for i in range(num_layers)]


def model_plans(model_id: str) -> dict[str, dict[str, Any]]:
    if model_id not in MODELS:
        raise ValueError(f"Unknown study model {model_id!r}")
    num_layers, _, _, kv_heads, head_dim = QWEN_SHAPES[model_id]
    plans = {}
    for k in (4, 5, 6):
        for placement in PLACEMENTS:
            plan_id = f"k{k}_v{8 - k}" if k == 4 else f"k{k}_v{8 - k}_{placement}"
            # At K4/V4 all five placements have precisely the same bits.
            if plan_id in plans:
                continue
            bits = layer_bits(num_layers, k, placement)
            payload_bytes = sum((kb + vb) * head_dim * kv_heads // 8 for kb, vb in bits)
            # One effective FP16 scale per K and V head vector per layer.
            scale_bytes = 4 * kv_heads * num_layers
            plans[plan_id] = {
                **mixed_kv_allocation_plan(bits),
                "model_id": model_id,
                "plan_id": plan_id,
                "num_layers": num_layers,
                "placement": "all_layers" if k == 4 else placement,
                "placement_aliases": list(PLACEMENTS) if k == 4 else [placement],
                "key_average_bits": k,
                "value_average_bits": 8 - k,
                "k8_layers": [i for i, (kb, _) in enumerate(bits) if kb == 8],
                "v4_layers": [i for i, (_, vb) in enumerate(bits) if vb == 4],
                "cache_bytes_per_token": {
                    "indices": payload_bytes,
                    "fp16_scales": scale_bytes,
                    "total": payload_bytes + scale_bytes,
                },
            }
    return plans


def _freeze_json(path: Path, record: dict) -> str:
    """Publish complete files atomically. Identical retries reuse the frozen copy."""
    data = (
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            path.hardlink_to(tmp.name)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Existing frozen file differs: {path}") from None
    return hashlib.sha256(data).hexdigest()


def freeze_plans(out: Path) -> Path:
    files = {}
    for model_id in MODELS:
        for plan_id, record in model_plans(model_id).items():
            name = f"plans/{model_id}/{plan_id}.json"
            files[name] = _freeze_json(out / name, record)
    protocol = {
        "schema_version": 1,
        "models": list(MODELS),
        "average_key_value_bits": [[4, 4], [5, 3], [6, 2]],
        "key_codecs": {"4": "Lloyd-Max", "8": "PU (structured_k8_two_region_v1)"},
        "value_codecs": {"2": "Lloyd-Max", "4": "Lloyd-Max"},
        "placements": PLACEMENTS,
        "assignment": (
            "For L equal-size KV layers allocate K8 to L*(Kmean-4)/4 layers "
            "and V4 to L*(Vmean-2)/2 layers. Apply the same placement to each "
            "quota independently. Layer indices are absolute and zero-based."
        ),
        "comparison": (
            "Report every distinct mask on the same four historical WikiText "
            "1024-token windows. Compare token-weighted mean next-token NLL and "
            "its exponential (PPL) within each model. These windows are a "
            "descriptive benchmark. No outcome-based mask selection is performed."
        ),
        "plan_files_sha256": files,
    }
    manifest = out / "selection_protocol.json"
    _freeze_json(manifest, protocol)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(freeze_plans(args.out))


if __name__ == "__main__":
    main()
