# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Rank layerwise K/V gains for Qwen3-1.7B under an exact cache budget."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import accumulate

from cross_model_plans import model_plans
from model_identity import QWEN_SHAPES

from qai_hub_models.models.templates.llm.turboquant.config import (
    mixed_kv_allocation_plan,
)

MODEL_ID = "qwen3_1_7b"
NUM_LAYERS = QWEN_SHAPES[MODEL_ID][0]


def allocation_study(
    key_gains: Sequence[float], value_gains: Sequence[float]
) -> dict:
    """Score arrays follow absolute layer order; equal gains favor earlier layers."""
    keys = [float(x) for x in key_gains]
    values = [float(x) for x in value_gains]
    if len(keys) != NUM_LAYERS or len(values) != NUM_LAYERS:
        raise ValueError(f"Expected {NUM_LAYERS} K gains and {NUM_LAYERS} V gains")
    if not all(math.isfinite(x) for x in keys + values):
        raise ValueError("Every layer gain must be finite")

    k_rank = sorted(range(NUM_LAYERS), key=keys.__getitem__, reverse=True)
    v_rank = sorted(range(NUM_LAYERS), key=values.__getitem__, reverse=True)
    k_prefix = list(accumulate((keys[i] for i in k_rank), initial=0.0))
    v_prefix = list(accumulate((values[i] for i in v_rank), initial=0.0))
    gains = [
        k_prefix[k] + v_prefix[NUM_LAYERS - 2 * k]
        for k in range(NUM_LAYERS // 2 + 1)
    ]
    # max returns the first maximum, i.e., the smaller k on an exact tie.
    best_k = max(range(len(gains)), key=gains.__getitem__)

    anchor = model_plans(MODEL_ID)["k5_v3_early"]
    early_gain = math.fsum(keys[i] for i in anchor["k8_layers"]) + math.fsum(
        values[i] for i in anchor["v4_layers"]
    )
    plans = {"early": {**anchor, "plan_id": "early", "predicted_gain": early_gain}}
    validation_ids = ["early"]
    for k, gain in enumerate(gains):
        v = NUM_LAYERS - 2 * k
        k8 = set(k_rank[:k])
        v4 = set(v_rank[:v])
        bits = [
            [8 if layer in k8 else 4, 4 if layer in v4 else 2]
            for layer in range(NUM_LAYERS)
        ]
        plan_id = f"ranked_{k:02}"
        plan = {
            **mixed_kv_allocation_plan(bits),
            "plan_id": plan_id,
            "model_id": MODEL_ID,
            "num_layers": NUM_LAYERS,
            "predicted_gain": gain,
            "key_average_bits": 4 + 4 * k / NUM_LAYERS,
            "value_average_bits": 2 + 2 * v / NUM_LAYERS,
            "k8_layers": sorted(k8),
            "v4_layers": sorted(v4),
            # The fixed bit budget preserves packed bytes and scale count.
            "cache_bytes_per_token": anchor["cache_bytes_per_token"],
        }
        plans[plan_id] = plan
        if abs(k - best_k) <= 1 and plan["config_hash"] != anchor["config_hash"]:
            validation_ids.append(plan_id)

    return {
        "best_k": best_k,
        "k_rank": k_rank,
        "v_rank": v_rank,
        "plans": plans,
        "validation_ids": validation_ids,
    }


def validation_winner(study: dict, mean_nll_by_id: Mapping[str, float]) -> str:
    """Select by pooled validation NLL; exact ties favor Early, then plan ID."""
    if set(mean_nll_by_id) != set(study["validation_ids"]):
        raise ValueError("Expected one validation NLL for every nominated plan")
    if not all(math.isfinite(x) for x in mean_nll_by_id.values()):
        raise ValueError("Expected finite validation NLL values")
    return min(
        mean_nll_by_id,
        key=lambda i: (mean_nll_by_id[i], i != "early", i),
    )
