# SPDX-License-Identifier: BSD-3-Clause
"""Allocate Qwen3-1.7B K/V widths using measured four-state relative errors."""

from __future__ import annotations

import math
from collections.abc import Sequence

from local_kv_plans import MODEL_ID, NUM_LAYERS
from model_identity import QWEN_SHAPES
from select_mixed_layers import OPTIONS, choose

from qai_hub_models.models.templates.llm.turboquant.config import (
    mixed_kv_allocation_plan,
)


def allocation_study(
    costs: Sequence[Sequence[float]], old_plans: dict, *, expand: bool = False
) -> dict:
    """Build the fixed K5/V3 comparison, or the expanded equal-byte quota sweep.

    Columns follow OPTIONS: K4/V2, K4/V4, K8/V2, K8/V4. Each row contains
    one directly measured error per state. No inter-state additivity is assumed.
    """
    rows = [[float(value) for value in row] for row in costs]
    if len(rows) != NUM_LAYERS or any(len(row) != len(OPTIONS) for row in rows):
        raise ValueError("Expected a 28 x 4 relative-error table")
    if not all(math.isfinite(value) and value >= 0 for row in rows for value in row):
        raise ValueError("Relative errors must be finite and nonnegative")
    _, _, _, heads, dim = QWEN_SHAPES[MODEL_ID]

    def make_plan(name: str, bits: Sequence[Sequence[int]], k: int) -> dict:
        plan = mixed_kv_allocation_plan([list(pair) for pair in bits])
        kb = [i for i, (a, _) in enumerate(bits) if a == 8]
        vb = [i for i, (_, b) in enumerate(bits) if b == 4]
        if len(bits) != NUM_LAYERS or len(kb) != k or len(vb) != NUM_LAYERS - 2 * k:
            raise ValueError(f"Plan {name} violates the layer quotas")
        packed = sum((a + b) * heads * dim // 8 for a, b in bits)
        scales = NUM_LAYERS * heads * 4  # FP16 K and V scales per head and layer
        return {
            **plan,
            "plan_id": name,
            "model_id": MODEL_ID,
            "num_layers": NUM_LAYERS,
            "k8_layers": kb,
            "v4_layers": vb,
            "key_average_bits": 4 + 4 * len(kb) / NUM_LAYERS,
            "value_average_bits": 2 + 2 * len(vb) / NUM_LAYERS,
            "cache_bytes_per_token": {
                "indices": packed,
                "fp16_scales": scales,
                "total": packed + scales,
            },
            "sum_relative_error": math.fsum(
                row[OPTIONS.index(tuple(pair))]
                for row, pair in zip(rows, bits, strict=True)
            ),
        }

    plans = {}
    for name, previous_id in (("early", "early"), ("absolute_07", "ranked_07")):
        historical = old_plans["plans"][previous_id]
        control = make_plan(name, historical["bits"], 7)
        if control["config_hash"] != historical["config_hash"]:
            raise ValueError(f"Changed frozen control: {previous_id}")
        if name == "early" and (
            control["k8_layers"] != list(range(7))
            or control["v4_layers"] != list(range(14))
        ):
            raise ValueError("Expected historical Early prefix masks")
        plans[name] = control

    quotas = list(range(NUM_LAYERS // 2 + 1)) if expand else [7]
    for k in quotas:
        bits = choose(rows, k, NUM_LAYERS - 2 * k)
        name = f"relative_{k:02}"
        plans[name] = make_plan(name, bits, k)
    best_k = min(quotas, key=lambda k: plans[f"relative_{k:02}"]["sum_relative_error"])
    candidates = ["early", "absolute_07", "relative_07", f"relative_{best_k:02}"]
    seen = set()
    selected = []
    for name in candidates:
        identity = plans[name]["config_hash"]
        if identity not in seen:
            selected.append(name)
            seen.add(identity)
    return {
        "states": [list(pair) for pair in OPTIONS],
        "quotas": quotas,
        "best_k": best_k,
        "candidate_ids": selected,
        "plans": plans,
    }
