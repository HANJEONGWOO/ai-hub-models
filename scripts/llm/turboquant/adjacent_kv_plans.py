# SPDX-License-Identifier: BSD-3-Clause
"""Recover the six fixed Qwen3-1.7B layer masks for adjacent KV budgets."""

from __future__ import annotations

import csv
from pathlib import Path

from model_identity import QWEN_SHAPES, sha256_file

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import (
    get_profile,
    mixed_kv_allocation_plan,
)

MODEL_ID = "qwen3_1_7b"
CAPACITY = 1024
CACHE_BYTES = 30_277_632
QUOTA_SHA256 = "6547d93c8d1e10a989597e557868d668d2699e4964ad4a1cfe0ecf6a526a3211"
COMPARISON_SHA256 = "9ab956f259073ded484e31e0b0009db23beb1f3b80166d129f70728a4ed38795"


def build_plans(table_path: Path) -> dict[str, dict]:
    """Use the prior committed DP masks and matched Early prefixes; allocate each cache."""
    comparison_path = table_path.with_name("comparison.csv")
    if (
        sha256_file(table_path) != QUOTA_SHA256
        or sha256_file(comparison_path) != COMPARISON_SHA256
    ):
        raise ValueError("The original adjacent-quota or control table has changed")
    with table_path.open(newline="") as stream:
        quotas = {int(row["k8_count"]): row for row in csv.DictReader(stream)}
    with comparison_path.open(newline="") as stream:
        controls = {row["plan_id"]: row for row in csv.DictReader(stream)}

    layers, _, _, heads, dim = QWEN_SHAPES[MODEL_ID]
    plans = {}
    for k in (6, 7, 8):
        v = layers - 2 * k
        row = quotas[k]
        if int(row["v4_count"]) != v:
            raise ValueError(f"Invalid V4 quota for {k} K8 layers")
        for placement in ("early", "dp"):
            name = f"{placement}_{k:02d}"
            if placement == "early":
                keys, values = list(range(k)), list(range(v))
            else:
                keys = list(map(int, row["k8_layers"].split()))
                values = list(map(int, row["v4_layers"].split()))
            bits = [
                [8 if i in keys else 4, 4 if i in values else 2] for i in range(layers)
            ]
            if (
                [i for i, (kb, _) in enumerate(bits) if kb == 8] != keys
                or [i for i, (_, vb) in enumerate(bits) if vb == 4] != values
                or len(keys) != k
                or len(values) != v
            ):
                raise ValueError(f"Invalid absolute layer indices for {name}")
            plan = mixed_kv_allocation_plan(bits)
            if placement == "dp" and plan["config_hash"] != row["config_hash"]:
                raise ValueError(
                    f"Recovered DP mask disagrees with the stored hash: {name}"
                )
            if k == 7:
                control = controls["early" if placement == "early" else "relative_07"]
                if plan["config_hash"] != control["config_hash"]:
                    raise ValueError(f"Stored validation control disagrees with {name}")
            config = get_profile(plan["profile"], layer_plan=plan)
            cache = TurboQuantKVCache(config, layers, heads, dim, CAPACITY)
            allocated = cache.memory_report(CAPACITY).allocated_bytes
            if allocated != CACHE_BYTES:
                raise ValueError(
                    f"Incorrect allocated KV capacity for {name}: {allocated}"
                )
            plans[name] = {
                **plan,
                "model_id": MODEL_ID,
                "plan_id": name,
                "num_layers": layers,
                "k8_layers": keys,
                "v4_layers": values,
                "key_average_bits": 4 + 4 * k / layers,
                "value_average_bits": 2 + 2 * v / layers,
                "allocated_cache_bytes": allocated,
            }
    return plans
