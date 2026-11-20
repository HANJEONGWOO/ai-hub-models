# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Freeze a bounded quota and placement comparison around the Qwen3 Early anchors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cross_model_plans import MODELS, _freeze_json
from cross_model_plans import model_plans as previous_model_plans

from qai_hub_models.models.templates.llm.turboquant.config import (
    mixed_kv_allocation_plan,
)

ANCHORS = {
    "qwen3_1_7b": "k5_v3_early",
    "qwen3_4b": "k5_v3_early",
    "qwen3_8b": "k6_v2_early",
}
BUDGET_IDS = {0: "anchor", -1: "kminus1_vplus2", 1: "kplus1_vminus2"}


def model_plans(model_id: str) -> dict[str, dict[str, Any]]:
    """All scored candidates and their matched prefix controls, in absolute layers."""
    if model_id not in ANCHORS:
        raise ValueError(f"Unknown study model {model_id!r}")
    anchor_id = ANCHORS[model_id]
    anchor = previous_model_plans(model_id)[anchor_id]
    n = anchor["num_layers"]
    anchor_k = set(anchor["k8_layers"])
    anchor_v = set(anchor["v4_layers"])
    quotas = {}
    for step in BUDGET_IDS:
        k, v = len(anchor_k) + step, len(anchor_v) - 2 * step
        if 0 <= k <= n and 0 <= v <= n:
            quotas[step] = {"k8": k, "v4": v}

    # An all-low V anchor first permits a V placement comparison after one exchange.
    nearest = {
        kind: next(step for step, quota in quotas.items() if 0 < quota[kind] < n)
        for kind in ("k8", "v4")
    }
    plans = {}
    for step, quota in quotas.items():
        prefix_id = BUDGET_IDS[step]
        probes: list[tuple[str, str | None, int | None]] = [(prefix_id, None, None)]
        for kind, q in quota.items():
            if not 0 < q < n:
                continue
            targets = {"adjacent": q}
            if step == nearest[kind]:
                targets.update(midpoint=(q + n - 1) // 2, last=n - 1)
            for location, layer in targets.items():
                probes.append((f"{prefix_id}__{kind}_{location}", kind, layer))
        for plan_id, move_kind, target in probes:
            k8 = set(range(quota["k8"]))
            v4 = set(range(quota["v4"]))
            if move_kind is not None and target is not None:
                moved = k8 if move_kind == "k8" else v4
                moved.remove(quota[move_kind] - 1)
                moved.add(target)
            bits = [[8 if i in k8 else 4, 4 if i in v4 else 2] for i in range(n)]
            category = (
                ("budget_only" if step else "anchor")
                if move_kind is None
                else ("budget_and_placement" if step else "placement_only")
            )
            plans[plan_id] = {
                **mixed_kv_allocation_plan(bits),
                "model_id": model_id,
                "num_layers": n,
                "plan_id": plan_id,
                "category": category,
                "anchor_source_plan_id": anchor_id,
                "anchor_config_hash": anchor["config_hash"],
                "budget_step_k8": step,
                "placement_control_id": prefix_id,
                "placement_swap": (
                    {
                        "kind": move_kind,
                        "from_layer": quota[move_kind] - 1,
                        "to_layer": target,
                    }
                    if move_kind is not None
                    else None
                ),
                "delta_from_anchor": {
                    "k8_added": sorted(k8 - anchor_k),
                    "k8_removed": sorted(anchor_k - k8),
                    "v4_added": sorted(v4 - anchor_v),
                    "v4_removed": sorted(anchor_v - v4),
                },
                "key_average_bits": sum(k for k, _ in bits) / n,
                "value_average_bits": sum(v for _, v in bits) / n,
                "k8_layers": sorted(k8),
                "v4_layers": sorted(v4),
                "cache_bytes_per_token": anchor["cache_bytes_per_token"],
            }
    return plans


def freeze_plans(out: Path) -> Path:
    files = {}
    for model_id in MODELS:
        for plan_id, record in model_plans(model_id).items():
            path = f"plans/{model_id}/{plan_id}.json"
            files[path] = _freeze_json(out / path, record)
    protocol = {
        "schema_version": 1,
        "models": list(MODELS),
        "anchors": ANCHORS,
        "budget_rule": (
            "Use each model's Early anchor and each feasible one-step exchange of "
            "one K8 layer for two V4 layers in either direction. At each budget, "
            "start with independent K8 and V4 prefixes, using absolute zero-based "
            "layers. Every plan averages exactly 8 K+V payload bits per layer."
        ),
        "placement_rule": (
            "At each budget independently move the final high-precision K or V layer "
            "to its first low-precision layer, holding the other mask fixed. For each "
            "kind also try the lower midpoint and last layer of the low-precision "
            "suffix at the feasible budget closest to the anchor. Distances are "
            "measured in exchanges, with the V-richer budget first in a tie. "
            "A kind requires both high- and low-precision layers to be movable."
        ),
        "diagnostics": (
            "Compare each candidate with the anchor. To isolate a placement effect, "
            "also compare with placement_control_id at identical quotas. Prefix "
            "controls isolate the effect of a budget exchange under Early placement. "
            "At the 8B anchor, every V layer is V2, so V placement can be measured "
            "only after exchanging one K8 for two V4 layers."
        ),
        "selection": (
            "Select within each model by pooled token NLL on four new CL1024 "
            "calibration windows, favoring the anchor on an exact tie, then "
            "lexicographic candidate ID. Evaluate only the selected plan and anchor "
            "on eight separate held-out CL1024 windows. The evaluation protocol "
            "records input identities and scoring details."
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
