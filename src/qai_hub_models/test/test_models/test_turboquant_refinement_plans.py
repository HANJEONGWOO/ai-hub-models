# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Exact quotas, paired placement controls and identities in the refinement cohort."""

import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

from qai_hub_models.models.templates.llm.turboquant.config import get_profile


@pytest.fixture
def refinement(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("refinement_plans")


@pytest.mark.parametrize(
    ("model_id", "n", "initial_k8", "initial_v4"),
    [("qwen3_1_7b", 28, 7, 14), ("qwen3_4b", 36, 9, 18), ("qwen3_8b", 36, 18, 0)],
)
def test_cohort(
    refinement: ModuleType, model_id: str, n: int, initial_k8: int, initial_v4: int
) -> None:
    plans = refinement.model_plans(model_id)
    anchor = plans["anchor"]
    assert anchor["k8_layers"] == list(range(initial_k8))
    assert anchor["v4_layers"] == list(range(initial_v4))
    assert anchor["config_hash"] == anchor["anchor_config_hash"]
    assert Counter(p["category"] for p in plans.values()) == {
        "anchor": 1,
        "budget_only": 2 if initial_v4 else 1,
        "placement_only": 6 if initial_v4 else 3,
        "budget_and_placement": 4,
    }
    assert len({p["config_hash"] for p in plans.values()}) == len(plans)
    for plan_id, plan in plans.items():
        bits = plan["bits"]
        cfg = get_profile(plan["profile"], layer_plan=plan)
        cfg.validate_for_model(n, 8, 128)
        assert plan["plan_id"] == plan_id
        assert plan["config_hash"] == cfg.config_hash()
        assert sum(k + v for k, v in bits) == n * 8
        assert plan["key_average_bits"] == sum(k for k, _ in bits) / n
        assert plan["value_average_bits"] == sum(v for _, v in bits) / n
        assert plan["cache_bytes_per_token"] == {
            "indices": n * 8 * 128,
            "fp16_scales": n * 8 * 4,
            "total": n * 8 * 132,
        }
        step = plan["budget_step_k8"]
        assert step in (-1, 0, 1)
        assert len(plan["k8_layers"]) == initial_k8 + step
        assert len(plan["v4_layers"]) == initial_v4 - 2 * step
        assert plan["k8_layers"] == [i for i, (k, _) in enumerate(bits) if k == 8]
        assert plan["v4_layers"] == [i for i, (_, v) in enumerate(bits) if v == 4]
        control = plans[plan["placement_control_id"]]
        assert control["budget_step_k8"] == step
        swap = plan["placement_swap"]
        for kind in ("k8", "v4"):
            mask = set(plan[kind + "_layers"])
            old = set(anchor[kind + "_layers"])
            assert plan["delta_from_anchor"][kind + "_added"] == sorted(mask - old)
            assert plan["delta_from_anchor"][kind + "_removed"] == sorted(old - mask)
            prefix = set(range(len(mask)))
            assert set(control[kind + "_layers"]) == prefix
            if swap and swap["kind"] == kind:
                assert prefix - mask == {swap["from_layer"]} == {len(mask) - 1}
                assert mask - prefix == {swap["to_layer"]}
            else:
                assert mask == prefix
        assert (control == plan) == (swap is None)


def test_v_placement_at_8b_boundary(refinement: ModuleType) -> None:
    plans = refinement.model_plans("qwen3_8b")
    assert not any(p["budget_step_k8"] > 0 for p in plans.values())
    assert not any(p.startswith("anchor__v4") for p in plans)
    for location, target in [("adjacent", 2), ("midpoint", 18), ("last", 35)]:
        plan = plans[f"kminus1_vplus2__v4_{location}"]
        assert plan["v4_layers"] == [0, target]
        assert plan["k8_layers"] == list(range(17))


def test_freeze(refinement: ModuleType, tmp_path: Path) -> None:
    manifest = refinement.freeze_plans(tmp_path)
    first = manifest.read_bytes()
    files = json.loads(first)["plan_files_sha256"]
    assert len(files) == 35
    assert refinement.freeze_plans(tmp_path).read_bytes() == first
    for name, sha in files.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == sha
    changed = tmp_path / next(iter(files))
    changed.write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        refinement.freeze_plans(tmp_path)
    assert changed.read_text() == "{}"
