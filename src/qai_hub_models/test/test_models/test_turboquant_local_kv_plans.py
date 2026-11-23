# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from qai_hub_models.models.templates.llm.turboquant.config import get_profile


@pytest.fixture
def allocator(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("local_kv_plans")


def test_scores_and_hashes(allocator: ModuleType) -> None:
    keys = list(range(-7, 21))
    values = [-i for i in range(28)]
    result = allocator.allocation_study(keys, values)
    assert result["k_rank"] == list(reversed(range(28)))
    assert result["v_rank"] == list(range(28))
    assert len(result["plans"]) == 16
    for k in range(15):
        plan = result["plans"][f"ranked_{k:02}"]
        assert plan["k8_layers"] == list(range(28 - k, 28))
        assert plan["v4_layers"] == list(range(28 - 2 * k))
        assert plan["predicted_gain"] == sum(keys[i] for i in plan["k8_layers"]) + sum(
            values[i] for i in plan["v4_layers"]
        )
        config = get_profile(plan["profile"], layer_plan=plan)
        assert config.config_hash() == plan["config_hash"]
        # Calculate bytes independently from the actual widths and tensor shape.
        data_bytes = sum((kb + vb) * 8 * 128 // 8 for kb, vb in plan["bits"])
        scale_bytes = 28 * 8 * 2 * 2
        assert plan["cache_bytes_per_token"] == {
            "indices": data_bytes,
            "fp16_scales": scale_bytes,
            "total": data_bytes + scale_bytes,
        }
        assert (data_bytes + scale_bytes) * 1024 / (1024**2) == 28.875


def test_overlap_dedup_and_heldout_nomination(allocator: ModuleType) -> None:
    # K8 favors precisely the Early prefix; K4/V4 are preferred elsewhere.
    out = allocator.allocation_study([4] * 7 + [-20] * 21, [1] * 28)
    assert out["best_k"] == 7
    assert out["k_rank"] == out["v_rank"] == list(range(28))
    early_hash = out["plans"]["early"]["config_hash"]
    assert early_hash == out["plans"]["ranked_07"]["config_hash"]
    assert out["validation_ids"] == ["early", "ranked_06", "ranked_08"]
    scores = dict.fromkeys(out["validation_ids"], 3.0)
    assert allocator.validation_winner(out, scores) == "early"
    scores["ranked_06"] = scores["ranked_08"] = 2.9
    assert allocator.validation_winner(out, scores) == "ranked_06"
    with pytest.raises(ValueError, match="every nominated plan"):
        allocator.validation_winner(out, {"early": 3.0})


@pytest.mark.parametrize(("key", "optimum"), [(0, 0), (1, 14)])
def test_boundary_quotas(allocator: ModuleType, key: int, optimum: int) -> None:
    study = allocator.allocation_study([key] * 28, [0] * 28)
    assert study["best_k"] == optimum
    assert len(study["validation_ids"]) == 3


def test_missing_and_nonfinite_scores(allocator: ModuleType) -> None:
    with pytest.raises(ValueError, match="28 K gains"):
        allocator.allocation_study([0] * 27, [0] * 28)
    with pytest.raises(ValueError, match="finite"):
        allocator.allocation_study([0] * 28, [float("nan")] * 28)
