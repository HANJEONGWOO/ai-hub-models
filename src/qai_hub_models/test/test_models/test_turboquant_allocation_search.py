# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Offline search rules, budget safety and checkpoint quantization replay."""

import importlib
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import (
    STRUCTURED_ALLOCATION_PROFILE,
    get_profile,
    structured_allocation_plan,
)


def test_cache_allocation_memory_and_incompatible_masks() -> None:
    configs = [
        get_profile(
            STRUCTURED_ALLOCATION_PROFILE,
            layer_plan=structured_allocation_plan(list(range(start, 28, 2))),
        )
        for start in (0, 1)
    ]
    caches = [TurboQuantKVCache(config, 28, 8, 128, 1024) for config in configs]
    assert all(cache.memory_report().allocated_bytes == 30277632 for cache in caches)
    with pytest.raises(ValueError, match="different TurboQuant config"):
        caches[1].load_state_dict(caches[0].state_dict())


@pytest.fixture
def search(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("search_nll_allocation")


def test_greedy_quota_decrease_and_bound(search: ModuleType) -> None:
    initial = list(range(1, 28, 2))

    def objective(layers: list[int]) -> float:
        return float(sum(layers))

    result = search.greedy(objective, initial, iterations=3)
    assert len(result["history"]) == 3
    assert result["final_nll"] < result["initial_nll"]
    for row in result["history"]:
        assert row["accepted"]
        assert len(row["promotions"]) == len(row["demotions"]) == 14
        assert len(row["swaps"]) == 16
        assert all(len(swap["k8_layers"]) == 14 for swap in row["swaps"])
        assert row["best_swap"]["mean_nll"] < row["before_nll"]


def test_greedy_no_improvement_retains_initial(search: ModuleType) -> None:
    initial = list(range(1, 28, 2))
    result = search.greedy(lambda _: 3.0, initial)
    assert result["final_layers"] == initial
    assert len(result["history"]) == 1 and not result["history"][0]["accepted"]
    assert search.mask_key(initial) != search.mask_key(list(range(0, 28, 2)))


def test_offline_codec_matches_existing_boundary_emulator(search: ModuleType) -> None:
    model = importlib.import_module("allocation_offline_model")
    old = importlib.import_module("select_mixed_layers")
    structured = importlib.import_module("evaluate_structured_k8_quality")
    rng = np.random.default_rng(8142)
    value = torch.tensor(
        rng.normal(size=(8, 5, 128)).astype(np.float16).astype(np.float32)
    )
    for bits in (2, 4):
        torch.testing.assert_close(
            model.codec(value, bits, 42), old.codec(value, bits, 42), rtol=0, atol=0
        )
    torch.testing.assert_close(
        model.codec(value, 8, 42, True),
        structured.structured_codec(value),
        rtol=0,
        atol=0,
    )


def test_offline_per_channel_weight_axis(search: ModuleType) -> None:
    model = importlib.import_module("allocation_offline_model")
    x = torch.tensor([[0.25, 0.5, 1.0], [0.1, 0.2, 0.3]])
    grid = model.Grid(
        {"bw": 4, "dtype": "INT", "scale": [0.125, 0.25, 0.5], "offset": [-8, -8, -8]},
        "cpu",
        2,
        1,
    )
    expected = torch.tensor([[0.25, 0.5, 1.0], [0.125, 0.25, 0.5]])
    torch.testing.assert_close(grid(x), expected)
