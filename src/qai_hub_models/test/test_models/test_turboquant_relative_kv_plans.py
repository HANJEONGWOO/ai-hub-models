# SPDX-License-Identifier: BSD-3-Clause

import importlib
import itertools
from pathlib import Path
from types import ModuleType

import pytest

from qai_hub_models.models.templates.llm.turboquant.config import get_profile


@pytest.fixture
def allocator(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("relative_kv_plans")


def previous() -> dict:
    return importlib.import_module("local_kv_plans").allocation_study(
        list(range(28)), list(range(28))
    )


def test_joint_states_against_brute_force(allocator: ModuleType) -> None:
    # The fourth state is deliberately much better than suggested by either
    # single-precision upgrade. Enumerate every path on a small instance.
    rows = [[9, 7, 8, 0], [4, 3, 5, 6], [3, 3, 4, 8], [7, 5, 6, 7]]
    paths = itertools.product(allocator.OPTIONS, repeat=4)
    solutions = [
        (sum(rows[i][allocator.OPTIONS.index(state)] for i, state in enumerate(p)), p)
        for p in paths
        if sum(k == 8 for k, _ in p) == 1 and sum(v == 4 for _, v in p) == 2
    ]
    assert allocator.choose(rows, 1, 2) == min(solutions)[1]


def test_quota_sweep_and_cache(allocator: ModuleType) -> None:
    # Joint state is cheapest for exactly seven layers. V4 costs nothing
    # on a further seven layers. The resulting fixed plan equals Early.
    rows = [[9, 6, 5, 0]] * 7 + [[5, 0, 6, 7]] * 7 + [[0, 4, 5, 9]] * 14
    old = previous()
    fixed = allocator.allocation_study(rows, old)
    expanded = allocator.allocation_study(rows, old, expand=True)
    assert fixed["quotas"] == [7] and expanded["quotas"] == list(range(15))
    assert fixed["best_k"] == expanded["best_k"] == 7
    assert fixed["candidate_ids"] == ["early", "absolute_07"]
    assert fixed["plans"]["relative_07"] == expanded["plans"]["relative_07"]
    for k in expanded["quotas"]:
        p = expanded["plans"][f"relative_{k:02}"]
        assert len(p["k8_layers"]) == k and len(p["v4_layers"]) == 28 - 2 * k
        assert p["sum_relative_error"] == sum(
            r[allocator.OPTIONS.index(tuple(bits))]
            for r, bits in zip(rows, p["bits"], strict=True)
        )
        assert get_profile(p["profile"], layer_plan=p).config_hash() == p["config_hash"]
        assert p["cache_bytes_per_token"]["total"] * 1024 == 30_277_632
    for name, old_name in (("early", "early"), ("absolute_07", "ranked_07")):
        assert (
            expanded["plans"][name]["config_hash"]
            == old["plans"][old_name]["config_hash"]
        )


def test_ties_and_invalid_costs(allocator: ModuleType) -> None:
    old = previous()
    assert allocator.allocation_study([[0] * 4] * 28, old, expand=True)["best_k"] == 0
    for bad in [
        [],
        [[0, 1, 2]] * 28,
        [[float("nan"), 0, 0, 0]] * 28,
        [[0, 1, -1, 1]] * 28,
    ]:
        with pytest.raises(ValueError, match=r"28 x 4|finite and nonnegative"):
            allocator.allocation_study(bad, old)
