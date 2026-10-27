# SPDX-License-Identifier: BSD-3-Clause
"""PPL aggregation and scoring-contract tests; no device execution."""

import importlib
import math
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def ppl(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("benchmark_k8_ppl")


def test_aggregate_nll_not_arithmetic_window_ppl(ppl: ModuleType) -> None:
    rows = [
        {"group": g, "window": w, "nll_sum": (i + 1) * 1023, "scored_tokens": 1023}
        for g in ppl.GROUPS
        for i, w in enumerate(ppl.WINDOWS)
    ]
    for row in ppl.aggregate(rows):
        assert row["scored_tokens"] == 4092
        assert row["mean_nll"] == 2.5
        assert row["ppl"] == math.exp(2.5)
    with pytest.raises(ValueError, match="four unique windows"):
        ppl.aggregate([*rows[:-1], rows[0]])
    with pytest.raises(ValueError, match="four unique windows"):
        ppl.aggregate(rows[:-1])


def test_score_contract(ppl: ModuleType) -> None:
    data = {
        "mode": "score",
        "context_length": 1024,
        "prompt_tokens": 1024,
        "context_buckets": [1024],
        "sessions": [{"session": 0, "cached_tokens_at_end": 1024}],
        "scored_tokens": 1023,
        "config_hash": ppl.config("odd_lm").config_hash(),
        "quantize_current_kv": True,
        "assets": {"tokens_file": "wikitext_w0.bin", "tokens_sha256": "test_hash"},
        "kv_store_bytes": 30277632,
        "nll_sum": 3069,
        "ppl": math.exp(3),
    }
    assert (
        ppl.score_row(data, "odd_lm", "wikitext_w0.bin", "test_hash")["mean_nll"] == 3
    )
    for patch in (
        {"scored_tokens": 1024},
        {"config_hash": "wrong"},
        {"quantize_current_kv": False},
        {"nll_sum": float("nan")},
        {"ppl": 23},
        {"sessions": data["sessions"] * 2},
        {"diagnostic_only": True},
    ):
        with pytest.raises(ValueError, match=r"Scoring contract|Runner PPL"):
            ppl.score_row(data | patch, "odd_lm", "wikitext_w0.bin", "test_hash")
