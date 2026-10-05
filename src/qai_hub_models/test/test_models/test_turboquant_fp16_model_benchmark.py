# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Cross-model FP16 comparison identity, cache size and layer coverage checks."""

import copy
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("benchmark_fp16_attention_once")


def inputs(
    benchmark: ModuleType, model_id: str = "qwen3_8b", parts: int = 5
) -> tuple[dict[str, Any], dict[str, Any]]:
    layers, hidden, heads, kv_heads, dim = benchmark.QWEN_SHAPES[model_id]
    model = {
        "model_id": model_id,
        "architecture": dict(
            num_hidden_layers=layers,
            hidden_size=hidden,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=dim,
        ),
        "config_tokenizer_sha256": {"config.json": "config"},
    }
    assets = {"model": model, "rope_half": dim // 2}
    metadata = {
        group: {
            "model": copy.deepcopy(model),
            "split_manifest_sha256": "shared-checkpoint",
            "profile": benchmark.PROFILES[group],
            "context_length": 1024,
            "context_buckets": [1024],
            "num_parts": parts,
            "parts": {f"part{i}_of_{parts}": {} for i in range(1, parts + 1)},
        }
        for group in ("fp16", "turboquant")
    }
    return assets, metadata


@pytest.mark.parametrize(
    ("model_id", "parts", "layers", "fp16", "tq"),
    [
        ("qwen3_0_6b", 2, 28, 112, 28.875),
        ("qwen3_1_7b", 4, 28, 112, 28.875),
        ("qwen3_4b", 4, 36, 144, 37.125),
        ("qwen3_8b", 5, 36, 144, 37.125),
    ],
)
def test_model_sizes(
    benchmark: ModuleType,
    model_id: str,
    parts: int,
    layers: int,
    fp16: float,
    tq: float,
) -> None:
    assets, metadata = inputs(benchmark, model_id, parts)
    assert benchmark.validate_inputs(model_id, assets, metadata) == (parts, layers)
    assert benchmark.expected_kv_bytes(model_id, "fp16") / 2**20 == fp16
    assert benchmark.expected_kv_bytes(model_id, "turboquant") / 2**20 == tq


@pytest.mark.parametrize(
    "change", ["identity", "weights", "parts", "shape", "rope", "missing"]
)
def test_reject_mismatch(benchmark: ModuleType, change: str) -> None:
    assets, metadata = inputs(benchmark)
    tq = metadata["turboquant"]
    if change == "identity":
        tq["model"]["config_tokenizer_sha256"]["config.json"] = "other"
    elif change == "weights":
        tq["split_manifest_sha256"] = "other"
    elif change == "parts":
        tq["parts"].pop("part5_of_5")
    elif change == "shape":
        tq["model"]["architecture"]["hidden_size"] = 2560
    elif change == "rope":
        assets["rope_half"] = 32
    else:
        del tq["model"]
    with pytest.raises(
        ValueError, match=r"identity|checkpoint|bundle|architecture|RoPE"
    ):
        benchmark.validate_inputs("qwen3_8b", assets, metadata)


def test_legacy_1_7b(benchmark: ModuleType) -> None:
    assets, metadata = inputs(benchmark, "qwen3_1_7b", 4)
    del assets["model"]
    for data in metadata.values():
        del data["model"]
        del data["split_manifest_sha256"]
    assert benchmark.validate_inputs("qwen3_1_7b", assets, metadata) == (4, 28)


def test_audit_all_36_layers(benchmark: ModuleType, tmp_path: Path) -> None:
    audit = {
        "passed": True,
        "bundle": str(tmp_path),
        "graphs": {
            f"{kind}_ar{ar}_cl1024_{part}_of_5": {
                "layers": list(range((part - 2) * 9, (part - 1) * 9)),
                "violations": [],
            }
            for kind, ar in (("token", 1), ("prompt", 128))
            for part in range(2, 6)
        },
    }
    benchmark.validate_audit(audit, tmp_path, 36)
    audit["graphs"]["token_ar1_cl1024_5_of_5"]["layers"][-1] = 34
    with pytest.raises(ValueError, match="missing/duplicating"):
        benchmark.validate_audit(audit, tmp_path, 36)
