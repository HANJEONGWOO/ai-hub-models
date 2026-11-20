# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Exact per-kind quotas and frozen cross-model plan identities."""

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest

from qai_hub_models.models.templates.llm.turboquant.config import get_profile


@pytest.fixture
def plans(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("cross_model_plans")


@pytest.mark.parametrize("n", [28, 36])
def test_exact_masks(plans: ModuleType, n: int) -> None:
    for k in (4, 5, 6):
        masks = set()
        for placement in plans.PLACEMENTS:
            bits = plans.layer_bits(n, k, placement)
            assert len(bits) == n
            assert sum(kb for kb, _ in bits) == k * n
            assert sum(vb for _, vb in bits) == (8 - k) * n
            assert all(kb in (4, 8) and vb in (2, 4) for kb, vb in bits)
            masks.add(tuple(map(tuple, bits)))
        assert len(masks) == (1 if k == 4 else 5)
    historical_odd = plans.layer_bits(n, 6, "interleaved")
    assert [i for i, (kb, _) in enumerate(historical_odd) if kb == 8] == list(
        range(1, n, 2)
    )
    historical_k5 = plans.layer_bits(n, 5, "interleaved")
    assert [i for i, (kb, _) in enumerate(historical_k5) if kb == 8] == list(
        range(3, n, 4)
    )
    assert [i for i, (_, vb) in enumerate(historical_k5) if vb == 4] == list(
        range(1, n, 2)
    )


def test_placement_rounding(plans: ModuleType) -> None:
    expected = {
        "early": list(range(7)),
        "late": list(range(21, 28)),
        "middle": list(range(10, 17)),
        "boundary": [0, 1, 2, 3, 25, 26, 27],
    }
    for placement, k8_layers in expected.items():
        bits = plans.layer_bits(28, 5, placement)
        assert [i for i, (kb, _) in enumerate(bits) if kb == 8] == k8_layers
    for n, k, placement, error in (
        (27, 5, "late", "layer count"),
        (28, 5.0, "late", "precision"),
        (36, 3, "early", "precision"),
        (28, 6, "oddish", "Unknown placement"),
    ):
        with pytest.raises(ValueError, match=error):
            plans.layer_bits(n, k, placement)


def test_freeze_manifest_and_inference(plans: ModuleType, tmp_path: Path) -> None:
    manifest = plans.freeze_plans(tmp_path)
    frozen = manifest.read_bytes()
    files = json.loads(frozen)["plan_files_sha256"]
    assert len(files) == 33
    assert plans.freeze_plans(tmp_path).read_bytes() == frozen
    for name, sha in files.items():
        data = (tmp_path / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha
        plan = json.loads(data)
        num_layers = 28 if plan["model_id"] == "qwen3_1_7b" else 36
        assert plan["num_layers"] == num_layers
        assert plan["cache_bytes_per_token"]["total"] == num_layers * 8 * (128 + 4)
        config = get_profile(plan["profile"], layer_plan=plan)
        config.validate_for_model(num_layers, 8, 128)
        for layer, (kb, vb) in enumerate(plan["bits"]):
            kspec = config.codec_for("key", layer)
            vspec = config.codec_for("value", layer)
            assert (kspec.bits, vspec.bits) == (kb, vb)
            assert kspec.quantizer == (
                "structured_k8_two_region_v1" if kb == 8 else "lloyd_max"
            )
            assert vspec.quantizer == "lloyd_max"
    changed = tmp_path / next(iter(files))
    changed.write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        plans.freeze_plans(tmp_path)
    assert changed.read_text() == "{}"
