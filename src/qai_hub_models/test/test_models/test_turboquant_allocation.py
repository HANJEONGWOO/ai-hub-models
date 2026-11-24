# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Fixed quota, canonical mask identities and immutable legacy defaults."""

import copy

import pytest

from qai_hub_models.models.templates.llm.turboquant.config import (
    STRUCTURED_ALLOCATION_PROFILE,
    Rotation,
    get_profile,
    structured_allocation_plan,
)


@pytest.mark.parametrize(
    "layers",
    [
        list(range(1, 28, 2)),
        list(range(0, 28, 2)),
        list(range(14)),
        list(range(14, 28)),
        list(range(7, 21)),
        list(range(7)) + list(range(21, 28)),
    ],
)
def test_allocation_quota_hash(layers: list[int]) -> None:
    plan = structured_allocation_plan(layers)
    config = get_profile(STRUCTURED_ALLOCATION_PROFILE, layer_plan=plan)
    assert plan == structured_allocation_plan(list(reversed(layers)))
    assert sum(k + v for k, v in config.layer_bits) == 28 * 8
    assert sum(k == 8 for k, _ in config.layer_bits) == 14
    assert all(v == 2 for _, v in config.layer_bits)
    assert config.config_hash() == plan["config_hash"]
    assert config.rotation == Rotation.DENSE_QR and config.precomputed_norm
    for i in range(28):
        assert config.codec_for("key", i).quantizer == (
            "structured_k8_two_region_v1" if i in layers else "lloyd_max"
        )
        assert config.codec_for("value", i).quantizer == "lloyd_max"


@pytest.mark.parametrize(
    "layers",
    [
        None,
        [],
        list(range(13)),
        list(range(15)),
        [0] * 14,
        [*range(13), 28],
        [*range(13), -1],
        [*range(13), True],
        [*range(13), 13.0],
    ],
)
def test_invalid_allocations(layers: object) -> None:
    with pytest.raises(ValueError, match="exactly 14"):
        structured_allocation_plan(layers)


def test_allocation_hash_rejects_mixed_caches_and_tampering() -> None:
    odd = structured_allocation_plan(list(range(1, 28, 2)))
    even = structured_allocation_plan(list(range(0, 28, 2)))
    assert odd["config_hash"] != even["config_hash"]
    for key in ("config_hash", "bits", "config", "policy"):
        altered = copy.deepcopy(odd)
        altered[key] = "tampered"
        with pytest.raises(ValueError, match="mismatch"):
            get_profile(STRUCTURED_ALLOCATION_PROFILE, layer_plan=altered)
    with pytest.raises(ValueError, match="frozen layer-plan"):
        get_profile(STRUCTURED_ALLOCATION_PROFILE)
    with pytest.raises(ValueError, match="Dense QR"):
        get_profile(STRUCTURED_ALLOCATION_PROFILE, Rotation.FWHT, layer_plan=odd)
    assert (
        get_profile("k4_v4_scaled").config_hash()
        == "d7ac74594b85ffd81086018773add722de8e6b9f1eee7339b929f8510374be6e"
    )
    assert (
        get_profile("mixed_k6_v2_uniform_structured_k8").config_hash()
        == "b68d0d5534b516363c892ecf65bd98f44789969d4e07cc31de73e7bdc3d11ed9"
    )
