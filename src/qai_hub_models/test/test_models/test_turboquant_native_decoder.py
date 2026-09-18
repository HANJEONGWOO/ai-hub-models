# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Native graph replacement contract and independent ONNX attention oracle."""

from __future__ import annotations

import math

import numpy as np
import onnx
import pytest

from qai_hub_models.models.templates.llm.turboquant.config import Rotation, get_profile
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)
from qai_hub_models.models.templates.llm.turboquant.native_decoder import (
    use_native_decoder,
    with_reference_decoder,
)
from qai_hub_models.models.templates.llm.turboquant.packing import unpack_indices
from qai_hub_models.models.templates.llm.turboquant.reference import load_codebook
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)
from qai_hub_models.test.test_models.test_turboquant_rotated_attention import run
from qai_hub_models.test.test_models.test_turboquant_tiled_attention import (
    CONTEXT,
    D,
    attention_part,
    feeds,
)


@pytest.mark.parametrize("seq", [1, 3])
@pytest.mark.parametrize("tile", [1, 7, 256])
@pytest.mark.parametrize("valid", [0, 13, 32])
@pytest.mark.parametrize("rotation", list(Rotation))
def test_native_replacement(
    seq: int, tile: int, valid: int, rotation: Rotation
) -> None:
    config = get_profile("k4_v4_scaled", rotation)
    model, encodings = attention_part(seq)
    tiled = tile_kv_attention(
        apply_kv_profile(model, encodings, config, seq, CONTEXT),
        config,
        tile,
        rotated=True,
    )
    before = tiled.model.SerializeToString()
    native = use_native_decoder(tiled, config)
    assert before == tiled.model.SerializeToString()
    assert native.encodings["param_encodings"] == tiled.encodings["param_encodings"]
    for side in ("input", "output"):
        assert [v.SerializeToString() for v in getattr(tiled.model.graph, side)] == [
            v.SerializeToString() for v in getattr(native.model.graph, side)
        ]
    ops = list(native.model.graph.node)
    assert len([n for n in ops if n.op_type == "Decode4"]) == 2 * math.ceil(
        (CONTEXT - seq) / tile
    )
    assert not any("_tile" in n.name and "_dec_" in n.name for n in ops)
    # Matrix products, rotations, softmax and calibrated grids are preserved.
    keep = {"MatMul", "Softmax"}
    assert [n.SerializeToString() for n in ops if n.op_type in keep] == [
        n.SerializeToString() for n in tiled.model.graph.node if n.op_type in keep
    ]
    oracle = with_reference_decoder(native.model)
    onnx.checker.check_model(oracle, full_check=True)
    data = feeds(seq, min(valid, CONTEXT - seq))
    for kind in ("key", "value"):
        indices = unpack_indices(data[f"tq_{kind}_0_packed_in"], 4, D)
        length = np.linalg.norm(load_codebook(4, D)[indices], axis=-1, keepdims=True)
        data[f"tq_{kind}_0_scale_in"] = (
            (data.pop(f"tq_{kind}_0_norm_in") / length)
            .astype(np.float16)
            .astype(np.float32)
        )
    expected, actual = run(tiled.model, data), run(oracle, data)
    for name, value in actual.items():
        if name.startswith(("tq_", "past_")):
            np.testing.assert_array_equal(value, expected[name])
        else:
            # FP16 centroid and product rounding is intentional, unlike graph
            # float32 arithmetic. Device verification tests exact FP16 results.
            np.testing.assert_allclose(value, expected[name], rtol=0.005, atol=0.0005)


def test_native_rejects_unrotated_and_legacy() -> None:
    model, encodings = attention_part(1)
    for profile in ("k4_v4", "k4_v4_scaled"):
        config = get_profile(profile)
        graph = apply_kv_profile(model, encodings, config, 1, CONTEXT)
        with pytest.raises(ValueError, match="rotated tiled attention"):
            use_native_decoder(graph, config)
        with pytest.raises(ValueError, match="rotated tiled attention"):
            use_native_decoder(tile_kv_attention(graph, config, 7), config)
