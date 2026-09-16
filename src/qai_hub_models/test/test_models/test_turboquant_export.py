# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""TurboQuant ONNX encode/decode graphs vs the float64 oracle (onnxruntime CPU)."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from qai_hub_models.models.templates.llm.turboquant.config import (
    BASELINE,
    get_profile,
)
from qai_hub_models.models.templates.llm.turboquant.export import (
    build_decode_model,
    build_encode_model,
    byte_centroid_lut,
)
from qai_hub_models.models.templates.llm.turboquant.numerics import (
    FLOAT32_GRAPH,
    HTP_FP16,
    compare_decode,
    compare_encode,
)
from qai_hub_models.models.templates.llm.turboquant.packing import pack_indices
from qai_hub_models.models.templates.llm.turboquant.reference import (
    PolarQuantReference,
    load_codebook,
)

HEADS, D = 8, 128
CONFIG = get_profile("k4_v4")
# Ops the QAIRT 2.48 HTP backend supports for the dtypes these graphs use.
HTP_SAFE_OPS = {
    "Abs", "Add", "Cast", "Div", "Gather", "Greater", "Identity", "MatMul", "Mul",
    "ReduceMax", "ReduceSum", "Reshape", "Slice", "Sqrt", "Where",
}  # fmt: skip


def run(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return session.run(None, feeds)


def sample_inputs(tokens: int) -> np.ndarray:
    rng = np.random.default_rng(tokens)
    x = rng.standard_normal((1, HEADS, tokens, D)) * rng.uniform(
        0.05, 400, (1, HEADS, tokens, 1)
    )
    x[0, 0, 0] = 0.0
    return x.astype(np.float32)


@pytest.mark.parametrize("tokens", [1, 128])
@pytest.mark.parametrize("which", ["key", "value"])
def test_encode_graph_matches_oracle(tokens: int, which: str) -> None:
    spec = getattr(CONFIG, which)
    x = sample_inputs(tokens)
    packed, norm = run(build_encode_model(CONFIG, spec, HEADS, tokens), {"x": x})
    assert packed.dtype == np.uint8 and packed.shape == (1, HEADS, tokens, D // 2)
    assert norm.shape == (1, HEADS, tokens, 1)
    report = compare_encode(
        PolarQuantReference(spec, D), x, packed, norm, FLOAT32_GRAPH
    )
    assert report["passed"], report
    assert report["zero_vectors"] == 1


@pytest.mark.parametrize("tokens", [1, 128])
def test_decode_graph_matches_oracle(tokens: int) -> None:
    spec = CONFIG.value
    codec = PolarQuantReference(spec, D)
    idx, norms = codec.encode(sample_inputs(tokens))
    packed = pack_indices(idx, 4)
    norms32 = norms.astype(np.float32)
    (x_hat,) = run(
        build_decode_model(CONFIG, spec, HEADS, tokens),
        {"packed": packed, "norm": norms32},
    )
    report = compare_decode(codec, packed, norms32, x_hat, FLOAT32_GRAPH)
    assert report["passed"], report


def test_encode_graph_handles_full_float16_input_range() -> None:
    """Rows up to the float16 limit take the power-of-two prescale path exactly."""
    x = np.zeros((1, HEADS, 1, D), dtype=np.float32)
    for h, mag in enumerate([1e-2, 255.0, 256.0, 257.0, 2e4, 3.6e4, 5e4, 6.5e4]):
        x[0, h, 0, (7 * h) % D] = mag
    x[0, 7, 0, :] = 5.6e3 * np.where(np.arange(D) % 2, 1, -1)
    packed, norm = run(build_encode_model(CONFIG, CONFIG.value, HEADS, 1), {"x": x})
    report = compare_encode(
        PolarQuantReference(CONFIG.value, D), x, packed, norm, FLOAT32_GRAPH
    )
    assert report["passed"], report


def test_encode_then_decode_graph_round_trip() -> None:
    spec = CONFIG.key
    x = sample_inputs(128)
    packed, norm = run(build_encode_model(CONFIG, spec, HEADS, 128), {"x": x})
    (x_hat,) = run(
        build_decode_model(CONFIG, spec, HEADS, 128), {"packed": packed, "norm": norm}
    )
    codec = PolarQuantReference(spec, D)
    expected = codec.decode(*codec.encode(x))
    scale = np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-30)
    assert np.max(np.abs(x_hat - expected) / scale) < 1e-4


def test_graphs_use_only_htp_safe_ops_and_rank_le_4() -> None:
    for build in (build_encode_model, build_decode_model):
        model = onnx.shape_inference.infer_shapes(build(CONFIG, CONFIG.key, HEADS, 128))
        assert {n.op_type for n in model.graph.node} <= HTP_SAFE_OPS
        for info in [*model.graph.value_info, *model.graph.output]:
            dims = info.type.tensor_type.shape.dim
            assert len(dims) <= 4, info.name


def test_compare_encode_rejects_flip_away_from_nearby_boundary() -> None:
    """Only a flip across the boundary the value is near may be excused."""
    codec = PolarQuantReference(CONFIG.key, D)
    x = sample_inputs(128)
    y, norms = codec.rotate_normalized(x)
    idx = np.searchsorted(codec.boundaries, y, side="left")
    lower_gap = np.abs(y - codec.boundaries[np.clip(idx - 1, 0, 14)])
    near_lower = np.argwhere((idx > 0) & (idx < 15) & (lower_gap < 1e-4))[:50]
    assert len(near_lower) > 0

    for shift, should_pass in ((-1, True), (1, False)):
        dev = idx.copy()
        for pos in near_lower:
            dev[tuple(pos)] += shift
        report = compare_encode(
            codec, x, pack_indices(dev.astype(np.uint8), 4), norms, HTP_FP16
        )
        assert report["passed"] is should_pass, (shift, report)


def test_compare_decode_detects_single_wrong_centroid() -> None:
    codec = PolarQuantReference(CONFIG.value, D)
    idx, norms = codec.encode(sample_inputs(128))
    wrong = idx.copy()
    wrong[..., 17] = np.where(wrong[..., 17] == 7, 8, 7)
    x_hat = codec.decode(wrong, norms)
    report = compare_decode(codec, pack_indices(idx, 4), norms, x_hat, HTP_FP16)
    assert not report["passed"]
    assert report["decode_max_rel_error"] > 3 * HTP_FP16.decode_rel


def test_byte_lut_is_msb_first() -> None:
    lut = byte_centroid_lut(4, D)
    c = load_codebook(4, D)
    assert lut[0x3A, 0] == c[3] and lut[0x3A, 1] == c[10]


def test_unsupported_bit_widths_raise() -> None:
    with pytest.raises(NotImplementedError, match="4-bit"):
        build_encode_model(get_profile("k4_v3"), get_profile("k4_v3").value, HEADS, 1)
    with pytest.raises(NotImplementedError, match="4-bit"):
        build_decode_model(CONFIG, BASELINE, HEADS, 1)
