# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Packed TurboQuant KV cache state tests (append, reset, limits, serialization)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.reference import (
    PolarQuantReference,
)

LAYERS, KV_HEADS, D, CTX = 3, 2, 128, 40


def make_step(rng: np.random.Generator, new: int) -> list[np.ndarray]:
    """Hub layout: key (kv, 1, head_dim, new), value (kv, 1, new, head_dim)."""
    flat = []
    for _ in range(LAYERS):
        flat.append(rng.standard_normal((KV_HEADS, 1, D, new)).astype(np.float32) * 20)
        flat.append(rng.standard_normal((KV_HEADS, 1, new, D)).astype(np.float32) * 5)
    return flat


def make_cache(profile: str = "k4_v4", ctx: int = CTX) -> TurboQuantKVCache:
    return TurboQuantKVCache(get_profile(profile), LAYERS, KV_HEADS, D, ctx)


def concat_steps(
    steps: list[list[np.ndarray]], layer: int
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.concatenate([s[2 * layer] for s in steps], axis=3)
    values = np.concatenate([s[2 * layer + 1] for s in steps], axis=2)
    return keys, values


def test_chunked_prefill_equals_token_by_token() -> None:
    rng = np.random.default_rng(0)
    prompt = make_step(rng, 12)
    chunked, stepped = make_cache(), make_cache()
    chunked.append(
        [t[..., :7] if i % 2 == 0 else t[:, :, :7] for i, t in enumerate(prompt)]
    )
    chunked.append(
        [t[..., 7:] if i % 2 == 0 else t[:, :, 7:] for i, t in enumerate(prompt)]
    )
    for pos in range(12):
        stepped.append(
            [
                t[..., pos : pos + 1] if i % 2 == 0 else t[:, :, pos : pos + 1]
                for i, t in enumerate(prompt)
            ]
        )
    assert chunked.get_seq_length() == stepped.get_seq_length() == 12
    for (ck, cv), (sk, sv) in zip(chunked.layers, stepped.layers, strict=True):
        for a, b in ((ck, sk), (cv, sv)):
            for name, arr in a.arrays().items():
                np.testing.assert_array_equal(arr, b.arrays()[name], err_msg=name)


def test_append_encodes_only_new_tokens_and_decodes_in_hub_layout() -> None:
    rng = np.random.default_rng(1)
    cache = make_cache()
    steps = [make_step(rng, 5), make_step(rng, 1), make_step(rng, 1)]
    for step in steps:
        cache.append(step)
    snapshot = cache.layers[0][0].arrays()["packed"][:, :, :5].copy()
    cache.append(make_step(rng, 1))
    np.testing.assert_array_equal(
        cache.layers[0][0].arrays()["packed"][:, :, :5], snapshot
    )

    key_codec = PolarQuantReference(get_profile("k4_v4").key, D)
    keys, values = concat_steps(steps, layer=2)
    dec_k, dec_v = cache.layer_float(2)
    assert dec_k.shape == (KV_HEADS, 1, D, 8)
    assert dec_v.shape == (KV_HEADS, 1, 8, D)
    expected_k = key_codec.decode(*key_codec.encode(np.swapaxes(keys, 2, 3)))
    np.testing.assert_allclose(
        np.swapaxes(dec_k[..., :7], 2, 3), expected_k, rtol=1e-3, atol=1e-2
    )
    cos = np.sum(dec_v[:, :, :7] * values, -1) / (
        np.linalg.norm(dec_v[:, :, :7], axis=-1) * np.linalg.norm(values, axis=-1)
    )
    assert cos.min() > 0.97


def test_k8_v4_keeps_keys_bit_exact() -> None:
    rng = np.random.default_rng(2)
    cache = make_cache("k8_v4")
    step = make_step(rng, 4)
    cache.append(step)
    dec_k, _ = cache.layer_float(1)
    np.testing.assert_array_equal(dec_k, step[2])
    assert cache.layers[0][0].packed is None
    assert cache.layers[0][1].packed is not None


def test_accepts_torch_tensors() -> None:
    rng = np.random.default_rng(3)
    step = make_step(rng, 2)
    a, b = make_cache(), make_cache()
    a.append(step)
    b.append([torch.from_numpy(t) for t in step])
    np.testing.assert_array_equal(a.layers[1][1].packed, b.layers[1][1].packed)


def test_overflow_is_rejected_without_partial_write() -> None:
    rng = np.random.default_rng(4)
    cache = make_cache(ctx=6)
    cache.append(make_step(rng, 5))
    before = cache.layers[0][1].arrays()["packed"].copy()
    with pytest.raises(ValueError, match="does not slide"):
        cache.append(make_step(rng, 2))
    assert cache.get_seq_length() == 5
    np.testing.assert_array_equal(cache.layers[0][1].arrays()["packed"], before)
    cache.append(make_step(rng, 1))
    assert cache.get_seq_length() == 6


def test_reset_isolates_sessions() -> None:
    rng = np.random.default_rng(5)
    first_session = make_step(rng, 3)
    cache = make_cache()
    cache.append(make_step(rng, 9))
    cache.reset()
    assert cache.get_seq_length() == 0
    assert all(
        not a.any() for pair in cache.layers for s in pair for a in s.arrays().values()
    )
    cache.append(first_session)
    fresh = make_cache()
    fresh.append(first_session)
    np.testing.assert_array_equal(cache.layer_float(0)[1], fresh.layer_float(0)[1])


def test_rejects_malformed_steps() -> None:
    rng = np.random.default_rng(6)
    cache = make_cache()
    step = make_step(rng, 2)
    with pytest.raises(ValueError, match="Expected 6"):
        cache.append(step[:4])
    swapped = list(step)
    swapped[1] = np.swapaxes(step[1], 2, 3)
    with pytest.raises(ValueError, match="Value shape"):
        cache.append(swapped)
    mixed = list(step)
    mixed[2], mixed[3] = make_step(rng, 3)[2:4]
    with pytest.raises(ValueError, match="disagree"):
        cache.append(mixed)
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        TurboQuantKVCache(get_profile("k4_v4"), LAYERS, KV_HEADS, D, CTX, batch_size=2)
    with pytest.raises(ValueError, match="head_dim"):
        TurboQuantKVCache(get_profile("k4_v4"), LAYERS, KV_HEADS, 64, CTX)


def test_state_dict_round_trip_and_version_checks() -> None:
    rng = np.random.default_rng(7)
    cache = make_cache()
    cache.append(make_step(rng, 6))
    state = cache.state_dict()
    restored = make_cache()
    restored.load_state_dict(state)
    assert restored.get_seq_length() == 6
    np.testing.assert_array_equal(restored.layer_float(2)[0], cache.layer_float(2)[0])

    with pytest.raises(ValueError, match="different TurboQuant config"):
        make_cache("k4_v3").load_state_dict(state)
    with pytest.raises(ValueError, match="format"):
        make_cache().load_state_dict({**state, "format_version": 99})
    with pytest.raises(ValueError, match="shape"):
        make_cache(ctx=CTX + 1).load_state_dict(state)


def test_state_dict_is_a_snapshot() -> None:
    rng = np.random.default_rng(9)
    cache = make_cache()
    cache.append(make_step(rng, 4))
    expected = cache.layer_float(1)[1]
    state = cache.state_dict()
    cache.reset()
    cache.append(make_step(rng, 3))
    restored = make_cache()
    restored.load_state_dict(state)
    assert restored.get_seq_length() == 4
    np.testing.assert_array_equal(restored.layer_float(1)[1], expected)


def test_bad_state_does_not_partially_overwrite() -> None:
    rng = np.random.default_rng(10)
    source, target = make_cache(), make_cache()
    source.append(make_step(rng, 5))
    target.append(make_step(rng, 2))
    before = target.state_dict()
    bad = source.state_dict()
    bad["layers"][-1]["value"]["norms"] = bad["layers"][-1]["value"]["norms"].astype(
        np.float32
    )
    with pytest.raises(ValueError, match="shape/dtype"):
        target.load_state_dict(bad)
    short = {**source.state_dict(), "layers": source.state_dict()["layers"][:1]}
    with pytest.raises(ValueError, match="number of layers"):
        target.load_state_dict(short)
    assert target.get_seq_length() == 2
    for (k, v), layer_state in zip(target.layers, before["layers"], strict=True):
        for store, kind in ((k, "key"), (v, "value")):
            for name, arr in store.arrays().items():
                np.testing.assert_array_equal(arr, layer_state[kind][name])


def test_memory_report_matches_spec_arithmetic() -> None:
    """Qwen3-1.7B, 4096 tokens: 4-bit K+V payload 112 MiB, FP16 block norms 3.5 MiB."""
    mib = 1024 * 1024
    cache = TurboQuantKVCache(get_profile("k4_v4"), 28, 8, 128, 4096)
    report = cache.memory_report(tokens=4096)
    assert report.packed_payload_bytes == report.formula_payload_bytes == 112 * mib
    assert report.norm_bytes == int(3.5 * mib)
    assert report.allocated_bytes == 112 * mib + int(3.5 * mib)
    assert report.baseline_float_bytes == 0

    k8 = TurboQuantKVCache(get_profile("k8_v4"), 28, 8, 128, 4096).memory_report(4096)
    assert k8.packed_payload_bytes == 56 * mib
    # Host generator stores baseline K as float32; int8 lives only inside the graph.
    assert k8.baseline_float_bytes == 28 * 8 * 4096 * 128 * 4

    partial = make_cache()
    partial.append(make_step(np.random.default_rng(8), 10))
    assert partial.memory_report().tokens == 10
    assert (
        partial.memory_report().allocated_bytes
        > partial.memory_report().packed_payload_bytes
    )
