# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in V protection: budgets, cache identity and current/past layer routing."""

from dataclasses import replace

import numpy as np
import onnx
import pytest

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import (
    TurboQuantConfig,
    get_profile,
    v8_layer_profile,
)
from qai_hub_models.models.templates.llm.turboquant.current_attention import (
    quantize_current_attention,
)
from qai_hub_models.models.templates.llm.turboquant.export import build_encode_model
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)
from qai_hub_models.models.templates.llm.turboquant.native_decoder import (
    use_native_decoder,
)
from qai_hub_models.models.templates.llm.turboquant.reference import load_codebook
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)
from qai_hub_models.test.test_models.test_turboquant_tiled_attention import (
    CONTEXT,
    attention_part,
)


@pytest.mark.parametrize(
    ("profile", "v8", "mib"),
    [
        ("k4_v4_scaled", (), 28.875),
        ("k4_v2_scaled", (), 21.875),
        ("k4_v8_boundary", (0, 1, 26, 27), 24.875),
        ("k4_v8_middle", (12, 13, 14, 15), 24.875),
    ],
)
def test_static_v_budget(profile: str, v8: tuple[int, ...], mib: float) -> None:
    c = get_profile(profile)
    assert all(c.codec_for("key", i).bits == 4 for i in range(28))
    assert tuple(i for i in range(28) if c.codec_for("value", i).bits == 8) == v8
    cache = TurboQuantKVCache(c, 28, 8, 128, 1024)
    assert cache.memory_report().allocated_bytes == mib * 2**20
    for i, (key, value) in enumerate(cache.layers):
        assert key.packed.shape[-1] == 64
        assert value.packed.shape[-1] == 16 * c.codec_for("value", i).bits
    if c.layer_bits:
        with pytest.raises(ValueError, match="layer count"):
            TurboQuantKVCache(c, 27, 8, 128, 1024)


def test_incompatible_cache_rejected() -> None:
    a = TurboQuantKVCache(get_profile("k4_v8_boundary"), 28, 8, 128, 8)
    b = TurboQuantKVCache(get_profile("k4_v8_middle"), 28, 8, 128, 8)
    with pytest.raises(ValueError, match="different TurboQuant config"):
        b.load_state_dict(a.state_dict())


@pytest.mark.parametrize("name", ["k4_v8_boundary", "k4_v8_middle"])
def test_layer_cache_append_current_past_reset(name: str) -> None:
    config = get_profile(name)
    whole = TurboQuantKVCache(config, 28, 1, 128, 9)
    stepped = TurboQuantKVCache(config, 28, 1, 128, 9)
    rng = np.random.default_rng(150)
    kv = [
        rng.normal(size=shape).astype(np.float32)
        for _ in range(28)
        for shape in ((1, 1, 128, 9), (1, 1, 9, 128))
    ]
    whole.append(kv)
    for i in range(9):
        stepped.append(
            [
                v[..., i : i + 1] if j % 2 == 0 else v[:, :, i : i + 1]
                for j, v in enumerate(kv)
            ]
        )
    for left, right in zip(whole.layers, stepped.layers, strict=True):
        for a, b in zip(left, right, strict=True):
            for key, value in a.arrays().items():
                np.testing.assert_array_equal(value, b.arrays()[key])
    snapshot = whole.state_dict()
    whole.reset()
    assert whole.get_seq_length() == 0
    for pair in whole.layers:
        for store in pair:
            assert all(not data.any() for data in store.arrays().values())
    whole.load_state_dict(snapshot)
    assert whole.get_seq_length() == 9


@pytest.mark.parametrize("edge", [2, 4, 6])
def test_future_boundary_expansion_requires_explicit_frozen_plan(edge: int) -> None:
    c = v8_layer_profile(tuple(range(edge)) + tuple(range(28 - edge, 28)))
    plan = {
        **c.to_dict()["layer_plan"],
        "profile": c.profile,
        "config_hash": c.config_hash(),
    }
    assert get_profile("k4_v8_custom", layer_plan=plan) == c
    assert sum(v == 8 for k, v in c.layer_bits) == 2 * edge
    with pytest.raises(ValueError, match="hash"):
        get_profile("k4_v8_custom", layer_plan={**plan, "config_hash": "wrong"})
    with pytest.raises(ValueError, match="requires"):
        get_profile("k4_v8_custom")


@pytest.mark.parametrize("layers", [(0, 0), (-1,), (28,), (True,), (1.5,)])
def test_bad_layer_ids(layers: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="V8 layer IDs"):
        v8_layer_profile(layers)


@pytest.mark.parametrize(
    "bits", [None, [4], [[8, 2]], [[4, 3]], [[4, 8.0]], [[4, True]]]
)
def test_custom_plan_rejects_bad_or_non_v_only_bits(bits: object) -> None:
    with pytest.raises(ValueError, match="requires K4"):
        get_profile(
            "k4_v8_custom",
            layer_plan={
                "profile": "k4_v8_custom",
                "policy": "v_static",
                "bits": bits,
            },
        )


def test_v8_direct_storage_non_affine_codebook() -> None:
    c = get_profile("k4_v8_scaled")
    graph = build_encode_model(c, c.value, 8, 128).graph
    assert not any(n.op_type in ("Slice", "Concat") for n in graph.node)
    assert not any("pack_" in n.name for n in graph.node)
    values = load_codebook(8, 128).astype(np.float16)
    assert len(values) == 256 and np.all(np.diff(values) > 0)
    assert not np.allclose(np.diff(values), np.diff(values)[0])
    assert get_profile("k8_v3").key.kind.value == "baseline"


@pytest.mark.parametrize("bits", [2, 8])
def test_v_layer_current_past_native_routing(bits: int) -> None:
    c = get_profile("k4_v8_boundary")
    c = replace(c, layer_bits=((4, bits), *c.layer_bits[1:]))
    model, enc = attention_part(3)

    def build(config: TurboQuantConfig) -> onnx.ModelProto:
        result = apply_kv_profile(model, enc, config, 3, CONTEXT)
        result = tile_kv_attention(result, config, 7, rotated=True)
        result = use_native_decoder(result, config)
        return quantize_current_attention(result, config).model

    whole, local = build(c), build(c.for_layer(0))
    assert whole.SerializeToString() == local.SerializeToString()
    native = [n.op_type for n in whole.graph.node if n.domain == "turboquant"]
    assert set(native) == {"Decode4", f"Decode{bits}"}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "k4_v4_scaled",
            "d7ac74594b85ffd81086018773add722de8e6b9f1eee7339b929f8510374be6e",
        ),
        (
            "k5_v3_scaled",
            "8d318195168176a4540c0b134f03d6288ed83f74a6a9175f44dbcf34e83ecd46",
        ),
        (
            "k6_v2_scaled",
            "87e49f856d74caa0be36be4c05bd8e22b67c525eb4116b89385be72a67a98411",
        ),
    ],
)
def test_legacy_profile_hashes(name: str, expected: str) -> None:
    assert get_profile(name).config_hash() == expected
