# SPDX-License-Identifier: BSD-3-Clause
"""Layer ABI, exact budgets, direct K8 storage and immutable plan identity."""

import importlib
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import (
    TurboQuantConfig,
    get_profile,
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


@pytest.mark.parametrize(("budget", "k8", "v4"), [(5, 7, 14), (6, 14, 0)])
def test_layer_budgets_cache_and_hash(budget: int, k8: int, v4: int) -> None:
    config = get_profile(f"mixed_k{budget}_v{8 - budget}_uniform")
    assert sum(k == 8 for k, v in config.layer_bits) == k8
    assert sum(v == 4 for k, v in config.layer_bits) == v4
    assert sum(k + v for k, v in config.layer_bits) == 28 * 8
    cache = TurboQuantKVCache(config, 28, 8, 128, 1024)
    assert cache.memory_report().allocated_bytes == 28.875 * 2**20
    with pytest.raises(ValueError, match="layer count"):
        TurboQuantKVCache(config, 4, 8, 128, 1024)
    changed = replace(config, layer_bits=config.layer_bits[::-1])
    assert changed.config_hash() != config.config_hash()
    other = TurboQuantKVCache(changed, 28, 8, 128, 1024)
    with pytest.raises(ValueError, match="different TurboQuant config"):
        other.load_state_dict(cache.state_dict())
    for layer, (ks, vs) in enumerate(cache.layers):
        assert ks.packed.shape[-1] == 16 * config.layer_bits[layer][0]
        assert vs.packed.shape[-1] == 16 * config.layer_bits[layer][1]


def test_mse_plan_requires_exact_quota_and_hash() -> None:
    uniform = get_profile("mixed_k5_v3_uniform")
    config = replace(
        uniform,
        profile="mixed_k5_v3_mse",
        layer_policy="attention_mse",
        layer_bits=uniform.layer_bits[::-1],
    )
    plan = dict(
        profile=config.profile,
        policy=config.layer_policy,
        bits=config.layer_bits,
        config_hash=config.config_hash(),
    )
    assert get_profile(config.profile, layer_plan=plan) == config
    with pytest.raises(ValueError, match="hash"):
        get_profile(config.profile, layer_plan={**plan, "config_hash": "wrong"})
    with pytest.raises(ValueError, match="budget"):
        get_profile(config.profile, layer_plan={**plan, "bits": [(8, 4)] * 28})
    with pytest.raises(ValueError, match="requires"):
        get_profile(config.profile)


def test_k8_no_packing_and_non_affine_centroids() -> None:
    config = get_profile("k8_v4_scaled")
    graph = build_encode_model(config, config.key, 8, 128).graph
    assert not any(n.op_type in ("Slice", "Concat") for n in graph.node)
    assert not any("pack_" in n.name for n in graph.node)
    c = load_codebook(8, 128).astype(np.float16)
    assert len(c) == 256 and np.all(np.diff(c) > 0)
    assert not np.allclose(np.diff(c), np.diff(c)[0])
    assert get_profile("k8_v3").key.kind.value == "baseline"


@pytest.mark.parametrize("pair", [(4, 2), (4, 4), (8, 2), (8, 4)])
def test_mixed_current_and_past_match_layer_resolved_graph(
    pair: tuple[int, int],
) -> None:
    base = get_profile("mixed_k5_v3_uniform")
    config = replace(base, layer_bits=(pair, *base.layer_bits[1:]))
    model, enc = attention_part(3)

    def build(c: TurboQuantConfig) -> onnx.ModelProto:
        result = apply_kv_profile(model, enc, c, 3, CONTEXT)
        result = tile_kv_attention(result, c, 7, rotated=True)
        result = use_native_decoder(result, c)
        return quantize_current_attention(result, c).model

    whole, local = build(config), build(config.for_layer(0))
    assert whole.SerializeToString() == local.SerializeToString()
    native = [n.op_type for n in whole.graph.node if n.domain == "turboquant"]
    assert set(native) == {f"Decode{b}" for b in pair}


def test_mse_quota_optimizer_matches_exhaustive_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    selector = importlib.import_module("select_mixed_layers")
    costs = np.random.default_rng(81).uniform(size=(4, 4)).tolist()
    candidates = []
    for plan in itertools.product(selector.OPTIONS, repeat=4):
        if sum(k == 8 for k, v in plan) == 2 and sum(v == 4 for k, v in plan) == 2:
            loss = sum(
                row[selector.OPTIONS.index(pair)]
                for row, pair in zip(costs, plan, strict=True)
            )
            candidates.append((loss, plan))
    assert selector.choose(costs, 2, 2) == min(candidates)[1]
