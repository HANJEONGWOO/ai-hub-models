# SPDX-License-Identifier: BSD-3-Clause
"""Layer-specific rotations preserve K/Q correspondence, V, shape and cache ABI."""

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import (
    TurboQuantConfig,
    get_profile,
)
from qai_hub_models.models.templates.llm.turboquant.current_attention import (
    quantize_current_attention,
)
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)
from qai_hub_models.models.templates.llm.turboquant.native_decoder import (
    use_native_decoder,
)
from qai_hub_models.models.templates.llm.turboquant.reference import DenseQRRotation
from qai_hub_models.models.templates.llm.turboquant.rotation_artifact import (
    save_layer_rotations,
    with_key_rotation,
)
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)
from qai_hub_models.test.test_models.test_turboquant_tiled_attention import (
    attention_part,
)


def config(
    tmp_path: Path, seeds: tuple[int, ...] = (43, 48)
) -> tuple[TurboQuantConfig, Path]:
    path = tmp_path / "P.json"
    save_layer_rotations(
        path,
        list(seeds),
        {s: DenseQRRotation(s, 128).matrix().astype(np.float32) for s in seeds},
        {},
    )
    return with_key_rotation(get_profile("k4_v4_scaled"), path), path


def test_layer_policy_and_matrix_hashes(tmp_path: Path) -> None:
    c, path = config(tmp_path)
    assert c.value == get_profile("k4_v4_scaled").value
    assert c.key_for_layer(0).seed == 43 and c.key_for_layer(1).seed == 48
    assert (
        c.config_hash()
        != replace(c, key_layers=tuple(reversed(c.key_layers))).config_hash()
    )
    with pytest.raises(ValueError, match="layer count"):
        c.validate_for_model(28, 8, 128)
    with pytest.raises(ValueError, match="Missing"):
        c.key_for_layer(2)
    with pytest.raises(ValueError, match="constant name"):
        replace(c, key_layers=(c.key_layers[0], replace(c.key_layers[1], seed=43)))
    with pytest.raises(ValueError, match="collides"):
        replace(c, key_layers=(replace(c.key_layers[0], seed=c.value.seed),))
    data = json.loads(path.read_text())
    data["layer_seeds"].reverse()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="policy checksum"):
        with_key_rotation(get_profile("k4_v4_scaled"), path)


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("seq", [1, 3])
def test_k_and_query_rotation_only(tmp_path: Path, layer: int, seq: int) -> None:
    c, _ = config(tmp_path)
    base = get_profile("k4_v4_scaled")
    models = []
    for setting in (base, c):
        model, enc = attention_part(seq)
        if layer:

            def rename(s: str) -> str:
                return s.replace("past_key_0", f"past_key_{layer}").replace(
                    "past_value_0", f"past_value_{layer}"
                )

            for node in model.graph.node:
                node.name = rename(node.name)
                for items in (node.input, node.output):
                    for i, name in enumerate(items):
                        items[i] = rename(name)
            for val in (
                list(model.graph.input)
                + list(model.graph.output)
                + list(model.graph.value_info)
            ):
                val.name = rename(val.name)
            for entry in enc["activation_encodings"]:
                entry["name"] = rename(entry["name"])
        result = apply_kv_profile(model, enc, setting, seq, 35)
        result = tile_kv_attention(result, setting, 7, rotated=True)
        result = use_native_decoder(result, setting)
        result = quantize_current_attention(result, setting)
        onnx.checker.check_model(result.model)
        models.append(result.model)
    graph = models[1].graph
    expected = f"tq_rotation_t_dense_qr_s{c.key_for_layer(layer).seed}_d128"
    relevant = [
        n for n in graph.node if n.op_type == "MatMul" and n.input[1] == expected
    ]
    assert len(relevant) == 5  # one K encoder, two KV heads x two Q heads
    assert any(n.output[0] == f"tq_key_{layer}_enc_rotated" for n in relevant)
    values = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    np.testing.assert_array_equal(
        values[expected],
        np.asarray(c.key_for_layer(layer).dense_matrix, np.float32).reshape(128, 128).T,
    )
    # Normalize only K constant name/data, leaving V, topology, types and shapes exact.
    original = next(
        t
        for t in models[0].graph.initializer
        if t.name == "tq_rotation_t_dense_qr_s42_d128"
    )
    for t in graph.initializer:
        if t.name == expected:
            t.CopyFrom(original)
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == expected:
                node.input[i] = original.name
    assert models[0].SerializeToString() == models[1].SerializeToString()


def test_layer_artifact_rejects_changed_matrix_and_missing_mapping(
    tmp_path: Path,
) -> None:
    _, path = config(tmp_path)
    data = json.loads(path.read_text())
    original = json.dumps(data)
    data["matrices"]["43"]["matrix_f32_sha256"] = "not-the-matrix"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="matrix checksum"):
        with_key_rotation(get_profile("k4_v4_scaled"), path)
    data = json.loads(original)
    del data["matrices"]["48"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incomplete"):
        with_key_rotation(get_profile("k4_v4_scaled"), path)


def test_source_audit_does_not_hide_wrong_query_or_v(tmp_path: Path) -> None:
    c, _ = config(tmp_path)
    model, enc = attention_part(1)
    result = quantize_current_attention(
        use_native_decoder(
            tile_kv_attention(
                apply_kv_profile(model, enc, c, 1, 35), c, 7, rotated=True
            ),
            c,
        ),
        c,
    )
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    sys.path.insert(0, str(scripts))
    try:
        verify = importlib.import_module("verify_rotated_attention")
        audit = importlib.import_module("benchmark_rotation_followup")
        graph = result.model.graph
        assert not verify.verify_rotations(graph, c.to_dict())
        signature = audit.graph_signature(graph)
        query = next(
            n
            for n in graph.node
            if n.op_type == "MatMul"
            and n.output[0].startswith("tq_attn_0_")
            and n.input[1].startswith("tq_rotation_t_")
        )
        original = query.input[1]
        query.input[1] = "tq_rotation_t_dense_qr_s48_d128"
        assert verify.verify_rotations(graph, c.to_dict())
        query.input[1] = original
        v = next(
            t for t in graph.initializer if t.name == "tq_rotation_t_dense_qr_s542_d128"
        )
        modified = numpy_helper.to_array(v).copy()
        modified[0, 0] += 0.01
        v.CopyFrom(numpy_helper.from_array(modified, v.name))
        assert audit.graph_signature(graph) != signature
        assert verify.verify_rotations(graph, c.to_dict())
    finally:
        sys.path.remove(str(scripts))


def test_layer_cache_packing_reset_and_incompatible_policy(tmp_path: Path) -> None:
    c, _ = config(tmp_path)
    a = TurboQuantKVCache(c, 2, 2, 128, 35)
    b = TurboQuantKVCache(
        replace(c, key_layers=tuple(reversed(c.key_layers))), 2, 2, 128, 35
    )
    rng = np.random.default_rng(9)
    kv = [rng.normal(size=(2, 1, 128, 3)), rng.normal(size=(2, 1, 3, 128))] * 2
    a.append(kv)
    b.append(kv)
    assert not np.array_equal(a.layers[0][0].packed, a.layers[1][0].packed)
    np.testing.assert_array_equal(a.layers[0][1].packed, a.layers[1][1].packed)
    with pytest.raises(ValueError, match="config"):
        a.load_state_dict(b.state_dict())
    before = [pair[0].packed.copy() for pair in a.layers]
    a.reset()
    a.append(kv)
    for layer in range(2):
        np.testing.assert_array_equal(before[layer], a.layers[layer][0].packed)
