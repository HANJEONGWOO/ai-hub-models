# SPDX-License-Identifier: BSD-3-Clause
"""K-only rotation identity, graph invariants, hard forward and cache isolation."""

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import numpy_helper

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import (
    Rotation,
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
from qai_hub_models.models.templates.llm.turboquant.packing import (
    pack_indices,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.reference import (
    DenseQRRotation,
    PolarQuantReference,
)
from qai_hub_models.models.templates.llm.turboquant.rotation_artifact import (
    load_rotation,
    save_rotation,
    with_key_rotation,
)
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)
from qai_hub_models.test.test_models.test_turboquant_tiled_attention import (
    attention_part,
)


def custom(tmp_path: Path, seed: int = 43) -> tuple[TurboQuantConfig, Path]:
    path = tmp_path / f"rotation{seed}.json"
    matrix = DenseQRRotation(seed, 128).matrix().astype(np.float32)
    save_rotation(path, matrix, {"seed": seed})
    return with_key_rotation(get_profile("k4_v4_scaled"), path), path


def test_default_hash_unchanged() -> None:
    assert (
        get_profile("k4_v4_scaled").config_hash()
        == "d7ac74594b85ffd81086018773add722de8e6b9f1eee7339b929f8510374be6e"
    )


def test_artifact_validation_and_identity(tmp_path: Path) -> None:
    config, path = custom(tmp_path)
    assert config.value == get_profile("k4_v4_scaled").value
    assert config.config_hash() != get_profile("k4_v4_scaled").config_hash()
    values = json.loads(path.read_text())
    values["matrix_f32_sha256"] = "bad"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="checksum"):
        load_rotation(path)
    with pytest.raises(ValueError, match="orthogonal"):
        save_rotation(tmp_path / "bad.json", np.ones((128, 128)), {})
    with pytest.raises(ValueError, match="requires"):
        replace(config, rotation=Rotation.FWHT)
    with pytest.raises(ValueError, match="K-only"):
        replace(config, value=config.key)


def test_qk_equivalence_packing_and_reset(tmp_path: Path) -> None:
    config, _ = custom(tmp_path)
    codec = PolarQuantReference(config.key, precomputed_norm=True)
    rng = np.random.default_rng(8)
    q, k = rng.normal(size=(9, 128)), rng.normal(size=(11, 128))
    np.testing.assert_allclose(
        codec.rotation.forward(q) @ codec.rotation.forward(k).T, q @ k.T, atol=5e-6
    )
    indices, scale = codec.encode(k)
    np.testing.assert_array_equal(
        unpack_indices(pack_indices(indices, 4), 4, 128), indices
    )
    assert np.isfinite(codec.decode(indices, scale)).all()
    cache = TurboQuantKVCache(config, 1, 2, 128, 35)
    base = TurboQuantKVCache(get_profile("k4_v4_scaled"), 1, 2, 128, 35)
    data = [rng.normal(size=(2, 1, 128, 3)), rng.normal(size=(2, 1, 3, 128))]
    cache.append(data)
    base.append(data)
    np.testing.assert_array_equal(cache.layers[0][1].packed, base.layers[0][1].packed)
    with pytest.raises(ValueError, match="config"):
        base.load_state_dict(cache.state_dict())
    before = cache.layers[0][0].packed.copy()
    cache.reset()
    assert cache.get_seq_length() == 0
    cache.append(data)
    np.testing.assert_array_equal(before, cache.layers[0][0].packed)


@pytest.mark.parametrize("seq", [1, 3])
def test_only_k_constant_changes_current_past_gqa(tmp_path: Path, seq: int) -> None:
    config, _ = custom(tmp_path)
    graphs = []
    for c in (get_profile("k4_v4_scaled"), config):
        model, enc = attention_part(seq)
        result = apply_kv_profile(model, enc, c, seq, 35)
        result = tile_kv_attention(result, c, 7, rotated=True)
        result = use_native_decoder(result, c)
        result = quantize_current_attention(result, c)
        graphs.append(result.model)
        assert len(result.current_kv_attention) == 2
    name = "tq_rotation_t_dense_qr_s42_d128"
    for model in graphs:
        assert sum(node.op_type == "MatMul" for node in model.graph.node) > 0
        assert all("Bitplane" not in node.op_type for node in model.graph.node)
    constants = [{t.name: t for t in m.graph.initializer} for m in graphs]
    assert not np.array_equal(
        numpy_helper.to_array(constants[0][name]),
        numpy_helper.to_array(constants[1][name]),
    )
    graphs[1].graph.initializer[list(constants[1]).index(name)].CopyFrom(
        constants[0][name]
    )
    assert graphs[0].SerializeToString() == graphs[1].SerializeToString()


def test_hard_forward_identical_with_and_without_gradients() -> None:
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    sys.path.insert(0, str(scripts))
    try:
        module = importlib.import_module("learn_key_rotation")
        torch.manual_seed(0)
        matrix = torch.tensor(DenseQRRotation(42, 128).matrix().astype(np.float32))
        x = torch.randn(3, 128)
        x[0] = 0
        direct = module.codec(x, matrix)
        matrix.requires_grad_()
        differentiable = module.codec(x, matrix)
        for a, b in zip(direct, differentiable, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        differentiable[0].square().sum().backward()
        assert torch.isfinite(matrix.grad).all()
    finally:
        sys.path.remove(str(scripts))


def test_capture_adds_outputs_without_changing_calibration() -> None:
    from qai_hub_models.models.templates.llm.turboquant.attention_capture import (
        add_attention_capture,
    )
    from qai_hub_models.models.templates.llm.turboquant.fp16_attention import (
        use_fp16_kv_attention,
    )

    config = get_profile("baseline_fp16_kv_fp16_attn")
    model, encodings = attention_part(3)
    result = use_fp16_kv_attention(
        apply_kv_profile(model, encodings, config, 3, 35), config
    )
    before_encodings = json.dumps(result.encodings, sort_keys=True)
    before_nodes = [n.SerializeToString() for n in result.model.graph.node]
    report = add_attention_capture(result, 3)
    assert len(report["0"]) == 4
    assert before_encodings == json.dumps(result.encodings, sort_keys=True)
    assert [n.SerializeToString() for n in result.model.graph.node[:-2]] == before_nodes
    onnx.checker.check_model(result.model, full_check=True)


@pytest.mark.parametrize("seq", [1, 128])
def test_conditioned_probe_preserves_current_past_and_gqa(seq: int) -> None:
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    sys.path.insert(0, str(scripts))
    try:
        module = importlib.import_module("rotation_probe")
        inputs, outputs = {}, {}
        for kind in ("key", "value"):
            for target, length, suffix in (
                (inputs, 1024 - seq, "in"),
                (outputs, seq, "out"),
            ):
                packed = np.empty((2, 1, length, 64), np.uint8)
                packed[0], packed[1] = 0x88, 0x99
                target[f"tq_{kind}_0_packed_{suffix}"] = packed
                target[f"tq_{kind}_0_scale_{suffix}"] = np.ones(
                    (2, 1, length, 1), np.float32
                )
        inputs["mask"] = np.zeros((1, 1, seq, 1024), np.float32)
        for h in range(2):
            for g in range(2):
                inputs[f"h{h}g{g}_q"] = np.zeros((1, 1, seq, 128), np.float32)
        actual = module.conditioned_attention(
            inputs, outputs, get_profile("k4_v4_scaled")
        )
        constant_v = module.CENTROIDS[torch.tensor([8, 8, 9, 9])][:, None, None]
        expected = module.rounded(
            constant_v.expand(4, seq, 128) @ module.rounded(module.VR)
        ).numpy()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)
    finally:
        sys.path.remove(str(scripts))
