# SPDX-License-Identifier: BSD-3-Clause
"""Tight asymmetric codec contracts, independent CPU unpack and cache identity."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.export import (
    Subgraph,
    _finish,
    _repack,
    _scalar_index_tree,
    build_encode_model,
)
from qai_hub_models.models.templates.llm.turboquant.native_decoder import (
    with_reference_decoder,
)
from qai_hub_models.models.templates.llm.turboquant.packing import (
    pack_indices,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.reference import (
    PolarQuantReference,
    load_boundaries,
    load_codebook,
)
from qai_hub_models.test.test_models.test_turboquant_export import run


@pytest.mark.parametrize("bits", [2, 3, 5, 6])
def test_tight_repacking_all_codes_and_random(bits: int) -> None:
    rng = np.random.default_rng(42)
    codes = rng.integers(0, 1 << bits, (2, 1, 9, 128), dtype=np.int32)
    codes.flat[:128] = np.arange(128) % (1 << bits)
    sg = Subgraph()
    packed = _repack(sg, "codes", bits, 8, 2, 9, 128, (2, 1), "pack_")
    indices = _repack(sg, packed, 8, bits, 2, 9, 128, (2, 1), "unpack_")
    graph = _finish(
        sg,
        "packing",
        [helper.make_tensor_value_info("codes", TensorProto.INT32, list(codes.shape))],
        [
            helper.make_tensor_value_info(
                packed, TensorProto.FLOAT, [2, 1, 9, 16 * bits]
            ),
            helper.make_tensor_value_info(
                indices, TensorProto.FLOAT, list(codes.shape)
            ),
        ],
    )
    actual, restored = run(graph, {"codes": codes})
    np.testing.assert_array_equal(actual, pack_indices(codes, bits))
    np.testing.assert_array_equal(restored, codes)


@pytest.mark.parametrize("bits", [2, 3, 5, 6])
def test_tree_boundaries_and_extremes(bits: int) -> None:
    boundaries = load_boundaries(bits, 128).astype(np.float32)
    x = np.concatenate(
        [
            np.nextafter(boundaries, -np.inf),
            boundaries,
            np.nextafter(boundaries, np.inf),
            [-65504, 0, 65504],
        ]
    ).astype(np.float32)
    sg = Subgraph()
    indices = _scalar_index_tree(sg, "x", bits, 128, "tree_")
    model = _finish(
        sg,
        "tree",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [len(x)])],
        [helper.make_tensor_value_info(indices, TensorProto.INT32, [len(x)])],
    )
    (actual,) = run(model, {"x": x})
    np.testing.assert_array_equal(actual, np.searchsorted(boundaries, x, side="left"))


@pytest.mark.parametrize("profile", ["k5_v3_scaled", "k6_v2_scaled"])
@pytest.mark.parametrize("kind", ["key", "value"])
def test_asymmetric_encoder_and_native_lut(profile: str, kind: str) -> None:
    config = get_profile(profile)
    spec = getattr(config, kind)
    rng = np.random.default_rng(43)
    x = rng.normal(size=(2, 1, 9, 128)).astype(np.float32)
    x[0, 0, 0] = 0
    packed, scale = run(
        build_encode_model(config, spec, 2, 9, head_major=True), {"x": x}
    )
    codec = PolarQuantReference(spec, 128, config.rotation, precomputed_norm=True)
    expected_codes, expected_scale = codec.encode(x)
    np.testing.assert_array_equal(
        unpack_indices(packed, spec.bits, 128), expected_codes
    )
    np.testing.assert_allclose(scale, expected_scale, rtol=2e-6, atol=1e-6)
    table = load_codebook(spec.bits, 128).astype(np.float16)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    f"Decode{spec.bits}",
                    ["packed", "scale", "lut"],
                    ["out"],
                    domain="turboquant",
                )
            ],
            "native",
            [
                helper.make_tensor_value_info(
                    "packed", TensorProto.UINT8, list(packed.shape)
                ),
                helper.make_tensor_value_info(
                    "scale", TensorProto.FLOAT16, list(scale.shape)
                ),
            ],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT16, list(x.shape))],
            [numpy_helper.from_array(table.reshape(1, 1, 1, -1), "lut")],
        ),
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("turboquant", 1),
        ],
        ir_version=8,
    )
    oracle = with_reference_decoder(model)
    onnx.checker.check_model(oracle, full_check=True)
    (actual,) = run(oracle, {"packed": packed, "scale": scale.astype(np.float16)})
    expected = (
        table[expected_codes].astype(np.float32)
        * scale.astype(np.float16).astype(np.float32)
    ).astype(np.float16)
    np.testing.assert_array_equal(actual, expected)


def test_default_and_same_storage_budget() -> None:
    baseline = get_profile("k4_v4_scaled")
    assert (
        baseline.config_hash()
        == "d7ac74594b85ffd81086018773add722de8e6b9f1eee7339b929f8510374be6e"
    )
    hashes = {baseline.config_hash()}
    for name in ("k5_v3_scaled", "k6_v2_scaled"):
        config = get_profile(name)
        assert config.key.bits + config.value.bits == 8
        assert (
            config.key.seed == baseline.key.seed
            and config.value.seed == baseline.value.seed
        )
        assert config.rotation == baseline.rotation and not config.qjl
        assert config.precomputed_norm and config.norm_dtype == "float16"
        hashes.add(config.config_hash())
        bytes_per_row = 16 * (config.key.bits + config.value.bits) + 4
        assert 28 * 8 * 1024 * bytes_per_row / 2**20 == 28.875
    assert len(hashes) == 3


@pytest.mark.parametrize("profile", ["k5_v3_scaled", "k6_v2_scaled"])
def test_asymmetric_cache_append_reset_and_identity(profile: str) -> None:
    def cache(name: str) -> TurboQuantKVCache:
        return TurboQuantKVCache(get_profile(name), 2, 2, 128, 16)

    rng = np.random.default_rng(8)
    kv = [
        rng.normal(size=shape).astype(np.float32)
        for _ in range(2)
        for shape in ((2, 1, 128, 9), (2, 1, 9, 128))
    ]
    whole, stepped = cache(profile), cache(profile)
    whole.append(kv)
    for i in range(9):
        stepped.append(
            [
                v[..., i : i + 1] if j % 2 == 0 else v[:, :, i : i + 1]
                for j, v in enumerate(kv)
            ]
        )
    assert (
        whole.memory_report().allocated_bytes
        == cache("k4_v4_scaled").memory_report().allocated_bytes
    )
    for left, right in zip(whole.layers, stepped.layers, strict=True):
        for a, b in zip(left, right, strict=True):
            for name, data in a.arrays().items():
                np.testing.assert_array_equal(data, b.arrays()[name])
    state = whole.state_dict()
    with pytest.raises(ValueError, match="different TurboQuant config"):
        cache("k4_v4_scaled").load_state_dict(state)
    whole.reset()
    assert whole.get_seq_length() == 0
    for layer in whole.layers:
        for store in layer:
            assert all(not a.any() for a in store.arrays().values())
    whole.load_state_dict(state)
    assert whole.get_seq_length() == 9


@pytest.mark.parametrize("profile", ["k5_v3_scaled", "k6_v2_scaled"])
def test_asymmetric_export_rejects_old_native_package(
    profile: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    monkeypatch.syspath_prepend(str(scripts))
    converter = importlib.import_module("convert_parts")
    package = tmp_path / "native4"
    for arch in ("x86_64-linux-clang", "hexagon-v81"):
        (package / arch).mkdir(parents=True)
        (package / arch / "libTurboQuantNative.so").write_bytes(b"unused test stub")
    (package / "manifest.json").write_text(json.dumps({"operations": ["Decode4"]}))
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_parts",
            "--profile",
            profile,
            "--split-dir",
            str(tmp_path / "missing_split"),
            "--out",
            str(out),
            "--native-decoder-package",
            str(package),
        ],
    )
    with pytest.raises(SystemExit) as error:
        converter.main()
    assert error.value.code == 2
    assert "Native package lacks the selected bit widths" in capsys.readouterr().err
    assert not out.exists()
