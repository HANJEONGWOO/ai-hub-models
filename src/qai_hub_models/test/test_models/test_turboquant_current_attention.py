# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Current KV must use cache output precision, including causal prefill and QJL."""

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from qai_hub_models.models.templates.llm.turboquant.config import Rotation, get_profile
from qai_hub_models.models.templates.llm.turboquant.current_attention import (
    quantize_current_attention,
)
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)
from qai_hub_models.models.templates.llm.turboquant.native_decoder import (
    use_native_decoder,
    with_reference_decoder,
)
from qai_hub_models.models.templates.llm.turboquant.packing import (
    pack_indices,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.qjl import (
    QJLKeyReference,
    add_qjl_attention,
)
from qai_hub_models.models.templates.llm.turboquant.reference import PolarQuantReference
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)
from qai_hub_models.test.test_models import test_turboquant_tiled_attention as fixture


@pytest.mark.parametrize("mode", ["full", "tiled", "rotated", "native", "qjl", "fwht"])
@pytest.mark.parametrize(("seq", "valid"), [(1, 0), (3, 13), (128, 0)])
def test_current_cache_values_feed_attention(
    mode: str, seq: int, valid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = max(35, seq + 17)
    monkeypatch.setattr(fixture, "CONTEXT", context)
    config = get_profile(
        "k4_v4"
        if mode in ("full", "tiled")
        else "k3qjl_v4_scaled"
        if mode == "qjl"
        else "k4_v4_scaled",
        Rotation.FWHT if mode == "fwht" else Rotation.DENSE_QR,
    )
    model, enc = fixture.attention_part(seq)
    before = apply_kv_profile(model, enc, config, seq, context)
    native = mode in ("native", "qjl", "fwht")
    if mode != "full":
        before = tile_kv_attention(before, config, 7, rotated=mode != "tiled")
    if native:
        before = use_native_decoder(before, config)
    if config.qjl:
        before = add_qjl_attention(before, config)
    serialized = before.model.SerializeToString()
    result = quantize_current_attention(before, config)
    assert before.model.SerializeToString() == serialized
    assert before.encodings["param_encodings"] == result.encodings["param_encodings"]
    for side in ("input", "output"):
        assert [v.SerializeToString() for v in getattr(before.model.graph, side)] == [
            v.SerializeToString() for v in getattr(result.model.graph, side)
        ]
    oracle = with_reference_decoder(result.model) if native else result.model
    onnx.checker.check_model(oracle, full_check=True)
    assert len({n.name for n in oracle.graph.node}) == len(oracle.graph.node)
    data = fixture.feeds(seq, valid)
    rng = np.random.default_rng(47)
    refs = {
        kind: PolarQuantReference(
            getattr(config, kind),
            128,
            config.rotation,
            precomputed_norm=config.precomputed_norm,
        )
        for kind in ("key", "value")
    }
    qref = QJLKeyReference(config) if config.qjl else None
    past_kv = {}
    for kind, ref in refs.items():
        x = rng.normal(size=(2, 1, context - seq, 128))
        x[:, :, : context - seq - valid] = 0
        if kind == "key" and qref:
            packed, scale, qscale = qref.encode(x)
            data["tq_key_0_qjlscale_in"] = qscale.astype(np.float32)
            past_kv[kind] = qref.decode(packed, scale, qscale)
        else:
            indices, scale = ref.encode(x)
            packed = pack_indices(indices, 4)
            scale = scale.astype(np.float16)
            past_kv[kind] = (
                ref.rotation.inverse(
                    (ref.centroids[indices].astype(np.float16) * scale).astype(
                        np.float16
                    )
                )
                if native
                else ref.decode(indices, scale)
            )
        data[f"tq_{kind}_0_packed_in"] = packed
        data.pop(f"tq_{kind}_0_norm_in")
        data[f"tq_{kind}_0_{'scale' if config.precomputed_norm else 'norm'}_in"] = (
            scale.astype(np.float32)
        )
    values = ReferenceEvaluator(oracle).run(None, data)
    actual = dict(zip((v.name for v in oracle.graph.output), values, strict=True))
    current_kv = {}
    for kind, ref in refs.items():
        packed = actual[f"tq_{kind}_0_packed_out"]
        scale = actual[
            f"tq_{kind}_0_{'scale' if config.precomputed_norm else 'norm'}_out"
        ].astype(np.float16)
        if kind == "key" and qref:
            current_kv[kind] = qref.decode(
                packed, scale, actual["tq_key_0_qjlscale_out"].astype(np.float16)
            )
        else:
            indices = unpack_indices(packed, 4, 128)
            current_kv[kind] = (
                ref.rotation.inverse(
                    (ref.centroids[indices].astype(np.float16) * scale).astype(
                        np.float16
                    )
                )
                if native
                else ref.decode(indices, scale)
            )
        assert not np.allclose(current_kv[kind], data[f"new_{kind}"])
    for h in range(2):
        k = np.concatenate(
            (past_kv["key"][h : h + 1], current_kv["key"][h : h + 1]), axis=2
        )
        v = np.concatenate(
            (past_kv["value"][h : h + 1], current_kv["value"][h : h + 1]), axis=2
        )
        for g in range(2):
            scores = data[f"h{h}g{g}_q"] @ k.swapaxes(-1, -2) + data["mask"]
            probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            np.testing.assert_allclose(
                actual[f"h{h}g{g}_out"], probs @ v, rtol=2e-4, atol=3e-6
            )
    # No raw current rotations remain; encoder computes each rotation only once.
    assert not any(
        n.name.endswith(("_key_rotated", "_value_rotated"))
        for n in result.model.graph.node
    )
    assert len(result.current_kv_attention) == 2
    with pytest.raises(ValueError, match="applied once"):
        quantize_current_attention(result, config)


@pytest.mark.parametrize("legacy", [False, True])
def test_conversion_current_kv_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    monkeypatch.syspath_prepend(str(scripts))
    converter = importlib.import_module("convert_parts")
    split, out = tmp_path / "split", tmp_path / "out"
    split.mkdir()
    (split / "split_manifest.json").write_text(
        json.dumps(
            {"parts": {"part1_of_1": {"bundle_dir": str(split), "class": "dummy"}}}
        )
    )
    monkeypatch.setattr(converter, "convert_graph", lambda *args: {})
    monkeypatch.setattr(converter, "build_context", lambda *args: 0.0)
    argv = [
        "convert_parts",
        "--split-dir",
        str(split),
        "--out",
        str(out),
        "--profile",
        "k4_v4_scaled",
        "--no-native-decoder",
    ]
    if legacy:
        argv.append("--no-quantize-current-kv")
    monkeypatch.setattr(sys, "argv", argv)
    converter.main()
    report = json.loads((out / "convert_report.json").read_text())
    assert report["quantize_current_kv"] is not legacy
    # A different current-token policy cannot silently reuse previous binaries.
    if legacy:
        argv.remove("--no-quantize-current-kv")
    else:
        argv.append("--no-quantize-current-kv")
    with pytest.raises(ValueError, match="metadata mismatch: quantize_current_kv"):
        converter.main()


@pytest.mark.parametrize(
    "profile", ["baseline_int16_kv", "k4_v4_scaled", "k3qjl_v4_scaled"]
)
def test_export_pipeline_current_kv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    monkeypatch.syspath_prepend(str(scripts))
    converter = importlib.import_module("convert_parts")
    model, enc = fixture.attention_part(1)
    source = tmp_path / "input.onnx"
    encodings = tmp_path / "input.encodings"
    onnx.save(model, source)
    encodings.write_text(json.dumps(enc))
    enabled = profile != "baseline_int16_kv"
    args = Namespace(
        profile=profile,
        rotation=None,
        context_length=35,
        attention_tile=7 if enabled else 0,
        rotated_attention=enabled,
        native_decoder_package=Path("unused") if enabled else None,
        quantize_current_kv=True,
    )
    path, _, summary = converter.apply_profile(
        args, source, encodings, "output", 1, tmp_path
    )
    assert summary["quantize_current_kv"] == enabled
    exported = onnx.load(path)
    if enabled:
        exported = with_reference_decoder(exported)
    onnx.checker.check_model(exported, full_check=True)
    outputs = {o for n in exported.graph.node for o in n.output}
    assert ("tq_key_0_current_native_fp16" in outputs) == enabled
    assert ("tq_value_0_current_native_fp16" in outputs) == enabled
    assert ("tq_key_0_current_qjl_native_fp16" in outputs) == (
        profile == "k3qjl_v4_scaled"
    )


@pytest.mark.parametrize("quantize_current", [True, False])
@pytest.mark.parametrize("profile", ["k4_v4_scaled", "baseline_int8"])
def test_host_cache_current_policy(
    quantize_current: bool, profile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    monkeypatch.syspath_prepend(str(scripts))
    evaluator = importlib.import_module("evaluate_qwen3_kv")
    config = get_profile(profile)
    cache = evaluator.PackedPastCache(config, 1, quantize_current)
    layer = cache.layers[0]
    rng = np.random.default_rng(11)
    previous = [torch.empty(1, 2, 0, 128), torch.empty(1, 2, 0, 128)]
    for tokens in (1, 3):
        states = [
            torch.from_numpy(rng.normal(size=(1, 2, tokens, 128)).astype(np.float32))
            for _ in range(2)
        ]
        stored = [
            evaluator.codec_roundtrip(x, codec, config.norm_dtype)
            for x, codec in zip(
                states, (layer.key_codec, layer.value_codec), strict=True
            )
        ]
        attended = layer.update(*states)
        for i, cached in enumerate((layer.keys, layer.values)):
            torch.testing.assert_close(
                cached, torch.cat([previous[i], stored[i]], dim=-2)
            )
            torch.testing.assert_close(
                attended[i],
                torch.cat(
                    [previous[i], stored[i] if quantize_current else states[i]], dim=-2
                ),
            )
            previous[i] = cached.clone()
    default_cache = evaluator.PackedPastCache(config, 1)
    assert default_cache.layers[0].quantize_current_kv
    with pytest.raises(NotImplementedError, match="does not implement QJL"):
        evaluator.PackedPastCache(get_profile("k3qjl_v4_scaled"), 1)
