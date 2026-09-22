# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Dense export identity and incompatible-cache protection."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qai_hub_models.models.templates.llm.turboquant.cache import TurboQuantKVCache
from qai_hub_models.models.templates.llm.turboquant.config import Rotation, get_profile
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)
from qai_hub_models.test.test_models.test_turboquant_tiled_attention import (
    CONTEXT,
    attention_part,
)

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"


@pytest.mark.parametrize("rotation", list(Rotation))
def test_rotations_are_audited(
    rotation: Rotation, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    verifier = importlib.import_module("verify_rotated_attention")
    config = get_profile("k4_v4_scaled", rotation)
    model, encodings = attention_part(1)
    graph = tile_kv_attention(
        apply_kv_profile(model, encodings, config, 1, CONTEXT), config, 7, rotated=True
    ).model.graph
    assert not verifier.verify_rotations(graph, config.to_dict())
    other = Rotation.FWHT if rotation == Rotation.DENSE_QR else Rotation.DENSE_QR
    other_config = get_profile(config.profile, other)
    assert verifier.verify_rotations(graph, other_config.to_dict())
    other_graph = tile_kv_attention(
        apply_kv_profile(model, encodings, other_config, 1, CONTEXT),
        other_config,
        7,
        rotated=True,
    ).model.graph
    assert verifier.rotation_only_change(graph, other_graph)
    other_graph.node[0].name += "_changed"
    assert not verifier.rotation_only_change(graph, other_graph)
    for tensor in graph.initializer:
        if tensor.name.startswith("tq_rotation"):
            tensor.raw_data = np.zeros((128, 128), dtype=np.float32).tobytes()
            break
    assert verifier.verify_rotations(graph, config.to_dict())


def test_dense_and_fwht_cache_states_cannot_be_mixed() -> None:
    dense = TurboQuantKVCache(get_profile("k4_v4_scaled"), 1, 2, 128, 16)
    fwht = TurboQuantKVCache(get_profile("k4_v4_scaled", Rotation.FWHT), 1, 2, 128, 16)
    for source, target in ((dense, fwht), (fwht, dense)):
        with pytest.raises(ValueError, match="different TurboQuant config"):
            target.load_state_dict(source.state_dict())


def test_dense_default_cannot_overwrite_old_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    converter = importlib.import_module("convert_parts")
    split = tmp_path / "split"
    split.mkdir()
    (split / "split_manifest.json").write_text('{"parts": {}}')
    output = tmp_path / "fwht"
    output.mkdir()
    report = output / "convert_report.json"
    original = json.dumps(
        {"config_hash": get_profile("k4_v4_scaled", Rotation.FWHT).config_hash()}
    )
    report.write_text(original)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_parts",
            "--split-dir",
            str(split),
            "--out",
            str(output),
            "--profile",
            "k4_v4_scaled",
            "--no-native-decoder",
        ],
    )
    with pytest.raises(ValueError, match="metadata mismatch: config_hash"):
        converter.main()
    assert report.read_text() == original


@pytest.mark.parametrize(
    "bad", [None, "missing", "rotation", "graph", "package", "current"]
)
def test_assemble_partial_native_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str | None
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    assembler = importlib.import_module("assemble_native_bundle")
    sources = [tmp_path / "base", tmp_path / "parts"]
    for source, numbers in zip(sources, ([1, 2], [3]), strict=True):
        source.mkdir()
        report: dict[str, Any] = {
            "config_hash": "dense",
            "context_length": 1024,
            "context_buckets": [1024],
            "attention_tile": 256,
            "rotated_attention": True,
            "native_decoder": {"sha256": "same-package"},
            "parts": {},
        }
        for number in numbers:
            name = f"part{number}_of_3"
            graph = f"token_ar1_cl1024_{number}_of_3"
            report["parts"][name] = {
                "context_s": 1,
                "graphs": {
                    graph: {"surgery": {"codec_io": [1], "native_decoder": True}}
                },
            }
            (source / f"{name}.bin").write_bytes(b"test context")
        if source == sources[1]:
            if bad == "missing":
                report["parts"] = {}
            elif bad == "rotation":
                report["config_hash"] = "fwht"
            elif bad == "graph":
                report["parts"]["part3_of_3"]["graphs"] = {"wrong": {}}
            elif bad == "package":
                report["native_decoder"] = {"sha256": "different"}
            elif bad == "current":
                report["quantize_current_kv"] = True
        (source / "convert_report.json").write_text(json.dumps(report))
    output = tmp_path / "combined"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assemble",
            "--base",
            str(sources[0]),
            "--parts",
            str(sources[1]),
            "--out",
            str(output),
        ],
    )
    if bad:
        with pytest.raises(
            ValueError, match=r"Incomplete|Incompatible|suffix|packages"
        ):
            assembler.main()
        assert not output.exists()
    else:
        assembler.main()
        result = json.loads((output / "convert_report.json").read_text())
        assert len(result["parts"]) == 3
        assert (output / "part2_of_3.bin").resolve() == sources[0] / "part2_of_3.bin"
