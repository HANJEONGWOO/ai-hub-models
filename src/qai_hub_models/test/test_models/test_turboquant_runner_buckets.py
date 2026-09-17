# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Host tests for the exact bucket selector used by the Android runner."""

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"


def test_runner_bucket_selector(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A host C++ compiler is required.")
    target = tmp_path / "buckets"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            str(SCRIPTS / "qnn_runner/test_buckets.cpp"),
            "-o",
            str(target),
        ],
        check=True,
    )
    subprocess.run([str(target)], check=True)


@pytest.mark.parametrize(
    "contents", ["", "converter running", "| Id | Name | Type |\n"]
)
def test_rotated_verifier_rejects_incomplete_dlc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: str
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    verifier = importlib.import_module("verify_rotated_attention")
    (tmp_path / "probe.kv_edits.json").write_text(
        json.dumps({"attention_tiles": [{"strategy": "rotated_precomputed_scale"}]})
    )
    (tmp_path / "probe.dlcinfo.txt").write_text(contents)
    with pytest.raises(ValueError, match="Missing/incomplete"):
        verifier.verify_graph(tmp_path, "probe")


def test_rotated_verifier_empty_bundle_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    verifier = importlib.import_module("verify_rotated_attention")
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_rotated_attention.py",
            "--bundle",
            str(tmp_path),
            "--report",
            str(report),
        ],
    )
    with pytest.raises(SystemExit) as error:
        verifier.main()
    assert error.value.code == 1
    assert json.loads(report.read_text())["passed"] is False


@pytest.mark.parametrize("profile", ["k4_v4", "k4_v4_scaled"])
def test_device_validation_selects_scalar_representation(
    monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    validation = importlib.import_module("htp_codec_validation")
    for which in ("key", "value"):
        _, codec = validation.codec_for(which, profile)
        assert codec.precomputed_norm == (profile == "k4_v4_scaled")
    assert len(validation.graph_specs(profile=profile)) == 8


def test_device_validation_range_scaling_preserves_real_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    validation = importlib.import_module("htp_codec_validation")
    snapshot = {
        f"layer{layer}_keys": np.ones((8, 129, 128), dtype=np.float32)
        for layer in (0, 27)
    }
    full = validation.encode_cases(snapshot, "key", 128)
    bounded = validation.encode_cases(snapshot, "key", 128, 0.5)
    for layer in ("layer0", "layer27"):
        np.testing.assert_array_equal(full[layer], bounded[layer])
    np.testing.assert_array_equal(full["range"] * 0.5, bounded["range"])
