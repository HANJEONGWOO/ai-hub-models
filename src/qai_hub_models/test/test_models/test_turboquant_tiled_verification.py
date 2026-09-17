# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""The compiled-graph verifier must never pass empty or incomplete converter output."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"


@pytest.fixture
def verifier(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("verify_tiled_attention")


@pytest.mark.parametrize(
    "contents", ["", "converter still running\n", "| Id | Name | Type |\n"]
)
def test_incomplete_dlcinfo_rejected(
    verifier: ModuleType, tmp_path: Path, contents: str
) -> None:
    (tmp_path / "probe.kv_edits.json").write_text(
        json.dumps({"attention_tiles": [{"layer": 0}]})
    )
    (tmp_path / "probe.dlcinfo.txt").write_text(contents)
    with pytest.raises(ValueError, match="missing/incomplete"):
        verifier.verify_graph(tmp_path, "probe")


def test_missing_tiling_manifest_rejected(verifier: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "probe.kv_edits.json").write_text("{}")
    with pytest.raises(ValueError, match="no tiled attention manifest"):
        verifier.verify_graph(tmp_path, "probe")


def test_failed_cli_writes_failed_report(
    verifier: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "probe.kv_edits.json").write_text(
        json.dumps({"attention_tiles": [{"layer": 0}]})
    )
    (tmp_path / "probe.dlcinfo.txt").write_text("")
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_tiled_attention.py",
            "--bundle",
            str(tmp_path),
            "--report",
            str(report),
        ],
    )
    with pytest.raises(SystemExit) as error:
        verifier.main()
    assert error.value.code == 1
    data = json.loads(report.read_text())
    assert data["passed"] is False
    assert data["graphs"]["probe"]["violations"]
