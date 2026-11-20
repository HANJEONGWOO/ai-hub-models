# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Regression checks for previously seen token and document inventories."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def test_prior_npz_rows_and_nested_niah_titles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    data = importlib.import_module("refinement_data")
    historical = tmp_path / "device_assets_cl1024"
    historical.mkdir()
    for i in range(4):
        (historical / f"wikitext_w{i}.bin").write_bytes(bytes(16 * 4))
    old = tmp_path / "earlier"
    old.mkdir()
    tokens = np.arange(32, dtype="<i4")
    np.savez_compressed(old / "tokens.npz", train=tokens.astype("int64").reshape(2, 16))
    (old / "data_manifest.json").write_text(
        json.dumps(
            {
                "base_examples": [
                    {
                        "source_documents": [
                            {"title": "= Prior Document =", "sha256": "doc"}
                        ]
                    }
                ]
            }
        )
    )
    current = tmp_path / "current"
    current.mkdir()
    (current / "data_manifest.json").write_text("{}")
    manifests, files = data.inventory_paths(tmp_path, current)
    info, titles, docs, spans = data.archive_inventory(tmp_path, manifests, files)
    assert len(manifests) == 1 and len(info["prior_token_files_sha256"]) == 5
    assert titles == {"prior document"} and docs == {"doc"}
    assert tokens[:16].tobytes() in spans and tokens[16:].tobytes() in spans
    assert tokens[1:17].tobytes() not in spans
