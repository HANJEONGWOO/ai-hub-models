# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""A plausible summary cannot mask a modified raw loss, token identity, or split."""

import importlib
import json
import math
from pathlib import Path

import pytest


def test_reconstruct_raw_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    verify = importlib.import_module("verify_kv_refinement").verify_raw
    plan = dict(
        plan_id="anchor",
        category="anchor",
        k8_layers=[0],
        v4_layers=[],
        config_hash="mask",
    )
    windows = [
        dict(file=f"calibration_0{i}.bin", tokens_sha256=f"sha{i}", scored_tokens=1023)
        for i in range(2)
    ]
    group = tmp_path / "scores/model/calibration/anchor"
    group.mkdir(parents=True)
    means = {}
    for i, doc in enumerate(windows):
        means[doc["file"]] = float(i + 2)
        raw = dict(
            config_hash="mask",
            scorer_sha256="scorer",
            window=doc["file"],
            tokens_sha256=doc["tokens_sha256"],
            split="calibration",
            scored_tokens=1023,
            nll_sum=(i + 2) * 1023.0,
            mean_nll=i + 2.0,
        )
        path = group / (Path(doc["file"]).stem + ".json")
        path.write_text(json.dumps(raw))
    summary = {**plan, "documents": means, "nll": 2.5, "ppl": math.exp(2.5)}
    args = (tmp_path, "model", "calibration", plan, windows, "scorer")
    assert verify(*args, summary) == (2.5, set(group.glob("*.json")))
    for field, value in [
        ("nll_sum", 1023.0),
        ("tokens_sha256", "sha0"),
        ("split", "heldout"),
    ]:
        path.write_text(json.dumps({**raw, field: value}))
        with pytest.raises(ValueError, match="calibration_01"):
            verify(*args, summary)
    path.write_text(json.dumps(raw))
    for field in ("nll", "ppl"):
        with pytest.raises(ValueError, match="anchor"):
            verify(*args, {**summary, field: 9.0})
