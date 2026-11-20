# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Check pairing, cohort completeness, cache accounting and frozen plan identity."""

import csv
import importlib
import json
import math
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image


@pytest.fixture
def report_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[ModuleType, Path]:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    plans = importlib.import_module("cross_model_plans")
    reporting = importlib.import_module("report_cross_model_budgets")
    root = tmp_path / "frozen"
    manifest = json.loads(plans.freeze_plans(root).read_bytes())
    rows = []
    for filename in manifest["plan_files_sha256"]:
        plan = json.loads((root / filename).read_bytes())
        k = plan["key_average_bits"]
        nll = k / 2
        if plan["placement"] == "late" and k == 6:
            nll = 1.95
        rows.append(
            {
                "model_id": plan["model_id"],
                "plan_id": plan["plan_id"],
                "config_hash": plan["config_hash"],
                "mean_nll": nll,
                "ppl": math.exp(nll),
                "scored_tokens": 4092,
                "cache_format_bytes_cl1024": plan["cache_bytes_per_token"]["total"]
                * 1024,
            }
        )
    (root / "offline_comparison.json").write_text(json.dumps({"rows": rows[::-1]}))
    return reporting, root


def test_report_tables(report_run: tuple[ModuleType, Path], tmp_path: Path) -> None:
    reporting, root = report_run
    out = tmp_path / "report"
    reporting.write_report(root, out)
    with (out / "offline_scores.csv").open() as stream:
        scores = list(csv.DictReader(stream))
    with (out / "paired_nll.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    assert len(scores) == 33 and len(pairs) == 15
    assert {(r["model_id"], r["plan_id"]) for r in scores} == set(
        reporting.load_scores(root)
    )
    for pair in pairs:
        assert float(pair["k5_minus_k4_nll"]) == 0.5
        if pair["placement"] == "late":
            assert float(pair["k5_minus_k6_nll"]) == pytest.approx(0.55)
            assert float(pair["k6_minus_k4_nll"]) == pytest.approx(-0.05)
        else:
            assert float(pair["k5_minus_k6_nll"]) == -0.5
            assert float(pair["k6_minus_k4_nll"]) == 1.0
    md = (out / "offline_report.md").read_text()
    assert md.count("k6_v2_late") == 6
    assert "28.875" in md and "37.125" in md
    assert "(offline_budget_heatmap.png)" in md
    with Image.open(out / "offline_budget_heatmap.png") as image:
        assert image.format == "PNG" and min(image.size) >= 1600
        image.verify()


def test_heatmap_shared_scale_and_labels(report_run: tuple[ModuleType, Path]) -> None:
    reporting, root = report_run
    scores = reporting.load_scores(root)
    scores["qwen3_8b", "k5_v3_early"]["mean_nll"] -= 0.25
    figure = reporting.budget_heatmap(scores, reporting.paired_rows(scores))
    assert len(figure.axes) == 4
    for model, axis in zip(reporting.MODELS, figure.axes[:3], strict=True):
        image = axis.images[0]
        assert image.norm is figure.axes[0].images[0].norm
        assert image.norm(0.0) == pytest.approx(0.5)
        assert image.get_array().shape == (2, 5)
        assert image.get_array()[0, 0] == (0.25 if model == "qwen3_8b" else 0.5)
        assert image.get_array()[1, 1] == pytest.approx(-0.05)
        assert len(axis.texts) == 10
        assert f"K4/V4 baseline PPL {math.exp(2):.2f}" in axis.get_title(loc="left")


@pytest.mark.parametrize(
    "failure", ["duplicate", "missing", "hash", "plan", "cache", "nll"]
)
def test_reject_inconsistent_scores(
    report_run: tuple[ModuleType, Path], failure: str
) -> None:
    reporting, root = report_run
    path = root / "offline_comparison.json"
    rows = json.loads(path.read_bytes())["rows"]
    if failure == "duplicate":
        rows[-1] = rows[0]
    elif failure == "missing":
        rows.pop()
    elif failure == "hash":
        rows[0]["config_hash"] = "changed"
    elif failure == "plan":
        (
            root / "plans" / rows[0]["model_id"] / (rows[0]["plan_id"] + ".json")
        ).write_text("{}")
    elif failure == "cache":
        rows[0]["cache_format_bytes_cl1024"] -= 1
    else:
        rows[0]["mean_nll"] += 1
    path.write_text(json.dumps({"rows": rows}))
    with pytest.raises(
        ValueError, match=r"comparison rows|hash changed|identity differs|Inconsistent"
    ):
        reporting.load_scores(root)
