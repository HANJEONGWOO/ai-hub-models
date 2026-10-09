# SPDX-License-Identifier: BSD-3-Clause
"""Validation-only single-pass NLL experiment integrity and aggregation."""

import csv
import importlib
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def experiment() -> Iterator[ModuleType]:
    scripts = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"
    sys.path.insert(0, str(scripts))
    try:
        yield importlib.import_module("benchmark_rotation_mse_nll")
    finally:
        sys.path.remove(str(scripts))


def valid_report() -> dict:
    return {
        "mode": "score",
        "context_length": 1024,
        "prompt_tokens": 1024,
        "scored_tokens": 1023,
        "assets": {"tokens_file": "validation_0.bin", "tokens_sha256": "frozen"},
        "quantize_current_kv": True,
        "context_buckets": [1024],
        "backend_api": "2.37.0/htp 5.48.0",
        "nll_sum": 2046.0,
        "ppl": math.exp(2),
    }


def test_weighted_nll_not_arithmetic_ppl(experiment: ModuleType) -> None:
    rows = [{"scored_tokens": 3, "nll_sum": 6}, {"scored_tokens": 1, "nll_sum": 4}]
    result = experiment.aggregate(rows)
    assert result == {
        "scored_tokens": 4,
        "nll_sum": 10,
        "mean_nll": 2.5,
        "ppl": math.exp(2.5),
    }
    assert result["ppl"] != (math.exp(2) + math.exp(4)) / 2


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"scored_tokens": 1, "nll_sum": math.nan}],
        [{"scored_tokens": 1, "nll_sum": -1}],
    ],
)
def test_invalid_nll_rejected(experiment: ModuleType, rows: list[dict]) -> None:
    with pytest.raises(ValueError, match="Invalid NLL"):
        experiment.aggregate(rows)


def test_exact_scoring_contract(experiment: ModuleType) -> None:
    report = valid_report()
    experiment.validate_score(
        report, {"file": "validation_0.bin", "tokens_sha256": "frozen"}
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "generate"),
        ("context_length", 2048),
        ("prompt_tokens", 1023),
        ("scored_tokens", 1024),
        ("quantize_current_kv", False),
        ("diagnostic_only", True),
        ("context_buckets", [512, 1024]),
        ("backend_api", "cpu"),
        ("nll_sum", math.nan),
        ("nll_sum", -1),
        ("ppl", 9),
        ("assets", {"tokens_file": "test_0.bin", "tokens_sha256": "frozen"}),
        ("assets", {"tokens_file": "validation_0.bin", "tokens_sha256": "changed"}),
    ],
)
def test_changed_scoring_contract_rejected(
    experiment: ModuleType, field: str, value: object
) -> None:
    report = valid_report()
    report[field] = value
    with pytest.raises(ValueError, match=r"Wrong input|Inconsistent"):
        experiment.validate_score(
            report, {"file": "validation_0.bin", "tokens_sha256": "frozen"}
        )


def test_frozen_single_pass_plan(experiment: ModuleType) -> None:
    orders = experiment.PROTOCOL["document_orders"]
    assert len(orders) == 4
    assert all(sorted(order) == ["A", "B", "P"] for order in orders)
    assert experiment.PROTOCOL["repeats"] == 1


def test_outputs_never_overwritten(experiment: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    experiment.write_json(path, {"original": True})
    with pytest.raises(FileExistsError):
        experiment.write_json(path, {"original": False})
    assert experiment.read(path) == {"original": True}


def test_summary_documents_groups_and_csv(
    experiment: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "reports").mkdir()
    windows = [
        {
            "title": str(i),
            "offset": i,
            "file": f"validation_{i}.bin",
            "tokens_sha256": "frozen",
        }
        for i in range(4)
    ]
    experiment.write_json(
        tmp_path / "data_manifest.json", {"windows": {"validation": windows}}
    )
    experiment.write_json(
        tmp_path / "attention_mse.json",
        {
            "groups": {
                g: {"mse": 1.0, "layers": {str(l): {"mse": 1.0} for l in range(28)}}
                for g in "ABP"
            }
        },
    )
    experiment.write_json(
        tmp_path / "reports/experiment.json",
        {"groups": {g: {"config_hash": g} for g in "ABP"}},
    )
    experiment.write_json(
        tmp_path / "reports/preservation.json", {"original_files_unchanged": True}
    )
    for i, window in enumerate(windows):
        for k, group in enumerate("ABP"):
            report = valid_report()
            report["assets"]["tokens_file"] = window["file"]
            report["config_hash"] = group
            report["nll_sum"] = (2 + k * 0.1) * 1023
            report["ppl"] = math.exp(2 + k * 0.1)
            experiment.write_json(
                tmp_path / f"reports/{group}_validation_w{i}.json", report
            )
    experiment.summarize(tmp_path)
    result = experiment.read(tmp_path / "reports/comparison.json")
    assert result["groups"]["P"]["scored_tokens"] == 4092
    assert result["pairs"]["P_vs_B"]["mean_nll_delta"] == pytest.approx(0.1)
    assert result["pairs"]["P_vs_B"]["documents_nll_improved"] == 0
    with (tmp_path / "reports/document_nll.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert all(set(row) == set(rows[0]) for row in rows)
    assert len(experiment.read(tmp_path / "reports/document_nll.json")) == 12
