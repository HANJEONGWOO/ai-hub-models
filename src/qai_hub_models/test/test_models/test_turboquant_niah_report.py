# SPDX-License-Identifier: BSD-3-Clause
"""NIAH retrieval scoring and incomplete-condition reporting, without a device."""

import copy
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def report(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/niah-report-test-matplotlib")
    return importlib.import_module("niah_report")


@pytest.mark.parametrize(
    ("response", "exact", "extracted", "format_issue", "count"),
    [
        ("001234", True, True, False, 1),
        (" \n001234\t ", True, True, False, 1),
        ("The code is 001234.", False, True, True, 1),
        ("001234 001234", False, False, True, 2),
        ("001234 or 981234", False, False, True, 2),
        ("981234", False, False, False, 1),
        ("1001234", False, False, True, 0),
        ("ID001234", False, False, True, 0),
        ("001234x", False, False, True, 0),
        ("1234", False, False, True, 0),
        ("\uff10\uff10\uff11\uff12\uff13\uff14", False, False, True, 0),
        ("\n\t", False, False, True, 0),
    ],
)
def test_exact_and_unique_extraction(
    report: ModuleType,
    response: str,
    exact: bool,
    extracted: bool,
    format_issue: bool,
    count: int,
) -> None:
    value = report.score_answer(response, "001234")
    assert value["exact_match"] == exact
    assert value["extracted_match"] == extracted
    assert value["format_issue"] == format_issue
    assert value["candidate_count"] == count
    assert value["empty_response"] == (not response.strip())


@pytest.mark.parametrize(
    "expected",
    [1234, "1234", "1234567", " 001234", "\uff10\uff10\uff11\uff12\uff13\uff14"],
)
def test_expected_code_never_coerced_to_integer(
    report: ModuleType, expected: str | int
) -> None:
    with pytest.raises(ValueError, match="six-digit ASCII"):
        report.score_answer("001234", expected)


def dataset(report: ModuleType) -> tuple[list[dict], dict]:
    cases = [
        {
            "case_id": f"b{base}_c{capacity}_d{depth}",
            "base_id": f"b{base}",
            "split": "test",
            "capacity": capacity,
            "depth": depth,
            "expected_answer": f"{base:06d}",
            "input_tokens": capacity - 64,
        }
        for base in range(3)
        for capacity in (1024, 4096)
        for depth in (0, 50)
    ]
    records = [
        {
            **case,
            "group": group,
            "response": case["expected_answer"],
            "status": "ok",
            "generated_tokens": 3,
            "max_new_tokens": 64,
            "finish_reason": "eos",
        }
        for group in report.GROUPS
        for case in cases
    ]
    return records, {"cases": cases}


def aggregate(report: ModuleType, records: list[dict], manifest: dict) -> dict:
    return report.aggregate_records(
        records,
        manifest,
        capacities=(1024, 4096),
        depths=(0, 50),
        bootstrap_samples=10000,
    )


def test_all_complete_and_cluster_bootstrap_reproducible(report: ModuleType) -> None:
    records, manifest = dataset(report)
    for row in records:
        if row["group"] == "boundary_structured" and row["base_id"] == "b0":
            row["response"] = "999999"
    data = aggregate(report, records, manifest)
    assert data["status"] == "complete"
    assert data["summary"]["fp16"]["exact_accuracy_pct"] == 100
    assert data["summary"]["boundary_structured"][
        "exact_accuracy_pct"
    ] == pytest.approx(200 / 3)
    comparison = next(
        row
        for row in data["paired"]["comparisons"]
        if (row["reference"], row["candidate"])
        == ("boundary_lm", "boundary_structured")
    )
    assert comparison["n_paired_cases"] == 12
    assert comparison["n_base_clusters"] == 3
    assert comparison["reference_correct_candidate_wrong"] == 4
    assert comparison["candidate_minus_reference_pp"] == pytest.approx(-100 / 3)
    assert comparison["paired_cluster_bootstrap_95pct_pp"] == [-100.0, 0.0]
    assert data["paired"] == aggregate(report, records, manifest)["paired"]


def test_errors_missing_unsupported_are_not_wrong_or_discarded(
    report: ModuleType,
) -> None:
    records, manifest = dataset(report)
    structured = [row for row in records if row["group"] == "boundary_structured"]
    structured[0]["status"] = "runtime_error"
    structured[0]["error"] = "QNN allocation failure"
    structured[1]["status"] = "unsupported"
    structured[1]["error"] = "capacity unavailable; no downsize"
    records.remove(structured[2])
    data = aggregate(report, records, manifest)
    stats = data["summary"]["boundary_structured"]
    assert data["status"] == "incomplete"
    assert stats["n_expected"] == 12
    assert stats["n_completed"] == 9
    assert stats["n_runtime_error"] == stats["n_unsupported"] == stats["n_missing"] == 1
    assert stats["exact_accuracy_pct"] == 100
    assert stats["coverage_pct"] == 75
    errors = [row for row in data["cases"] if row["status"] == "runtime_error"]
    assert errors[0]["exact_match"] is None
    assert errors[0]["response"] == errors[0]["expected_answer"]
    assert data["paired"]["complete_common_conditions"] == [
        {"capacity": 4096, "depth": 50}
    ]
    assert all(row["n_paired_cases"] == 3 for row in data["paired"]["comparisons"])


def test_empty_output_is_model_failure_and_length_flag_is_separate(
    report: ModuleType,
) -> None:
    records, manifest = dataset(report)
    records[0]["response"] = ""
    records[1]["response"] = "The answer is 000000."
    records[1]["finish_reason"] = "length"
    records[1]["generated_tokens"] = 64
    data = aggregate(report, records, manifest)
    assert data["status"] == "complete"
    row = data["summary"]["fp16"]
    assert row["n_completed"] == 12
    assert row["exact_successes"] == 10
    assert row["extracted_successes"] == 11
    assert row["n_empty"] == 1
    assert row["n_format_issue"] == 2
    assert row["n_output_truncated"] == 1


def test_invalid_duplicates_input_mismatches_and_pilot_exclusion(
    report: ModuleType,
) -> None:
    records, manifest = dataset(report)
    with pytest.raises(ValueError, match="duplicate result"):
        aggregate(report, [*records, records[0]], manifest)
    altered = copy.deepcopy(records)
    altered[0]["expected_answer"] = "123456"
    with pytest.raises(ValueError, match="input contract mismatch"):
        aggregate(report, altered, manifest)
    altered = copy.deepcopy(records)
    altered[0]["input_tokens"] = 500
    with pytest.raises(ValueError, match="input contract mismatch"):
        aggregate(report, altered, manifest)
    altered = copy.deepcopy(records)
    del altered[0]["response"]
    with pytest.raises(ValueError, match="completed result missing response"):
        aggregate(report, altered, manifest)
    records.append(
        {"split": "pilot", "group": "invalid", "response": "malformed pilot"}
    )
    assert aggregate(report, records, manifest)["status"] == "complete"


def test_no_measurement_full_axes_are_unknown_and_incomplete(
    report: ModuleType,
) -> None:
    _, manifest = dataset(report)
    data = report.aggregate_records([], manifest)
    assert data["status"] == "incomplete"
    assert data["capacities"] == [1024, 4096, 8192, 16384]
    assert data["depths"] == [0, 25, 50, 75, 100]
    assert len(data["cells"]) == 80
    assert all(row["exact_accuracy_pct"] is None for row in data["cells"])
    assert data["paired"]["complete_common_conditions"] == []
    assert all(
        row["candidate_minus_reference_pp"] is None
        for row in data["paired"]["comparisons"]
    )


def test_report_preserves_files_and_exports_gray_unknown_heatmaps(
    report: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    records, manifest = dataset(report)
    records = [row for row in records if row["capacity"] == 1024]
    data = report.aggregate_records(records, manifest)
    arrays = []
    original = Axes.imshow

    def capture(self: Any, values: Any, *args: Any, **kwargs: Any) -> Any:
        arrays.append(np.ma.asarray(values))
        return original(self, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", capture)
    destination = tmp_path / "report"
    report.write_report(data, destination)
    assert len(arrays) == 5
    for values in arrays:
        assert values.shape == (5, 4)
        assert np.ma.getmaskarray(values)[:, 1:].all()
        assert not np.ma.getmaskarray(values)[0, 0]
    stored = json.loads((destination / "summary.json").read_text())
    assert stored["status"] == "incomplete"
    assert "not completed full-context" in (destination / "report.txt").read_text()
    for basename in ("niah_accuracy", "niah_structured_minus_lm"):
        for extension in ("png", "pdf", "svg"):
            assert (destination / f"{basename}.{extension}").stat().st_size > 1000
    with pytest.raises(FileExistsError):
        report.write_report(data, destination)


def test_missing_results_can_use_manifest_metadata(report: ModuleType) -> None:
    records, manifest = dataset(report)
    reduced = [
        {key: row[key] for key in ("group", "case_id", "status", "response")}
        for row in records
    ]
    assert aggregate(report, reduced, manifest)["status"] == "complete"
