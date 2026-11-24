# SPDX-License-Identifier: BSD-3-Clause
"""The one-shot diagnostic must preserve its allocation and honest time units."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("profile_structured_stages_once")


def config() -> dict:
    return {
        "layer_plan": {"bits": [[8 if i % 2 else 4, 2] for i in range(28)]},
        "rotation": "dense_qr",
        "qjl": False,
        "norm_correction": True,
        "norm_representation": "effective_scale",
        "norm_dtype": "float16",
        "structured_k8": {},
    }


def test_only_selected_14_layer_configuration(harness: ModuleType) -> None:
    c = config()
    harness.validate_config(c, list(range(1, 28, 2)))
    with pytest.raises(ValueError, match="mask"):
        harness.validate_config(c, list(range(0, 28, 2)))
    c["layer_plan"]["bits"][0] = [8, 2]
    with pytest.raises(ValueError, match="exactly 14"):
        harness.validate_config(c, list(range(1, 28, 2)))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rotation", "wht"),
        ("qjl", True),
        ("norm_correction", False),
        ("norm_representation", "norm"),
        ("norm_dtype", "float32"),
    ],
)
def test_reject_changed_numerics(harness: ModuleType, key: str, value: object) -> None:
    c = config()
    c[key] = value
    with pytest.raises(ValueError, match="numerical"):
        harness.validate_config(c, list(range(1, 28, 2)))


def test_seven_stage_definition_and_route(harness: ModuleType) -> None:
    summary = importlib.import_module("summarize_structured_stages")
    assert len(summary.SEVEN) == len(set(summary.SEVEN)) == 7
    assert sum(p == "prefill" for p, _ in summary.SEVEN) == 3
    assert sum(p == "decode_126" for p, _ in summary.SEVEN) == 4
    assert summary.route({"kv": "key", "layer": "0"}, {0}) == "Structured K8"
    assert summary.route({"kv": "key", "layer": "1"}, {0}) == "K4"
    assert summary.route({"kv": "value", "layer": "0"}, {0}) == "V2"
    assert summary.route({"kv": "key", "layer": ""}, {0}) == "unassigned"


def test_no_invented_seconds_or_invocations(harness: ModuleType) -> None:
    summary = importlib.import_module("summarize_structured_stages")
    entry = {
        "busy_cycles": 120,
        "events": 3,
        "percent_compute_busy_cycles": 60,
        "interval_union_cycles": 50,
        "by_kv": {"key": 120},
        "resources": {"HVX": 120},
    }
    data = {
        "prefill": {"stages": {"encode.pack": entry}, "compute_busy_cycles": 200},
        "decode": {},
    }
    row = summary.stage_rows("structured_k8", data)[0]
    assert row["stage_wall_ms"] is row["stage_wall_s"] is None
    assert row["denominator_busy_cycles"] == 200
    assert row["mean_cycles_per_physical_lane_event"] == 40
    assert row["physical_lane_events"] == 3
    assert row["interval_union_cycles"] != row["busy_cycles"]


def test_sdk_time_is_not_cycle_conversion(harness: ModuleType) -> None:
    summary = importlib.import_module("summarize_structured_stages")
    raw = {
        "op_profiles": [
            {
                "step": "decode_126",
                "ar": 1,
                "cached_before": 1023,
                "parts": [
                    {
                        "timings": [
                            {"event": "Accelerator (execute) time", "us": 1500},
                            {
                                "event": "Accelerator (execute excluding wait) time",
                                "us": 1000,
                            },
                        ]
                    }
                ],
            }
        ]
    }
    row = summary.instrumented_times("structured_k8", raw)[0]
    assert row["accelerator_execute_ms"] == 1.5
    assert row["accelerator_execute_s"] == 0.0015


def test_device_capacity_reserves_trace_space(
    harness: ModuleType, tmp_path: Path
) -> None:
    for i in range(1, 5):
        p = tmp_path / harness.GROUP / f"part{i}_of_4" / f"part{i}_of_4.bin"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"test")
    assert harness.device_space_required(tmp_path) == 16 + 512 * 1024**2


def test_cleanup_refuses_unapproved_directory(
    harness: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="restricted"):
        harness.release_control_device_copy(
            tmp_path, Path("nonexistent-adb"), {"historical_controls": "/tmp/unrelated"}
        )


def test_immutable_report_write(harness: ModuleType, tmp_path: Path) -> None:
    summary = importlib.import_module("summarize_structured_stages")
    p = tmp_path / "rows.csv"
    summary.csv_new(p, [{"x": 1}])
    with pytest.raises(FileExistsError):
        summary.csv_new(p, [{"x": 2}])
    assert "1" in p.read_text()
