# SPDX-License-Identifier: BSD-3-Clause
"""Repeated-performance orchestration, without executing a device or changing codecs."""

import copy
import csv
import importlib
import json
import math
from collections import Counter
from io import StringIO
from itertools import combinations
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def repeat(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("benchmark_k8_repeat")


def test_balanced_predeclared_schedule(repeat: ModuleType) -> None:
    orders = repeat.schedule()
    assert len(orders) == 10
    assert all(set(row) == set(repeat.GROUPS) and len(row) == 4 for row in orders)
    for group in repeat.GROUPS:
        assert sorted(Counter(row.index(group) for row in orders).values()) == [
            2,
            2,
            3,
            3,
        ]
    for a, b in combinations(repeat.GROUPS, 2):
        assert sum(row.index(a) < row.index(b) for row in orders) == 5


def test_boundary_only_changes_mask_not_codec_or_old_hashes(repeat: ModuleType) -> None:
    odd = repeat.config("odd_lm")
    boundary = repeat.config("boundary_lm")
    assert (
        odd.config_hash()
        == "361bdafb45505ef7334c26dfc9e08023ef67cd0633fb30a5588bc9ab71bb4a5c"
    )
    assert (
        repeat.config("odd_structured").config_hash()
        == "b68d0d5534b516363c892ecf65bd98f44789969d4e07cc31de73e7bdc3d11ed9"
    )
    assert (
        repeat.config("boundary_structured").config_hash()
        == "c6e905f027b370548fba9ca92e51abd627d170e363573c9f767d473492039d60"
    )
    assert boundary.layer_bits == repeat.config("boundary_structured").layer_bits
    assert boundary.converged_k8 and not boundary.structured_k8
    assert sum(sum(bits) for bits in boundary.layer_bits) == 28 * 8
    for layer in range(28):
        assert boundary.codec_for("value", layer) == odd.codec_for("value", layer)
        matching = 1 if boundary.layer_bits[layer][0] == 8 else 0
        assert boundary.codec_for("key", layer) == odd.codec_for("key", matching)
    for name in (
        "rotation",
        "key",
        "value",
        "norm_correction",
        "norm_dtype",
        "precomputed_norm",
        "qjl",
    ):
        assert getattr(odd, name) == getattr(boundary, name)


def test_only_current_hal_temperatures_and_no_htp_claim(repeat: ModuleType) -> None:
    raw = """Thermal Status: 0
Cached temperatures:
 Temperature{mValue=99.0, mType=0, mName=AP, mStatus=5}
Current temperatures from HAL:
 Temperature{mValue=30.5, mType=0, mName=AP, mStatus=0}
 Temperature{mValue=NaN, mType=2, mName=BAT, mStatus=0}
Current cooling devices from HAL:
"""
    parsed = repeat.parse_thermal(raw)
    assert parsed["current_sensors"]["AP"]["celsius"] == 30.5
    assert "BAT" not in parsed["current_sensors"]
    assert parsed["android_status"] == 0
    assert parsed["htp_specific_throttling"].startswith("unknown")
    summary = repeat.thermal_summary([{"thermal": parsed}])
    assert summary["ap_temp_mean_c"] == 30.5
    assert not summary["android_thermal_warning"]
    assert summary["htp_throttling"] == "unknown"


def sample_report(repeat: ModuleType) -> dict:
    session = {
        "session": 1,
        "generated": [0] * 128,
        "cached_tokens_at_end": 1024,
        "ttft_s": 0.5,
        "prefill_tok_per_s": 1794.0,
        "decode_tok_per_s": 40.0,
    }
    warmup = {
        **session,
        "session": 0,
        "ttft_s": 2.0,
        "prefill_tok_per_s": 448.5,
        "decode_tok_per_s": 20.0,
    }
    return {
        "mode": "generate",
        "sessions": [warmup, session],
        "generated": session["generated"],
        "prompt_tokens": 897,
        "context_length": 1024,
        "context_buckets": [1024],
        "steps": [{"kind": "decode", "graph_context": 1024}] * 127,
        "decode_step_s": [0.025] * 127,
        "burst_power": True,
        "kv_store_bytes": 30277632,
        "io_buffer_bytes": 100,
        "config_hash": repeat.config("odd_structured").config_hash(),
        "quantize_current_kv": True,
        "assets": {"tokens_sha256": "input"},
        "prefill_s": 0.5,
        **{k: session[k] for k in ("ttft_s", "prefill_tok_per_s", "decode_tok_per_s")},
    }


def test_warmup_exclusion_and_exact_token_accounting(repeat: ModuleType) -> None:
    data = sample_report(repeat)
    metrics, flags = repeat.extract_metrics(data, "odd_structured", "input")
    assert metrics["ttft_ms"] == 500 and metrics["warmup_ttft_ms"] == 2000
    assert metrics["decode_tok_per_s"] == 40
    assert metrics["cached_tokens_at_end"] == 897 + 128 - 1
    assert metrics["decode_iterations"] == 127 and not flags
    for key, value in (
        ("burst_power", False),
        ("config_hash", "wrong"),
        ("diagnostic_only", True),
    ):
        broken = copy.deepcopy(data)
        broken[key] = value
        with pytest.raises(ValueError, match="contract failed"):
            repeat.extract_metrics(broken, "odd_structured", "input")
    broken = copy.deepcopy(data)
    broken["sessions"][1]["cached_tokens_at_end"] = 1025
    with pytest.raises(ValueError, match="contract failed"):
        repeat.extract_metrics(broken, "odd_structured", "input")


def test_sample_sd_and_improvement_directions(repeat: ModuleType) -> None:
    stats = repeat.describe([1, 2, 3])
    assert stats == {"n": 3, "mean": 2, "sample_sd": 1, "median": 2, "min": 1, "max": 3}
    assert repeat.improvement(800, 500, "ttft_ms") == 37.5
    assert math.isclose(repeat.improvement(32, 36, "decode_tok_per_s"), 12.5)
    assert repeat.improvement(100, 120, "ttft_ms") == -20


def test_telemetry_wraps_only_runner_and_keeps_raw_snapshots(
    repeat: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import argparse

    monkeypatch.setattr(repeat, "snapshot", lambda full=False: {"full": full})
    record: dict = {}
    samples: list[dict] = []
    stream = StringIO()
    wrapped = repeat.instrument_adb(
        lambda *args, **kwargs: "ok", record, tmp_path, samples, stream
    )
    args = argparse.Namespace()
    assert wrapped(args, "shell", "getprop") == "ok"
    assert not samples
    assert wrapped(args, "shell", "/bin/qnn-llm-runner; echo __RC__$?") == "ok"
    assert samples == [{"full": True}, {"full": True}]
    assert "runner_start_utc" in record and "runner_end_utc" in record
    assert len(stream.getvalue().splitlines()) == 2
    assert (tmp_path / "runner_command.json").exists()


def test_missing_and_warning_telemetry_are_not_claimed_normal(
    repeat: ModuleType,
) -> None:
    assert repeat.describe([])["n"] == 0
    assert repeat.thermal_summary([])["thermal_status_max"] is None
    bad = repeat.thermal_summary(
        [{"thermal": {"android_status": 2, "current_sensors": {}}}]
    )
    assert bad["android_thermal_warning"]
    assert bad["htp_throttling"] == "unknown"


def test_all_forty_samples_including_warning_outlier_are_reported(
    repeat: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(repeat, "PREVIOUS", tmp_path)
    monkeypatch.setattr(repeat, "BOUNDARY_SOURCE", tmp_path / "boundary")
    with (tmp_path / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", *repeat.METRICS])
        writer.writeheader()
        for group in ("structured", "lm8_converged"):
            writer.writerow(
                dict(
                    group=group,
                    ttft_ms=800,
                    prefill_tok_per_s=1200,
                    decode_tok_per_s=32,
                )
            )
    repeat.save_new(
        tmp_path / "reports/boundary_perf.json",
        dict(ttft_s=0.5, prefill_tok_per_s=1800, decode_tok_per_s=38),
    )
    for rnd, order in enumerate(repeat.schedule(), 1):
        for pos, group in enumerate(order, 1):
            ordinal = (rnd - 1) * 4 + pos
            warning = ordinal == 1
            repeat.save_new(
                tmp_path / f"attempts/{ordinal:02d}/result.json",
                {
                    "round": rnd,
                    "position": pos,
                    "ordinal": ordinal,
                    "group": group,
                    "ttft_ms": 1500
                    if warning
                    else 500
                    if group.endswith("structured")
                    else 800,
                    "prefill_tok_per_s": 1800,
                    "decode_tok_per_s": 38,
                    "flags": ["android_thermal_warning"] if warning else [],
                    "thermal_status_max": 2 if warning else 0,
                    "ap_temp_mean_c": 25 + rnd,
                },
            )
    repeat.save_new(tmp_path / "measurement_complete.json", {"measured": 40})
    repeat.analyze(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["summary"]["odd_structured"]["ttft_ms"]["n"] == 10
    assert summary["summary"]["odd_structured"]["ttft_ms"]["mean"] == 600
    assert summary["diagnostics"]["odd_structured"]["flags"] == [1]
    with (tmp_path / "raw.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 40
    reporter = importlib.import_module("report_k8_repeat")
    reporter.report(tmp_path)
    assert "600.000" in (tmp_path / "paper_revision.tex").read_text()
    assert "Flagged samples retained: ['1']" in (tmp_path / "report.txt").read_text()
