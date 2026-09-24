# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Explicit 4B selection, model provenance and single-run benchmark guards."""

import copy
import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("benchmark_model_once")


def model_info(model_id: str = "qwen3_4b") -> dict[str, Any]:
    small = model_id == "qwen3_1_7b"
    return {
        "model_id": model_id,
        "architecture": {
            "model_type": "qwen3",
            "num_hidden_layers": 28 if small else 36,
            "hidden_size": 2048 if small else 2560,
            "num_attention_heads": 16 if small else 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
        },
        "config_tokenizer_sha256": {"config.json": "same-config"},
    }


def prepare_bundles(work: Path, benchmark: ModuleType, parts: int = 4) -> None:
    model = model_info()
    split = {
        "model_id": "qwen3_4b",
        "model": model,
        "parts": {f"part{i}_of_{parts}": {} for i in range(1, parts + 1)},
    }
    (work / "split").mkdir()
    source = work / "split/split_manifest.json"
    source.write_text(json.dumps(split))
    (work / "assets").mkdir()
    assets: dict[str, Any] = {
        "model": model,
        "context_length": 1024,
        "rope_half": 64,
        "rope": "rope.bin",
        "wikitext_windows": [f"w{i}.bin" for i in range(4)],
        "sha256": {},
    }
    for name in ["rope.bin", *assets["wikitext_windows"]]:
        path = work / "assets" / name
        path.write_bytes(b"input")
        assets["sha256"][name] = benchmark.sha256_file(path)
    (work / "assets/assets.json").write_text(json.dumps(assets))
    for group, profile in benchmark.GROUPS.items():
        out = work / group
        out.mkdir()
        metadata = {
            "model": model,
            "split_manifest_sha256": benchmark.sha256_file(source),
            "num_parts": parts,
            "profile": profile,
            "config_hash": group,
            "config": {"rotation": "dense_qr", "qjl": False},
            "native_decoder": {"libraries": {"hexagon-v81": {"sha256": "native"}}}
            if group == "turboquant"
            else None,
            "attention_tile": 256 if group == "turboquant" else 0,
            "rotated_attention": group == "turboquant",
            "quantize_current_kv": group == "turboquant",
            "context_length": 1024,
            "context_buckets": benchmark.BUCKETS[group],
            "parts": {name: {"context_s": 1, "graphs": {}} for name in split["parts"]},
        }
        (out / "convert_report.json").write_text(json.dumps(metadata))
        for part in split["parts"]:
            (out / f"{part}.bin").write_bytes(b"context")


def test_default_remains_1_7b(benchmark: ModuleType) -> None:
    assert (
        benchmark.parser().parse_args(["build", "--work-dir", "unused"]).model_id
        == "qwen3_1_7b"
    )
    assert (
        benchmark.parser()
        .parse_args(["build", "--work-dir", "unused", "--model-id", "qwen3_4b"])
        .model_id
        == "qwen3_4b"
    )


@pytest.mark.parametrize(
    ("model_id", "int16", "turboquant"),
    [("qwen3_1_7b", 112, 28.875), ("qwen3_4b", 144, 37.125)],
)
def test_model_specific_kv_memory(
    benchmark: ModuleType, model_id: str, int16: float, turboquant: float
) -> None:
    model = model_info(model_id)
    assert benchmark.expected_kv_bytes(model, "baseline_int16") / 2**20 == int16
    assert benchmark.expected_kv_bytes(model, "turboquant") / 2**20 == turboquant


def test_explicit_head_dim_and_checkpoint_identity(
    tmp_path: Path, benchmark: ModuleType
) -> None:
    identity = importlib.import_module("model_identity")
    config = model_info()["architecture"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text("{}")
    actual = identity.checkpoint_identity(tmp_path, "qwen3_4b")
    assert actual["architecture"]["head_dim"] == 128
    assert actual["config_tokenizer_sha256"]["tokenizer.json"] == benchmark.sha256_file(
        tmp_path / "tokenizer.json"
    )
    with pytest.raises(ValueError, match="architecture"):
        identity.checkpoint_identity(tmp_path, "qwen3_1_7b")
    del config["head_dim"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="architecture"):
        identity.checkpoint_identity(tmp_path, "qwen3_4b")


@pytest.mark.parametrize("parts", [4, 5])
def test_part_count_from_manifest(
    tmp_path: Path, benchmark: ModuleType, parts: int
) -> None:
    prepare_bundles(tmp_path, benchmark, parts)
    result = benchmark.validate_bundles(tmp_path, "qwen3_4b")
    assert result["turboquant"]["num_parts"] == parts


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", model_info("qwen3_1_7b")),
        ("split_manifest_sha256", "other"),
        ("quantize_current_kv", False),
        ("num_parts", 5),
        ("context_buckets", [1024]),
    ],
)
def test_incompatible_bundle_rejected(
    tmp_path: Path, benchmark: ModuleType, field: str, value: Any
) -> None:
    prepare_bundles(tmp_path, benchmark)
    path = tmp_path / "turboquant/convert_report.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Wrong model"):
        benchmark.validate_bundles(tmp_path, "qwen3_4b")


def test_mutated_input_rejected(tmp_path: Path, benchmark: ModuleType) -> None:
    prepare_bundles(tmp_path, benchmark)
    (tmp_path / "assets/rope.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Input asset changed"):
        benchmark.validate_bundles(tmp_path, "qwen3_4b")


def test_conversion_cannot_reuse_another_model(
    tmp_path: Path, benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter = importlib.import_module("convert_parts")
    config = converter.get_profile("k4_v4_scaled")
    split, out = tmp_path / "split", tmp_path / "out"
    split.mkdir()
    out.mkdir()
    manifest = split / "split_manifest.json"
    manifest.write_text(
        json.dumps({"model_id": "qwen3_4b", "model": model_info(), "parts": {}})
    )
    report = {
        "config_hash": config.config_hash(),
        "attention_tile": 256,
        "rotated_attention": True,
        "context_buckets": [1024],
        "native_decoder": None,
        "quantize_current_kv": True,
        "model": model_info("qwen3_1_7b"),
        "split_manifest_sha256": benchmark.sha256_file(manifest),
    }
    path = out / "convert_report.json"
    original = json.dumps(report)
    path.write_text(original)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_parts",
            "--split-dir",
            str(split),
            "--out",
            str(out),
            "--profile",
            "k4_v4_scaled",
            "--no-native-decoder",
        ],
    )
    with pytest.raises(ValueError, match="metadata mismatch: model"):
        converter.main()
    assert path.read_text() == original


def test_wrong_measured_model_rejected(tmp_path: Path, benchmark: ModuleType) -> None:
    prepare_bundles(tmp_path, benchmark)
    metadata = benchmark.validate_bundles(tmp_path, "qwen3_4b")["baseline_int16"]
    assets = json.loads((tmp_path / "assets/assets.json").read_text())
    data = {
        **metadata,
        "kv_store_bytes": 144 * 2**20,
        "assets": {
            "tokens_file": "w0.bin",
            "tokens_sha256": assets["sha256"]["w0.bin"],
            "rope_sha256": assets["sha256"]["rope.bin"],
        },
    }
    benchmark.validate_run(data, metadata, assets, "baseline_int16")
    data["model"] = model_info("qwen3_1_7b")
    with pytest.raises(ValueError, match="Wrong measured model"):
        benchmark.validate_run(data, metadata, assets, "baseline_int16")


def test_audit_requires_all_36_layers(benchmark: ModuleType) -> None:
    metadata = {"model": model_info(), "context_buckets": [1024]}
    audit = {
        "graphs": {
            f"{kind}_ar{ar}_cl1024_2_of_4": {
                "layers": list(range(36)),
                "violations": [],
            }
            for kind, ar in (("prompt", 128), ("token", 1))
        }
    }
    benchmark.validate_audit(
        audit, metadata
    )  # Baseline audit has no top-level 'passed'.
    incomplete = copy.deepcopy(audit)
    incomplete["graphs"]["token_ar1_cl1024_2_of_4"]["layers"].pop()
    with pytest.raises(ValueError, match="missing or duplicating"):
        benchmark.validate_audit(incomplete, metadata)
    with pytest.raises(ValueError, match="audit failed"):
        benchmark.validate_audit({"graphs": {}}, metadata)


def test_serial_build_and_explicit_4b_opt_in(
    tmp_path: Path, benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_bundles(tmp_path, benchmark)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        benchmark, "execute", lambda script, options, log: commands.append(options)
    )
    argv = ["benchmark", "build", "--work-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="model-id"):
        benchmark.main()
    assert not commands
    argv += ["--model-id", "qwen3_4b"]
    benchmark.main()
    assert [c[-1] for c in commands] == ["1", "2", "3", "4"] * 2
    assert [c[c.index("--profile") + 1] for c in commands] == [
        "baseline_int16_kv"
    ] * 4 + ["k4_v4_scaled"] * 4


def test_attempt_log_prevents_repeat(
    tmp_path: Path, benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "attempt.log"
    log.write_text("previous attempt")
    monkeypatch.setattr(
        benchmark.subprocess, "run", lambda *args, **kwargs: pytest.fail("Repeated run")
    )
    with pytest.raises(FileExistsError):
        benchmark.execute("run_device_llm.py", [], log)
    assert log.read_text() == "previous attempt"


def test_performance_is_once_per_group_and_condition(
    tmp_path: Path, benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_bundles(tmp_path, benchmark)
    assets_path = tmp_path / "assets/assets.json"
    assets = json.loads(assets_path.read_text())
    assets.update(
        {"prompt_tokens": 35, "prompt_ids": "prompt.bin", "boundary_prompt": "long.bin"}
    )
    assets_path.write_text(json.dumps(assets))
    runner = tmp_path / "runner"
    runner.write_bytes(b"runner")
    reports = tmp_path / "reports"
    reports.mkdir()
    experiment = {
        "model": model_info(),
        "split_manifest_sha256": benchmark.sha256_file(
            tmp_path / "split/split_manifest.json"
        ),
        "runner_sha256": benchmark.sha256_file(runner),
        "device_prefix": "test",
        "config_hashes": {group: group for group in benchmark.GROUPS},
        "assets_sha256": assets["sha256"],
        "performance_sessions_per_configuration_condition": 1,
    }
    (reports / "experiment.json").write_text(json.dumps(experiment))
    calls: list[list[str]] = []

    def execute(script: str, options: list[str], log: Path) -> None:
        calls.append(options)
        Path(options[options.index("--report") + 1]).write_text("{}")

    monkeypatch.setattr(benchmark, "execute", execute)
    monkeypatch.setattr(benchmark, "validate_run", lambda *args: None)
    monkeypatch.setattr(benchmark, "functional_checks", lambda *args: {"passed": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "performance",
            "--model-id",
            "qwen3_4b",
            "--work-dir",
            str(tmp_path),
            "--runner",
            str(runner),
            "--device-prefix",
            "test",
        ],
    )
    benchmark.main()
    assert len(calls) == 4
    assert [c[c.index("--tokens") + 1] for c in calls] == [
        "prompt.bin",
        "prompt.bin",
        "long.bin",
        "long.bin",
    ]
    assert all(c[c.index("--sessions") + 1] == "1" for c in calls)
    with pytest.raises(FileExistsError, match="repeat a measurement"):
        benchmark.main()
    assert len(calls) == 4


def test_device_model_mismatch_rejected_before_push(
    tmp_path: Path, benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = importlib.import_module("run_device_llm")
    (tmp_path / "assets.json").write_text(
        json.dumps(
            {"context_length": 1024, "sha256": {}, "model": model_info("qwen3_1_7b")}
        )
    )
    calls = []

    def adb(args: Namespace, *cmd: str, **kwargs: Any) -> str:
        calls.append(cmd)
        if cmd[1].startswith("ls "):
            return "runtime_manifest.json part1_of_4.bin"
        if cmd[1] == "cat":
            return json.dumps({"context_length": 1024, "model": model_info()})
        pytest.fail("Unexpected device mutation")

    monkeypatch.setattr(runner, "adb", adb)
    with pytest.raises(ValueError, match="different models"):
        runner.cmd_run(Namespace(assets=tmp_path, name="4b"))
    assert len(calls) == 2
