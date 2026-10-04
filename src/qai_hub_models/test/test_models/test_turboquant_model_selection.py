# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Explicit model selection, fixed-context mode and single-run benchmark guards."""

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
    layers, hidden, heads = {
        "qwen3_0_6b": (28, 1024, 16),
        "qwen3_1_7b": (28, 2048, 16),
        "qwen3_4b": (36, 2560, 32),
        "qwen3_8b": (36, 4096, 32),
    }[model_id]
    return {
        "model_id": model_id,
        "architecture": {
            "model_type": "qwen3",
            "num_hidden_layers": layers,
            "hidden_size": hidden,
            "num_attention_heads": heads,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
        },
        "config_tokenizer_sha256": {"config.json": "same-config"},
    }


def prepare_bundles(
    work: Path,
    benchmark: ModuleType,
    parts: int = 4,
    model_id: str = "qwen3_4b",
    cl1024_only: bool = False,
) -> None:
    model = model_info(model_id)
    split = {
        "model_id": model_id,
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
            "context_buckets": [1024] if cl1024_only else benchmark.BUCKETS[group],
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
    defaults = benchmark.parser().parse_args(["build", "--work-dir", "unused"])
    assert not defaults.cl1024_only
    small = benchmark.parser().parse_args(
        ["build", "--work-dir", "unused", "--model-id", "qwen3_0_6b", "--cl1024-only"]
    )
    assert small.model_id == "qwen3_0_6b" and small.cl1024_only
    assert (
        benchmark.parser()
        .parse_args(["prepare", "--work-dir", "unused", "--model-id", "qwen3_8b"])
        .model_id
        == "qwen3_8b"
    )
    assert benchmark.context_buckets(True) == {
        "baseline_int16": [1024],
        "turboquant": [1024],
    }
    assert benchmark.performance_conditions(True) == ("long",)
    assert benchmark.performance_conditions() == ("short", "long")


@pytest.mark.parametrize(
    ("model_id", "int16", "turboquant"),
    [
        ("qwen3_0_6b", 112, 28.875),
        ("qwen3_1_7b", 112, 28.875),
        ("qwen3_4b", 144, 37.125),
        ("qwen3_8b", 144, 37.125),
    ],
)
def test_model_specific_kv_memory(
    benchmark: ModuleType, model_id: str, int16: float, turboquant: float
) -> None:
    model = model_info(model_id)
    assert benchmark.expected_kv_bytes(model, "baseline_int16") / 2**20 == int16
    assert benchmark.expected_kv_bytes(model, "turboquant") / 2**20 == turboquant


@pytest.mark.parametrize("model_id", ["qwen3_0_6b", "qwen3_4b"])
def test_explicit_head_dim_and_checkpoint_identity(
    tmp_path: Path, benchmark: ModuleType, model_id: str
) -> None:
    identity = importlib.import_module("model_identity")
    config = model_info(model_id)["architecture"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text("{}")
    actual = identity.checkpoint_identity(tmp_path, model_id)
    assert actual["architecture"]["head_dim"] == 128
    assert actual["config_tokenizer_sha256"]["tokenizer.json"] == benchmark.sha256_file(
        tmp_path / "tokenizer.json"
    )
    with pytest.raises(ValueError, match="architecture"):
        identity.checkpoint_identity(tmp_path, "qwen3_1_7b")
    del config["head_dim"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="architecture"):
        identity.checkpoint_identity(tmp_path, model_id)


def test_0_6b_fixed_context_two_parts(tmp_path: Path, benchmark: ModuleType) -> None:
    prepare_bundles(
        tmp_path, benchmark, parts=2, model_id="qwen3_0_6b", cl1024_only=True
    )
    bundles = benchmark.validate_bundles(tmp_path, "qwen3_0_6b", True)
    assert all(m["num_parts"] == 2 for m in bundles.values())
    with pytest.raises(ValueError, match="Wrong model"):
        benchmark.validate_bundles(tmp_path, "qwen3_0_6b")
    with pytest.raises(ValueError, match="model mismatch"):
        benchmark.validate_bundles(tmp_path, "qwen3_4b", True)


def test_fixed_context_rejects_bucketed_bundle(
    tmp_path: Path, benchmark: ModuleType
) -> None:
    prepare_bundles(tmp_path, benchmark)
    with pytest.raises(ValueError, match="Wrong model"):
        benchmark.validate_bundles(tmp_path, "qwen3_4b", True)


def test_functional_checks_use_fixed_context(
    tmp_path: Path, benchmark: ModuleType
) -> None:
    group = "turboquant"
    reset = {
        "sessions": [{"generated": list(range(8)), "cached_tokens_at_end": 42}] * 2
    }
    eos = {"generated": [151645], "stop_reason": "eos"}
    steps = [{"ar": 128, "cached_before": 0, "new_tokens": 35, "graph_context": 1024}]
    steps += [
        {"ar": 1, "cached_before": 35 + i, "new_tokens": 1, "graph_context": 1024}
        for i in range(599)
    ]
    switches = {
        "generated": list(range(600)),
        "sessions": [{"cached_tokens_at_end": 634}],
        "steps": steps,
    }
    for label, data in [("reset", reset), ("eos", eos), ("switches", switches)]:
        (tmp_path / f"generation_{group}_{label}.json").write_text(json.dumps(data))
    assert all(benchmark.functional_checks(tmp_path, group, 35, [1024]).values())
    assert not benchmark.functional_checks(tmp_path, group, 35)["switches"]
    steps[-1]["graph_context"] = 512
    (tmp_path / f"generation_{group}_switches.json").write_text(json.dumps(switches))
    assert not benchmark.functional_checks(tmp_path, group, 35, [1024])["switches"]


def test_fixed_context_summary_requires_no_short_run(
    tmp_path: Path, benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_bundles(
        tmp_path, benchmark, parts=2, model_id="qwen3_0_6b", cl1024_only=True
    )
    assets_path = tmp_path / "assets/assets.json"
    assets = json.loads(assets_path.read_text())
    assets["prompt_tokens"] = 35
    assets_path.write_text(json.dumps(assets))
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "experiment.json").write_text('{"cl1024_only": true}')
    for group in benchmark.GROUPS:
        filenames = [f"perf_{group}_long_once.json"]
        filenames += [
            f"generation_{group}_{label}.json" for label in ("reset", "eos", "switches")
        ]
        for filename in filenames:
            (reports / filename).write_text('{"device": {"soc_model": "test"}}')
        for i in range(4):
            (reports / f"score_{group}_w{i}.json").write_text(
                json.dumps(
                    {
                        "device": {"soc_model": "test"},
                        "mode": "score",
                        "scored_tokens": 1023,
                        "nll_sum": 1023,
                    }
                )
            )
    calls = []

    def performance(path: Path, prompt: int) -> dict[str, float]:
        calls.append((path.name, prompt))
        return dict.fromkeys(
            [
                "ttft_ms",
                "prefill_tok_per_s",
                "decode_tok_per_s",
                "host_kv_MiB",
                "end_VmRSS_MiB",
            ],
            1.0,
        )

    monkeypatch.setattr(benchmark, "performance", performance)
    monkeypatch.setattr(benchmark, "validate_run", lambda *args: None)
    monkeypatch.setattr(
        benchmark,
        "functional_checks",
        lambda *args: {"reset": True, "eos": True, "switches": True},
    )
    result = benchmark.summarize(
        tmp_path, benchmark.validate_bundles(tmp_path, "qwen3_0_6b", True), True
    )
    assert result["performance_conditions"] == ["long"]
    assert "short" not in result and "short_change_percent" not in result
    assert result["context_buckets"] == {g: [1024] for g in benchmark.GROUPS}
    assert calls == [(f"perf_{g}_long_once.json", 897) for g in benchmark.GROUPS]
    checks = {
        g: {"reset": True, "eos": g == "baseline_int16", "switches": True}
        for g in benchmark.GROUPS
    }
    monkeypatch.setattr(
        benchmark, "functional_checks", lambda root, g, *args: checks[g]
    )
    (reports / "performance_policy.json").write_text(
        json.dumps(
            {
                "allow_eos_failure": True,
                "functional": checks,
                "cl1024_only": True,
                "performance_conditions": ["long"],
            }
        )
    )
    bundles = benchmark.validate_bundles(tmp_path, "qwen3_0_6b", True)
    with pytest.raises(ValueError, match="diagnostic policy"):
        benchmark.summarize(tmp_path, bundles, True)
    diagnostic = benchmark.summarize(tmp_path, bundles, True, True)
    assert diagnostic["long"]["turboquant"]["diagnostic_only"]
    assert not diagnostic["long"]["baseline_int16"]["diagnostic_only"]
    assert not diagnostic["functional"]["turboquant"]["eos"]
    checks["turboquant"]["reset"] = False
    with pytest.raises(ValueError, match="Functional checks failed"):
        benchmark.summarize(tmp_path, bundles, True, True)


@pytest.mark.parametrize("failed", [None, "eos", "reset", "switches", "unknown"])
@pytest.mark.parametrize("allow_eos_failure", [False, True])
def test_diagnostic_mode_never_ignores_cache_failures(
    benchmark: ModuleType, failed: str | None, allow_eos_failure: bool
) -> None:
    checks = {"reset": True, "eos": True, "switches": True}
    if failed is not None:
        checks[failed] = False
    assert benchmark.functional_acceptable(checks, allow_eos_failure) == (
        failed is None or (failed == "eos" and allow_eos_failure)
    )
    assert not benchmark.functional_acceptable({}, allow_eos_failure)
    assert not benchmark.functional_acceptable({"eos": False}, allow_eos_failure)


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


def test_serial_0_6b_build_is_cl1024_only(
    tmp_path: Path, benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_bundles(
        tmp_path, benchmark, parts=2, model_id="qwen3_0_6b", cl1024_only=True
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        benchmark, "execute", lambda script, options, log: commands.append(options)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "build",
            "--work-dir",
            str(tmp_path),
            "--model-id",
            "qwen3_0_6b",
            "--cl1024-only",
        ],
    )
    benchmark.main()
    assert [c[-1] for c in commands] == ["1", "2", "1", "2"]
    assert all(
        c[c.index("--context-buckets") + 1 : c.index("--parts")] == ["1024"]
        for c in commands
    )


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


@pytest.mark.parametrize("cl1024_only", [False, True])
@pytest.mark.parametrize("allow_eos_failure", [False, True])
def test_performance_is_once_per_group_and_condition(
    tmp_path: Path,
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    cl1024_only: bool,
    allow_eos_failure: bool,
) -> None:
    model_id = "qwen3_0_6b" if cl1024_only else "qwen3_4b"
    prepare_bundles(
        tmp_path,
        benchmark,
        parts=2 if cl1024_only else 4,
        model_id=model_id,
        cl1024_only=cl1024_only,
    )
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
        "model": model_info(model_id),
        "split_manifest_sha256": benchmark.sha256_file(
            tmp_path / "split/split_manifest.json"
        ),
        "runner_sha256": benchmark.sha256_file(runner),
        "device_prefix": "test",
        "config_hashes": {group: group for group in benchmark.GROUPS},
        "assets_sha256": assets["sha256"],
        "performance_sessions_per_configuration_condition": 1,
    }
    if cl1024_only:
        experiment["cl1024_only"] = True
    (reports / "experiment.json").write_text(json.dumps(experiment))
    calls: list[list[str]] = []

    def execute(script: str, options: list[str], log: Path) -> None:
        calls.append(options)
        Path(options[options.index("--report") + 1]).write_text("{}")

    monkeypatch.setattr(benchmark, "execute", execute)
    monkeypatch.setattr(benchmark, "validate_run", lambda *args: None)
    monkeypatch.setattr(
        benchmark,
        "functional_checks",
        lambda root, group, *args: {
            "reset": True,
            "eos": not allow_eos_failure or group == "baseline_int16",
            "switches": True,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "performance",
            "--model-id",
            model_id,
            "--work-dir",
            str(tmp_path),
            "--runner",
            str(runner),
            "--device-prefix",
            "test",
            *(["--cl1024-only"] if cl1024_only else []),
        ],
    )
    if allow_eos_failure:
        with pytest.raises(ValueError, match="Functional checks must pass"):
            benchmark.main()
        assert not calls and not (reports / "performance_policy.json").exists()
        sys.argv.append("--allow-eos-failure")
    benchmark.main()
    policy = json.loads((reports / "performance_policy.json").read_text())
    assert policy["allow_eos_failure"] == allow_eos_failure
    assert policy["functional"]["turboquant"]["eos"] == (not allow_eos_failure)
    assert len(calls) == (2 if cl1024_only else 4)
    assert [c[c.index("--tokens") + 1] for c in calls] == (
        ["long.bin", "long.bin"]
        if cl1024_only
        else ["prompt.bin", "prompt.bin", "long.bin", "long.bin"]
    )
    assert all(c[c.index("--sessions") + 1] == "1" for c in calls)
    with pytest.raises(FileExistsError, match="repeat a measurement"):
        benchmark.main()
    assert len(calls) == (2 if cl1024_only else 4)
    if cl1024_only:
        del experiment["cl1024_only"]
        (reports / "experiment.json").write_text(json.dumps(experiment))
        with pytest.raises(ValueError, match="Experiment identity"):
            benchmark.main()


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
