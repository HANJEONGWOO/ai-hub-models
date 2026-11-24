# SPDX-License-Identifier: BSD-3-Clause
"""NIAH runner/provenance tests without an attached device or model inference."""

import copy
import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("benchmark_niah")


def sample(
    runner: ModuleType,
    group: str = "boundary_structured",
    generated: list[int] | None = None,
) -> tuple[dict, dict, dict]:
    generated = [17, 18, 151645] if generated is None else generated
    case = {
        "case_id": "test_00_cl1024_d000",
        "base_id": "test_00",
        "split": "test",
        "capacity": 1024,
        "input_tokens": 960,
        "tokens_file": "assets/test.bin",
        "tokens_sha256": "a" * 64,
        "expected_answer": "000042",
        "depth": 0,
    }
    native = (
        None if group == "fp16" else {"sha256": "n" * 64, "library": "libNative.so"}
    )
    dep = {
        "group": group,
        "config_hash": "config",
        "native_decoder": native,
        "files": [
            {"remote": f"/bundle/part{i}_of_4.bin", "bytes": 100} for i in range(1, 5)
        ],
        "compiled_io": {
            "host_kv_bytes": 1024 * (114688 if group == "fp16" else 29568),
            "runtime_io_bytes": 2 * 1024 * (114688 if group == "fp16" else 29568)
            + 40818180
            + 258 * 1024,
        },
    }
    steps = [
        {
            "kind": "prefill",
            "ar": 128,
            "graph_context": 1024,
            "cached_before": start,
            "new_tokens": count,
        }
        for start, count in runner.partial_first_chunks(960)
    ]
    steps += [
        {
            "kind": "decode",
            "ar": 1,
            "graph_context": 1024,
            "cached_before": 960 + i,
            "new_tokens": 1,
        }
        for i in range(len(generated) - 1)
    ]
    report = {
        "mode": "generate",
        "backend_api": "2.37.0/htp 5.48.0",
        "context_length": 1024,
        "context_buckets": [1024],
        "prompt_tokens": 960,
        "assets": {"tokens_file": "test.bin", "tokens_sha256": "a" * 64},
        "bundle_name": "bundle",
        "config_hash": "config",
        "native_decoder": native,
        "quantize_current_kv": group != "fp16",
        "kv_store_bytes": 1024 * (114688 if group == "fp16" else 29568),
        "io_buffer_bytes": dep["compiled_io"]["runtime_io_bytes"],
        "load_s": 1.0,
        "ttft_s": 0.5,
        "prefill_s": 0.49,
        "generated": generated,
        "stop_reason": "length" if len(generated) == 64 else "eos",
        "sessions": [
            {
                "session": 0,
                "generated": generated,
                "cached_tokens_at_end": 960 + len(generated) - 1,
            }
        ],
        "steps": steps,
        "op_profiles": [],
        "bins": [
            {"path": item["remote"], "bytes": item["bytes"]} for item in dep["files"]
        ],
    }
    return case, dep, report


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, [(0, 1)]),
        (64, [(0, 64)]),
        (128, [(0, 128)]),
        (129, [(0, 1), (1, 128)]),
        (960, [(0, 64), *[(i, 128) for i in range(64, 960, 128)]]),
    ],
)
def test_partial_first_matches_original_runner(
    runner: ModuleType, count: int, expected: list
) -> None:
    assert runner.partial_first_chunks(count) == expected
    assert expected[-1][0] == max(0, count - 128)


@pytest.mark.parametrize(
    "group", ["fp16", "k4_v4", "boundary_lm", "boundary_structured"]
)
def test_single_generate_session_exact_cache_contract(
    runner: ModuleType, group: str
) -> None:
    case, dep, report = sample(runner, group)
    checked = runner.validate_report(report, case, dep, "bundle")
    assert all(checked["checks"].values())
    assert checked["cached_tokens_at_end"] == 962
    assert checked["decode_iterations"] == 2


def test_eos_first_and_exact_generation_limit(runner: ModuleType) -> None:
    for tokens in ([151645], [10] * 64, [10] * 63 + [151645]):
        case, dep, report = sample(runner, generated=tokens)
        assert runner.validate_report(report, case, dep, "bundle")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_length", 2048),
        ("mode", "score"),
        ("config_hash", "different"),
        ("quantize_current_kv", False),
        ("backend_api", "cpu"),
        ("kv_store_bytes", 10),
        ("io_buffer_bytes", 10),
        ("ttft_s", float("nan")),
        ("stop_reason", "eos_invalid"),
        ("diagnostic_only", True),
    ],
)
def test_invalid_runner_contract_fails(
    runner: ModuleType, field: str, value: object
) -> None:
    case, dep, report = sample(runner)
    report[field] = value
    with pytest.raises(ValueError, match="execution contract"):
        runner.validate_report(report, case, dep, "bundle")


def test_partial_last_prefill_and_cache_overflow_rejected(runner: ModuleType) -> None:
    case, dep, report = sample(runner)
    report["steps"][0]["new_tokens"] = 128
    with pytest.raises(ValueError, match="partial_first_prefill"):
        runner.validate_report(report, case, dep, "bundle")
    case, dep, report = sample(runner)
    report["sessions"][0]["cached_tokens_at_end"] = 1025
    with pytest.raises(ValueError, match="single_session"):
        runner.validate_report(report, case, dep, "bundle")


def test_invalid_tokens_native_or_early_eos_rejected(runner: ModuleType) -> None:
    for mutation in ("tokens", "native", "early_eos"):
        case, dep, report = sample(runner)
        if mutation == "tokens":
            report["assets"]["tokens_sha256"] = "changed"
        elif mutation == "native":
            report["native_decoder"] = None
        else:
            report["generated"][0] = 151645
        with pytest.raises(ValueError, match="execution contract"):
            runner.validate_report(report, case, dep, "bundle")


def test_namespace_preserves_original_nonprofiled_runner(
    runner: ModuleType, tmp_path: Path
) -> None:
    args = runner.namespace(tmp_path / "niah", 4096, "fp16")
    assert args.sessions == 1 and args.n_gen == 64 and args.stop_on_eos
    assert args.mode == "generate" and not args.dump_logits and not args.profile_prefill
    assert args.profile_decode_step == -1 and args.optrace_out is None
    assert args.runner == runner.RUNNER
    assert args.name == "niah_cl4096_fp16"
    with pytest.raises(ValueError, match="Unsafe"):
        runner.valid_name("bad;rm")


def test_remote_hash_completeness_and_idle_checks(runner: ModuleType) -> None:
    device = SimpleNamespace(
        adb=lambda *args: "a" * 64 + "  /a\n" + "b" * 64 + "  /b\n"
    )
    assert runner.remote_hashes(device, None, ["/a", "/b"]) == {
        "/a": "a" * 64,
        "/b": "b" * 64,
    }
    with pytest.raises(RuntimeError, match="Incomplete"):
        runner.remote_hashes(device, None, ["/a", "/b", "/c"])
    device.adb = lambda *args: "__IDLE__"
    runner.assert_idle(device, None)
    device.adb = lambda *args: "123"
    with pytest.raises(RuntimeError, match="active"):
        runner.assert_idle(device, None)
    with pytest.raises(FileExistsError):
        runner.require_new_remote_report(device, None, "/report.json")


def test_success_resume_requires_hash_and_errors_never_retry(
    runner: ModuleType, tmp_path: Path
) -> None:
    case, _, _ = sample(runner)
    (tmp_path / "results").mkdir()
    report = tmp_path / "runner.json"
    report.write_text("{}")
    result = {
        "status": "ok",
        "case_id": case["case_id"],
        "group": "fp16",
        "tokens_sha256": case["tokens_sha256"],
        "deployment_sha256": "deployment",
        "runner_report": "runner.json",
        "runner_report_sha256": runner.digest(report),
    }
    path = runner.result_path(tmp_path, "fp16", case)
    assert runner.check_existing(tmp_path, "fp16", case, "deployment") is None
    path.write_text(json.dumps(result))
    assert runner.check_existing(tmp_path, "fp16", case, "deployment") == result
    with pytest.raises(ValueError, match="different frozen"):
        runner.check_existing(tmp_path, "fp16", case, "wrong")
    report.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed"):
        runner.check_existing(tmp_path, "fp16", case, "deployment")
    result["status"] = "error"
    path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="no automatic retry"):
        runner.check_existing(tmp_path, "fp16", case, "deployment")


def test_cached_asset_push_and_tokenizer_are_scoped(
    runner: ModuleType, tmp_path: Path
) -> None:
    file = tmp_path / "tokens.bin"
    file.write_bytes(b"abc")
    a = {"context_length": 1024, "sha256": {"tokens.bin": runner.digest(file)}}

    def push(*args: Any) -> bool:
        return True

    auto = object()

    def adb(*args: Any, **kwargs: Any) -> str:
        return "OK"

    device = SimpleNamespace(push_if_changed=push, AutoTokenizer=auto, adb=adb)
    args = SimpleNamespace(assets=tmp_path)
    tokenizer = object()
    with runner.efficient_device(
        device, args, tokenizer, a, tmp_path / "commands.jsonl"
    ):
        assert device.AutoTokenizer.from_pretrained("ignored") is tokenizer
        assert not device.push_if_changed(
            args, file, runner.remote_assets(a) + "/tokens.bin"
        )
        assert device.adb(args, "shell", "true") == "OK"
        with pytest.raises(ValueError, match="Unexpected"):
            device.push_if_changed(args, file, "/wrong")
    assert (
        device.push_if_changed is push
        and device.AutoTokenizer is auto
        and device.adb is adb
    )
    log = json.loads((tmp_path / "commands.jsonl").read_text())
    assert log["command"] == ["shell", "true"] and log["stdout"] == "OK"


def test_gates_pilot_only_fp16_and_no_tests_before_probe(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = {"protocol": {"capacities": [1024, 4096]}, "cases": []}
    runner.gates(tmp_path, 1024, "fp16", "pilot", data)
    with pytest.raises(ValueError, match="Pilot must"):
        runner.gates(tmp_path, 1024, "k4_v4", "pilot", data)
    case, _, _ = sample(runner)
    pilots = [dict(case, case_id=f"pilot{i}", split="pilot") for i in range(2)]
    smoke = dict(case, case_id="smoke", split="smoke")
    data["cases"] = [*pilots, smoke]
    with pytest.raises(ValueError, match="pilots must pass"):
        runner.gates(tmp_path, 1024, "fp16", "probe", data)
    (tmp_path / "results").mkdir()
    (tmp_path / "deployments").mkdir()
    (tmp_path / "stage_completion").mkdir()
    (tmp_path / "stage_completion/pilot_cl1024_fp16.json").write_text(
        json.dumps({"pre_post_host_device_hashes_verified": True})
    )
    runner.deployment_path(tmp_path, 1024, "fp16").write_text("{}")
    monkeypatch.setattr(runner, "check_existing", lambda *args: {})
    for pilot in pilots:
        runner.result_path(tmp_path, "fp16", pilot).write_text(
            json.dumps({"status": "ok", "exact_match": True})
        )
    runner.gates(tmp_path, 1024, "fp16", "probe", data)
    with pytest.raises(ValueError, match="operational smoke"):
        runner.gates(tmp_path, 1024, "fp16", "measure", data)
    # Wrong retrieval answer is not an infrastructure failure or a reason to
    # exclude this configuration from formal evaluation.
    runner.result_path(tmp_path, "fp16", smoke).write_text(
        json.dumps({"status": "ok", "exact_match": False})
    )
    (tmp_path / "stage_completion/probe_cl1024_fp16.json").write_text(
        json.dumps({"pre_post_host_device_hashes_verified": True})
    )
    runner.gates(tmp_path, 1024, "fp16", "measure", data)


def test_protocol_does_not_silently_change_context_range(
    runner: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "data").mkdir()
    p = {
        "groups": list(runner.GROUPS),
        "context_capacities": [1024, 4096],
        "max_generated_tokens": 64,
    }
    d = {"protocol": {"capacities": [1024, 4096], "max_new_tokens": 64}}
    (tmp_path / "protocol.json").write_text(json.dumps(p))
    (tmp_path / "data/data_manifest.json").write_text(json.dumps(d))
    assert runner.protocol(tmp_path, 1024) == (p, d)
    with pytest.raises(ValueError, match="not authorized"):
        runner.protocol(tmp_path, 32768)
    altered = copy.deepcopy(d)
    altered["protocol"]["capacities"] = [1024]
    (tmp_path / "data/data_manifest.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="do not match"):
        runner.protocol(tmp_path, 1024)
