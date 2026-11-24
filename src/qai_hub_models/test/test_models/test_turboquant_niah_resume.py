# SPDX-License-Identifier: BSD-3-Clause
"""Disconnected NIAH build recovery without executing a converter."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def helper(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("niah_resume_build")


def converter(helper: ModuleType, root: Path, group: str = "fp16", number: int = 2):
    output = root / "builds/cl4096" / group / f"part{number}"
    effective = "fp16" if group == "shared" else group
    return helper.Process(123, tuple([
        "/repo/venv/bin/python", "/repo/scripts/convert_parts.py",
        *helper.build.convert_options(root, 4096, effective, number, output),
    ]))


def test_matching_converter_waits_without_relaunch(helper: ModuleType, tmp_path: Path):
    p = converter(helper, tmp_path)
    found = helper.build_processes(tmp_path, [p])
    assert found == [{
        "pid": 123, "kind": "converter",
        "output": str(tmp_path / "builds/cl4096/fp16/part2"),
    }]


def test_shared_part_uses_fp16_configuration(helper: ModuleType, tmp_path: Path):
    assert helper.build_processes(tmp_path, [converter(helper, tmp_path, "shared", 1)])


def test_mismatched_converter_fails_closed(helper: ModuleType, tmp_path: Path):
    p = converter(helper, tmp_path)
    changed = helper.Process(p.pid, (*p.argv, "--skip-context"))
    with pytest.raises(ValueError, match="configuration mismatch"):
        helper.build_processes(tmp_path, [changed])


def test_wrong_target_fails_closed(helper: ModuleType, tmp_path: Path):
    with pytest.raises(ValueError, match="Unexpected converter target"):
        helper.build_processes(tmp_path, [converter(helper, tmp_path, "fp16", 1)])


def test_orphan_qnn_child_blocks_restart(helper: ModuleType, tmp_path: Path):
    p = helper.Process(234, (
        "/sdk/qnn-context-binary-generator", "--output_dir",
        str(tmp_path / "builds/cl4096/fp16/part2"),
    ))
    assert helper.build_processes(tmp_path, [p])[0]["kind"] == "qnn_child"


def test_unrelated_converter_and_shell_are_ignored(helper: ModuleType, tmp_path: Path):
    unrelated = converter(helper, tmp_path.parent / "other_experiment")
    shell = helper.Process(456, ("/bin/bash", "-c", "python convert_parts.py --out " + str(tmp_path / "builds")))
    assert helper.build_processes(tmp_path, [unrelated, shell]) == []


@pytest.mark.parametrize("script,stage", [("niah_build.py", "build"), ("niah_resume_build.py", "resume")])
def test_other_orchestrator_refused(helper: ModuleType, tmp_path: Path, script: str, stage: str):
    p = helper.Process(345, ("/repo/python", "/repo/" + script, stage, "--root", str(tmp_path)))
    with pytest.raises(RuntimeError, match="Another build orchestrator"):
        helper.build_processes(tmp_path, [p])


def test_completed_and_missing_parts_are_safe(helper: ModuleType):
    helper.ready_parts([
        {"state": "complete", "log_exists": True, "output": "complete"},
        {"state": "missing", "log_exists": False, "output": "missing"},
    ])


def test_inactive_partial_and_old_attempt_are_preserved(helper: ModuleType):
    for state, log, error in [("partial", True, "Inactive partial"), ("missing", True, "Previous attempt")]:
        with pytest.raises(FileExistsError, match=error):
            helper.ready_parts([{"state": state, "log_exists": log, "output": "kept"}])


def test_inventory_deduplicates_shared_part(helper: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(helper.build, "completed_part", lambda *args: False)
    partial = tmp_path / "builds/cl4096/fp16/part2"
    partial.mkdir(parents=True)
    states = helper.part_states(tmp_path, 4096, ["fp16", "k4_v4"])
    assert len(states) == 7
    assert sum("/shared/" in s["output"] for s in states) == 1
    assert states[1]["state"] == "partial"
    assert states[0]["state"] == "missing"


def test_resume_waits_then_calls_unchanged_build(helper: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    values = iter([
        {"utc": "first", "active": [{"pid": 123}], "parts": []},
        {"utc": "ready", "active": [], "parts": []},
        {"utc": "done", "active": [], "parts": []},
    ])
    calls = []
    monkeypatch.setattr(helper, "snapshot", lambda *a: next(values))
    monkeypatch.setattr(helper.build, "check", lambda *a: calls.append("check"))
    monkeypatch.setattr(helper.build, "build", lambda *a: calls.append("build"))
    monkeypatch.setattr(helper.time, "sleep", lambda delay: calls.append(("wait", delay)))
    helper.resume(tmp_path, 4096, ["fp16"], 2, 30)
    assert calls == ["check", ("wait", 30), "build"]
    run = next((tmp_path / "recovery").iterdir())
    assert {p.name for p in run.iterdir()} == {"start.json", "ready.json", "complete.json"}


def test_resume_never_retries_inactive_partial(helper: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state = {"utc": "bad", "active": [], "parts": [{"state": "partial", "log_exists": True, "output": "preserved"}]}
    calls = []
    monkeypatch.setattr(helper, "snapshot", lambda *a: state)
    monkeypatch.setattr(helper.build, "check", lambda *a: None)
    monkeypatch.setattr(helper.build, "build", lambda *a: calls.append("unsafe"))
    with pytest.raises(FileExistsError, match="Inactive partial"):
        helper.resume(tmp_path, 4096, ["fp16"], 2, 30)
    assert calls == []
    assert next((tmp_path / "recovery").glob("*/start.json")).is_file()


def test_sandbox_process_view_is_rejected(helper: ModuleType, tmp_path: Path):
    (tmp_path / "1").mkdir()
    (tmp_path / "1/cmdline").write_bytes(b"codex-linux-sandbox\0--argument")
    with pytest.raises(RuntimeError, match="Host processes are hidden"):
        helper.processes(tmp_path)


def test_transient_report_write_is_not_treated_as_complete(helper: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import json

    output = tmp_path / "builds/cl4096/shared/part1"
    output.mkdir(parents=True)

    def incomplete(*args):
        raise json.JSONDecodeError("still being written", "", 0)

    monkeypatch.setattr(helper.build, "completed_part", incomplete)
    assert helper.part_states(tmp_path, 4096, [])[0]["state"] == "partial"
