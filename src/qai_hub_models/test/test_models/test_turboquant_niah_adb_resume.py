# SPDX-License-Identifier: BSD-3-Clause
"""ADB recovery is explicit, exact-scope and never repeats successful runs."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"))
    return importlib.import_module("niah_adb_resume")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


@pytest.fixture
def prepared_fixture(module, monkeypatch, tmp_path):
    monkeypatch.setattr(module, "ensure_idle", lambda root: None)
    marker = tmp_path / "prior_failure.json"
    write(marker, "preserved old failure")
    monkeypatch.setattr(module, "prior_provenance", lambda root, python: {"prior_failure.json": module.original.digest(marker)})
    monkeypatch.setattr(module.original, "verify_frozen", lambda root: {})
    monkeypatch.setattr(module, "verify_gates", lambda root: None)
    keep = {}
    for index in range(241):
        name = f"successes/{index}.json"
        write(tmp_path / name, {"status": "ok", "index": index})
        keep[name] = module.recovery.inventory(tmp_path / name, {})
    monkeypatch.setattr(module, "successful_inventory", lambda root: keep)
    write(tmp_path / module.TARGETS[0], {"status": "error"})
    write(tmp_path / module.TARGETS[1] / "commands.jsonl", "failed before runner")
    return tmp_path, keep


def test_prepare_preserves_241_and_archives_only_exact_failure_once(module, prepared_fixture):
    root, keep = prepared_fixture
    before = {name: module.recovery.inventory(root / name, {}) for name in module.TARGETS}
    archive = module.prepare(root, Path(sys.executable), "재개")
    intent = module.verify_prepared(root, archive, Path(sys.executable))
    assert len(intent["preserved"]) == 241
    assert intent["capacities"] == [1024, 2048, 4096, 8192]
    assert intent["successful_main_preserved"] == 241 and intent["remaining_main"] == 559
    for name in module.TARGETS:
        assert not (root / name).exists()
        assert module.recovery.inventory(archive / "preserved" / name, {}) == before[name]
    for name, expected in keep.items():
        assert module.recovery.inventory(root / name, {}) == expected
    with pytest.raises(FileExistsError):
        module.prepare(root, Path(sys.executable), "again")


def test_authorization_required(module, prepared_fixture):
    root, _ = prepared_fixture
    with pytest.raises(ValueError, match="authorization"):
        module.prepare(root, Path(sys.executable), " ")
    assert all((root / name).exists() for name in module.TARGETS)


@pytest.mark.parametrize("change", ["success", "archive", "prior", "intent"])
def test_tamper_blocks_resume(module, prepared_fixture, change):
    root, keep = prepared_fixture
    archive = module.prepare(root, Path(sys.executable), "approved")
    path = {"success": root / next(iter(keep)), "archive": archive / "preserved" / module.TARGETS[0],
            "prior": root / "prior_failure.json", "intent": archive / "intent.json"}[change]
    write(path, "changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        module.verify_prepared(root, archive, Path(sys.executable))


def test_partial_archive_kept_without_retry(module, prepared_fixture, monkeypatch):
    root, _ = prepared_fixture
    real = module.recovery.archive_one
    def move(root, archive, name, links, expected):
        if name == module.TARGETS[1]:
            raise OSError("injected failure")
        real(root, archive, name, links, expected)
    monkeypatch.setattr(module.recovery, "archive_one", move)
    with pytest.raises(OSError):
        module.prepare(root, Path(sys.executable), "approved")
    archive = next(root.glob("recovery_adb_*"))
    assert (archive / "prepare_failed.json").exists()
    assert (archive / "preserved" / module.TARGETS[0]).exists()
    assert (root / module.TARGETS[1]).exists()
    assert not (archive / "prepared.json").exists()


def test_run_one_shot_consumed_before_controller(module, prepared_fixture, monkeypatch):
    root, _ = prepared_fixture
    archive = module.prepare(root, Path(sys.executable), "approved")
    calls = []
    class Controller:
        def __init__(self, root, python, archive):
            self.run_dir = root / "mock"
        def run(self):
            assert (archive / "run_consumed.json").exists()
            calls.append("run")
    monkeypatch.setattr(module, "RecoveryController", Controller)
    module.run(root, Path(sys.executable), archive)
    with pytest.raises(FileExistsError):
        module.run(root, Path(sys.executable), archive)
    assert calls == ["run"]


def test_detach_releases_locks_before_child_and_is_one_shot(module, prepared_fixture, monkeypatch):
    root, _ = prepared_fixture
    archive = module.prepare(root, Path(sys.executable), "approved")
    calls = []
    def popen(argv, **kwargs):
        with module.recovery.locks(root):
            pass
        assert kwargs["start_new_session"] is True
        assert "--detach" not in argv
        calls.append(argv)
        return SimpleNamespace(pid=321)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    module.launch(root, Path(sys.executable), archive)
    with pytest.raises(FileExistsError):
        module.launch(root, Path(sys.executable), archive)
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", [None, "runner", "extra_command", "wrong_command", "wrong_case", "stdout"])
def test_only_pre_runner_failure_is_eligible(module, tmp_path, mutation):
    supplement = tmp_path / module.scoped.SUPPLEMENT_NAME
    attempt = supplement / "attempts" / module.LABEL
    error = "adb.exe: no devices/emulators found"
    result = {"status": "error", "case_id": module.CASE_ID, "group": "fp16", "capacity": 2048,
              "split": "test", "failure_class": "infrastructure_or_contract_failure_not_model_wrong_answer",
              "error": error, "deployment_sha256": "hash"}
    command = {"command": ["shell", f"ls /data/local/tmp/qaihm_turboquant/llm/bundles/{supplement.name}_cl2048_fp16"], "error": error}
    write(attempt / "start.json", {"stage": "measure", "case_id": module.CASE_ID, "group": "fp16", "deployment_sha256": "hash",
                                  "settings": {"n_gen": 64, "sessions": 1, "stop_on_eos": True,
                                               "tokens": f"{module.CASE_ID}.bin", "bundle": f"{supplement.name}_cl2048_fp16"}})
    write(attempt / "host_stdout.log", "")
    lines = [command]
    if mutation == "runner":
        write(attempt / "runner.json", {})
    elif mutation == "extra_command":
        lines.append(command)
    elif mutation == "wrong_command":
        command["command"] = ["shell", "qnn-llm-runner"]
    elif mutation == "wrong_case":
        result["case_id"] = "another"
    elif mutation == "stdout":
        write(attempt / "host_stdout.log", "runner started")
    write(attempt / "commands.jsonl", "\n".join(json.dumps(v) for v in lines))
    if mutation is None:
        module.validate_pre_runner_failure(result, attempt, supplement)
    else:
        with pytest.raises(ValueError, match="pre-runner"):
            module.validate_pre_runner_failure(result, attempt, supplement)


@pytest.mark.parametrize("script,stage", [("niah_2k_supplement.py", "build"), ("niah_2k_supplement.py", "audit"),
                                         ("benchmark_niah.py", "assets"), ("benchmark_niah.py", "pilot"),
                                         ("benchmark_niah.py", "probe"), ("niah_build.py", "build")])
def test_command_guard_forbids_rebuild_or_gate_reruns(module, tmp_path, script, stage):
    controller = module.RecoveryController(tmp_path, Path(sys.executable), tmp_path / "archive")
    with pytest.raises(ValueError, match="forbids rebuilds"):
        controller.command("bad", script, stage, "--root", str(tmp_path), "--capacity", "1024")


def test_run_checks_all_gates_then_fixed_measure_order_without_builds(module, tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    write(archive / "intent.json", {})
    write(archive / "prepared.json", {})
    write(tmp_path / module.scoped.SUPPLEMENT_NAME / "data/data_manifest.json", {})
    controller = module.RecoveryController(tmp_path, Path(sys.executable), archive)
    events = []
    monkeypatch.setattr(module.original, "verify_frozen", lambda root: {})
    monkeypatch.setattr(controller, "contract", lambda: events.append("contract"))
    monkeypatch.setattr(controller, "all_probes", lambda: events.append("all_16_gates"))
    monkeypatch.setattr(controller, "block", lambda stage, capacity, group: events.append((stage, capacity, group)))
    monkeypatch.setattr(controller, "command", lambda *a: pytest.fail("No build/probe commands allowed"))
    def report():
        events.append("report")
        write(tmp_path / "continuation_report_8k.json", {})
    monkeypatch.setattr(controller, "report", report)
    controller.run()
    assert events == ["contract", "all_16_gates", *[("measure", c, g) for c in module.scoped.CAPACITIES for g in module.scoped.GROUPS], "report"]
    assert module.original.read(controller.run_dir / "status.json")["status"] == "complete"


def test_completed_stage_is_skipped(module, tmp_path, monkeypatch):
    controller = module.RecoveryController(tmp_path, Path(sys.executable), tmp_path / "archive")
    monkeypatch.setattr(module.original, "stage_complete", lambda *a: True)
    monkeypatch.setattr(controller, "deploy", lambda *a: pytest.fail("Successful stage must not be redeployed"))
    monkeypatch.setattr(controller, "command", lambda *a: pytest.fail("Successful stage must not rerun"))
    releases = []
    monkeypatch.setattr(controller, "release", lambda *a: releases.append(a))
    controller.block("measure", 1024, "fp16")
    assert releases == [(1024, "fp16")]


def test_missing_gate_stops_before_any_measure(module, tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    write(tmp_path / module.scoped.SUPPLEMENT_NAME / "data/data_manifest.json", {})
    controller = module.RecoveryController(tmp_path, Path(sys.executable), archive)
    monkeypatch.setattr(module.original, "verify_frozen", lambda root: {})
    monkeypatch.setattr(controller, "contract", lambda: None)
    def fail():
        raise ValueError("missing gate")
    monkeypatch.setattr(controller, "all_probes", fail)
    monkeypatch.setattr(controller, "block", lambda *a: pytest.fail("No main inference before gates"))
    with pytest.raises(ValueError, match="missing gate"):
        controller.run()
    assert module.original.read(controller.run_dir / "status.json")["status"] == "failed"
