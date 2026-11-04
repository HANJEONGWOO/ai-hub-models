# SPDX-License-Identifier: BSD-3-Clause
"""Continuation scheduling tests: no conversion, inference, or device access."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("niah_continue")


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fake_controller(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Any, list, set]:
    controller = module.Controller(tmp_path, Path(sys.executable))
    events, complete = [], {("pilot", 1024, "fp16")}
    monkeypatch.setattr(module, "verify_frozen", lambda root: {})
    monkeypatch.setattr(controller, "contract", lambda: None)
    monkeypatch.setattr(
        module,
        "stage_complete",
        lambda root, data, stage, capacity, group: (stage, capacity, group) in complete,
    )
    monkeypatch.setattr(
        controller,
        "audit",
        lambda capacity, group: events.append(("audit", capacity, group)),
    )
    monkeypatch.setattr(
        controller,
        "deploy",
        lambda capacity, group: events.append(("deploy", capacity, group)),
    )
    monkeypatch.setattr(
        controller,
        "release",
        lambda capacity, group: events.append(("release", capacity, group)),
    )
    monkeypatch.setattr(controller, "report", lambda: events.append(("report",)))

    def command(label: str, script: str, *options: str) -> None:
        capacity = int(options[options.index("--capacity") + 1])
        group = options[options.index("--group") + 1] if "--group" in options else None
        stage = options[0]
        events.append((stage, capacity, group))
        if stage == "measure":
            assert all(
                ("probe", c, g) in complete
                for c in module.CAPACITIES
                for g in module.GROUPS
            )
        if stage in ("pilot", "probe", "measure"):
            complete.add((stage, capacity, group))

    monkeypatch.setattr(controller, "command", command)
    return controller, events, complete


def test_all_probes_before_main_fixed_order(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, events, _ = fake_controller(module, monkeypatch, tmp_path)
    controller.run()
    probes = [e for e in events if e[0] == "probe"]
    measures = [e for e in events if e[0] == "measure"]
    expected = [(c, g) for c in module.CAPACITIES for g in module.GROUPS]
    assert [e[1:] for e in probes] == expected
    assert [e[1:] for e in measures] == expected
    assert events.index(probes[-1]) < events.index(measures[0])
    assert [e[1] for e in events if e[0] == "resume"] == [4096, 8192, 16384]
    assert not any(e[0] == "pilot" for e in events)
    assert events[-1] == ("report",)
    assert module.read(controller.run_dir / "status.json")["status"] == "complete"


def test_probe_error_stops_before_any_main_or_larger_context(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, events, _ = fake_controller(module, monkeypatch, tmp_path)
    previous = controller.command

    def failing(label: str, script: str, *options: str) -> None:
        if label == "probe_cl4096_fp16":
            raise subprocess.CalledProcessError(1, [script])
        previous(label, script, *options)

    monkeypatch.setattr(controller, "command", failing)
    with pytest.raises(subprocess.CalledProcessError):
        controller.run()
    assert not any(e[0] == "measure" for e in events)
    assert not any(len(e) > 1 and e[1] == 8192 for e in events)
    status = module.read(controller.run_dir / "status.json")
    assert status["status"] == "failed"
    assert "no" in status["policy"].lower() or "without" in status["policy"]


def test_success_exit_without_completion_is_not_accepted(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _, _ = fake_controller(module, monkeypatch, tmp_path)
    monkeypatch.setattr(controller, "command", lambda *args: None)
    with pytest.raises(ValueError, match="omitted verified probe"):
        controller.block("probe", 1024, "fp16")


def test_completed_block_skips_inference_but_releases_our_copy(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, events, complete = fake_controller(module, monkeypatch, tmp_path)
    complete.add(("measure", 1024, "fp16"))
    controller.block("measure", 1024, "fp16")
    assert events == [("release", 1024, "fp16")]


def test_restore_vs_new_deploy_and_release_cli(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller = module.Controller(tmp_path, Path(sys.executable))
    calls = []
    monkeypatch.setattr(controller, "command", lambda *args: calls.append(args))
    controller.deploy(1024, "fp16")
    assert calls[-1][1:3] == ("benchmark_niah.py", "deploy")
    dump(tmp_path / "deployments/cl1024_fp16.json", {})
    controller.deploy(1024, "fp16")
    assert calls[-1][1:3] == ("niah_device_copies.py", "restore")
    controller.release(1024, "fp16")
    assert calls[-1][1:3] == ("niah_device_copies.py", "release")
    assert calls[-1][-1] == "--if-present"


def test_duplicate_process_lock_rejected(module: ModuleType, tmp_path: Path) -> None:
    with (
        module.exclusive(tmp_path),
        pytest.raises(RuntimeError, match="Another"),
        module.exclusive(tmp_path),
    ):
        pass
    with module.exclusive(tmp_path):
        pass


def test_subprocess_failure_has_exclusive_command_log(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller = module.Controller(tmp_path, Path(sys.executable))

    def fail(argv: list[str], **kwargs: Any) -> None:
        assert kwargs["check"] is True
        assert kwargs["cwd"] == module.REPO
        assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
        kwargs["stdout"].write("preserved failure output\n")
        raise subprocess.CalledProcessError(3, argv)

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        controller.command("probe_test", "benchmark_niah.py", "probe")
    assert list(controller.run_dir.glob("*.start.json"))
    assert not list(controller.run_dir.glob("*.complete.json"))
    assert (
        next(controller.run_dir.glob("*.log")).read_text()
        == "preserved failure output\n"
    )


def stage_fixture(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict, Path, Path, list]:
    case = {"case_id": "smoke", "split": "smoke", "capacity": 1024}
    data = {"cases": [case]}
    result_file = tmp_path / "results/fp16_smoke.json"
    completion_file = tmp_path / "stage_completion/probe_cl1024_fp16.json"
    dep_file = tmp_path / "deployments/cl1024_fp16.json"
    dump(tmp_path / "audit.json", {"passed": True})
    dump(tmp_path / "execution_freeze.json", {})
    dump(tmp_path / "bundles/cl1024/fp16/convert_report.json", {})
    dump(tmp_path / "raw.json", {})
    dep = {
        "compiled_audit_file": "audit.json",
        "compiled_audit_sha256": module.digest(tmp_path / "audit.json"),
        "execution_freeze_sha256": module.digest(tmp_path / "execution_freeze.json"),
        "convert_report_sha256": module.digest(
            tmp_path / "bundles/cl1024/fp16/convert_report.json"
        ),
    }
    dump(dep_file, dep)
    completion = {
        "cases": ["smoke"],
        "pre_post_host_device_hashes_verified": True,
        "deployment_sha256": module.digest(dep_file),
    }
    dump(completion_file, completion)
    result = {"status": "ok", "runner_report": "raw.json"}
    dump(result_file, result)
    validated = []
    runtime = SimpleNamespace(
        result_path=lambda *args: result_file,
        deployment_path=lambda *args: dep_file,
        check_existing=lambda *args: result,
        validate_report=lambda *args: validated.append(args),
        name=lambda *args: "bundle",
    )
    monkeypatch.setitem(sys.modules, "benchmark_niah", runtime)
    monkeypatch.setattr(module, "verify_audit", lambda *args: None)
    return data, result_file, completion_file, validated


def test_completed_stage_validates_raw_contract(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data, _, _, validated = stage_fixture(module, monkeypatch, tmp_path)
    assert module.stage_complete(tmp_path, data, "probe", 1024, "fp16")
    assert len(validated) == 1


def test_failed_record_never_retried_without_completion(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data, result, completion, _ = stage_fixture(module, monkeypatch, tmp_path)
    completion.unlink()
    dump(result, {"status": "error"})
    with pytest.raises(ValueError, match="no automatic retry"):
        module.stage_complete(tmp_path, data, "probe", 1024, "fp16")


def test_completion_tamper_fails_closed(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data, _, completion, _ = stage_fixture(module, monkeypatch, tmp_path)
    value = module.read(completion)
    value["deployment_sha256"] = "changed"
    dump(completion, value)
    with pytest.raises(ValueError, match="provenance mismatch"):
        module.stage_complete(tmp_path, data, "probe", 1024, "fp16")


def test_existing_uncertified_report_not_overwritten(
    module: ModuleType, tmp_path: Path
) -> None:
    controller = module.Controller(tmp_path, Path(sys.executable))
    (tmp_path / "report").mkdir()
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        controller.report()


def test_report_once_then_hash_verified_resume(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller = module.Controller(tmp_path, Path(sys.executable))
    dump(tmp_path / "results/example.json", {"status": "ok"})
    calls = []

    def fake_report(*args: Any) -> None:
        calls.append(args)
        dump(tmp_path / "report/summary.json", {"status": "complete"})

    monkeypatch.setattr(controller, "command", fake_report)
    controller.report()
    controller.report()
    assert len(calls) == 1
    dump(tmp_path / "results/example.json", {"status": "error"})
    with pytest.raises(ValueError, match="inputs changed"):
        controller.report()


def test_detach_launch_separate_session_and_exclusive_metadata(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def fake_popen(argv: list[str], **options: Any) -> SimpleNamespace:
        calls.append((argv, options))
        assert options["start_new_session"] is True
        assert options["stdin"] == subprocess.DEVNULL
        assert options["stderr"] == subprocess.STDOUT
        assert "--detach" not in argv
        options["stdout"].write("child launch output\n")
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    first = module.launch(tmp_path, Path(sys.executable), 2)
    second = module.launch(tmp_path, Path(sys.executable), 2)
    assert first["status"] == "launched_not_verified_complete"
    assert first["pid"] == 12345
    assert first["log"] != second["log"]
    assert Path(first["log"]).read_text() == "child launch output\n"
    assert len(list((tmp_path / "continuation_launches").glob("*/launch.json"))) == 2
