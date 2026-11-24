# SPDX-License-Identifier: BSD-3-Clause
"""Health watcher tests use fake /proc and never launch experiment processes."""

import base64
import importlib
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("niah_health")


def fake_process(
    proc: Path,
    root: Path,
    pid: int = 42,
    ppid: int = 1,
    ticks: int = 100,
    start: int = 200,
    state: str = "S",
    script: str = "niah_oom_resume.py",
) -> None:
    directory = proc / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 22
    fields[0], fields[1], fields[11], fields[12] = state, str(ppid), str(ticks), "10"
    fields[19], fields[21] = str(start), "20"
    (directory / "stat").write_text(f"{pid} (python (worker)) " + " ".join(fields))
    (directory / "cmdline").write_bytes(
        f"python\0/repo/scripts/{script}\0--root\0{root}\0".encode()
    )


@pytest.fixture
def environment(tmp_path: Path) -> tuple[Path, Path, Path]:
    root, proc = tmp_path / "experiment", tmp_path / "proc"
    proc.mkdir()
    (proc / "stat").write_text("cpu 0 0 0\nbtime 100\n")
    (proc / "meminfo").write_text(
        "MemTotal: 1000 kB\nMemAvailable: 200 kB\nSwapTotal: 100 kB\nSwapFree: 50 kB\n"
    )
    status = root / "continuation_runs" / "new_42" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps({"pid": 42, "status": "running", "stage": "build_cl16384"})
    )
    fake_process(proc, root)
    return root, proc, status


def binding(module: ModuleType, environment: tuple[Path, Path, Path]) -> dict:
    root, proc, _ = environment
    return module.bind(root, 42, proc=proc)


def test_proc_parser_with_parentheses_and_memory(
    module: ModuleType, environment: tuple
) -> None:
    _, proc, _ = environment
    value = module.process(42, proc)
    assert value["cpu_ticks"] == 110
    assert value["start_ticks"] == 200
    assert value["rss_bytes"] == 20 * os.sysconf("SC_PAGE_SIZE")
    assert module.memory(proc)["SwapFree_bytes"] == 50 * 1024
    assert module.process(999, proc) is None


def test_tree_excludes_unrelated_process(
    module: ModuleType, environment: tuple
) -> None:
    root, proc, _ = environment
    fake_process(proc, root, pid=43, ppid=42)
    fake_process(proc, root, pid=44, ppid=43)
    fake_process(proc, root, pid=45, ppid=1)
    assert [p["pid"] for p in module.process_tree(42, proc)] == [42, 43, 44]


def test_bind_ignores_historical_failed_controller(
    module: ModuleType, environment: tuple
) -> None:
    root, proc, status = environment
    previous = root / "continuation_runs" / "old_7" / "status.json"
    previous.parent.mkdir()
    previous.write_text(json.dumps({"pid": 7, "status": "failed"}))
    assert module.bind(root, 42, proc=proc)["status_file"] == str(status)


@pytest.mark.parametrize(
    "script",
    [
        "niah_continue.py",
        "niah_oom_resume.py",
        "niah_swap_resume.py",
        "niah_8k_continue.py",
        "niah_adb_resume.py",
    ],
)
def test_bind_supported_recovery_controllers(
    module: ModuleType, environment: tuple, script: str
) -> None:
    root, proc, status = environment
    fake_process(proc, root, script=script)
    assert module.bind(root, 42, proc=proc)["status_file"] == str(status)


@pytest.mark.parametrize("script", ["other.py", "qnn-converter"])
def test_bind_rejects_wrong_program(
    module: ModuleType, environment: tuple, script: str
) -> None:
    root, proc, _ = environment
    fake_process(proc, root, script=script)
    with pytest.raises(ValueError, match="approved"):
        module.bind(root, 42, proc=proc)


def test_bind_rejects_other_root_and_reused_pid(
    module: ModuleType, environment: tuple
) -> None:
    root, proc, status = environment
    fake_process(proc, root / "other")
    with pytest.raises(ValueError, match="approved"):
        module.bind(root, 42, proc=proc)
    fake_process(proc, root)
    os.utime(status, (1, 1))
    with pytest.raises(ValueError, match="predates"):
        module.bind(root, 42, proc=proc)


def test_static_stage_is_not_failure(module: ModuleType, environment: tuple) -> None:
    _, proc, _ = environment
    result = module.sample(
        binding(module, environment), proc=proc, observation_seconds=0
    )
    assert result["status"] == "needs_review"
    assert result["controller_identity_verified"]
    assert "not_a_failure" in result["reason"]
    assert result["status"] not in module.TERMINAL


def test_cpu_motion_counts_as_running(
    module: ModuleType, environment: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, proc, _ = environment
    frozen = binding(module, environment)
    monkeypatch.setattr(
        module.time, "sleep", lambda _: fake_process(proc, root, ticks=120)
    )
    result = module.sample(frozen, proc=proc)
    assert result["status"] == "running"
    assert result["observed_cpu_ticks_delta"] == 20


def test_recent_log_counts_as_activity(module: ModuleType, environment: tuple) -> None:
    _, proc, status = environment
    (status.parent / "convert.log").write_text("converting")
    result = module.sample(
        binding(module, environment), proc=proc, observation_seconds=0
    )
    assert result["status"] == "running"
    assert result["log_activity"]["newest_log_size_bytes"] == 10


def test_interval_cpu_motion_uses_same_incarnation(
    module: ModuleType, environment: tuple
) -> None:
    root, proc, _ = environment
    frozen = binding(module, environment)
    previous = module.sample(frozen, proc=proc, observation_seconds=0)
    fake_process(proc, root, ticks=200)
    result = module.sample(frozen, previous, proc=proc, observation_seconds=0)
    assert result["status"] == "running"
    assert result["interval_cpu_ticks_delta_surviving_processes"] == 100
    fake_process(proc, root, ticks=1000, start=999)
    result = module.sample(frozen, previous, proc=proc, observation_seconds=0)
    assert result["status"] == "unexpected_stopped"
    assert result["interval_cpu_ticks_delta_surviving_processes"] == 0


@pytest.mark.parametrize("state", ["complete", "failed"])
def test_terminal_status_recorded(
    module: ModuleType, environment: tuple, state: str
) -> None:
    _, proc, status = environment
    frozen = binding(module, environment)
    status.write_text(
        json.dumps({"pid": 42, "status": state, "stage": "build_cl16384"})
    )
    result = module.sample(frozen, proc=proc, observation_seconds=0)
    assert result["status"] == state
    assert result["reason"] == "controller_reported_terminal_state"


def test_status_read_failure_is_review_not_failure(
    module: ModuleType, environment: tuple
) -> None:
    _, proc, status = environment
    frozen = binding(module, environment)
    status.write_text("{")
    result = module.sample(frozen, proc=proc, observation_seconds=0)
    assert result["status"] == "needs_review"
    assert result["status_read_error"]


def test_zombie_is_unexpected_stop(module: ModuleType, environment: tuple) -> None:
    root, proc, _ = environment
    frozen = binding(module, environment)
    fake_process(proc, root, state="Z")
    assert (
        module.sample(frozen, proc=proc, observation_seconds=0)["status"]
        == "unexpected_stopped"
    )


def test_record_preserves_all_samples(
    module: ModuleType, environment: tuple, tmp_path: Path
) -> None:
    _, proc, _ = environment
    result = module.sample(
        binding(module, environment), proc=proc, observation_seconds=0
    )
    output = tmp_path / "health"
    output.mkdir()
    module.record(output, result)
    second = {**result, "status": "failed"}
    module.record(output, second)
    assert len((output / "samples.jsonl").read_text().splitlines()) == 2
    assert json.loads((output / "latest.json").read_text())["status"] == "failed"
    assert not (output / "latest.pending.json").exists()


def test_watch_stops_after_terminal_record(
    module: ModuleType, environment: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, proc, _ = environment
    frozen = binding(module, environment)
    result = module.sample(frozen, proc=proc, observation_seconds=0)
    result["status"] = "failed"
    monkeypatch.setattr(module, "bind", lambda *args: frozen)
    monkeypatch.setattr(module, "sample", lambda *args: result)
    monkeypatch.setattr(
        module.time, "sleep", lambda _: pytest.fail("Terminal monitor must not sleep")
    )
    module.watch(root, 42, 7200)
    directories = list((root / "health_runs").iterdir())
    assert len(directories) == 1
    assert (
        json.loads((directories[0] / "latest.json").read_text())["status"] == "failed"
    )


def test_detach_uses_separate_session_and_records_pid(
    module: ModuleType, environment: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _ = environment
    frozen = binding(module, environment)
    monkeypatch.setattr(module, "bind", lambda *args: frozen)
    calls = []

    def launch(argv: list, **kwargs: object) -> SimpleNamespace:
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    result = module.launch(root, 42, 7200)
    assert result["monitor_pid"] == 1234
    assert calls[0][1]["start_new_session"] is True
    assert "--detach" not in calls[0][0]
    assert "no_chat" in result["notification"]
    assert (Path(result["directory"]) / "launch.json").exists()


def test_notification_uses_encoded_safe_payload_and_does_not_claim_seen(
    module: ModuleType, environment: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, proc, _ = environment
    value = module.sample(
        binding(module, environment), proc=proc, observation_seconds=0
    )
    value["stage"] = "build_$(danger)'<xml>&"
    calls = []

    def run(argv: list, **kwargs: object) -> SimpleNamespace:
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout='{"submitted":true,"method":"WinRTToast","visually_confirmed":false}',
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    result = module.windows_notification(value)
    assert result["submitted"] is True
    assert result["visually_confirmed"] is False
    assert result["status"] == "submitted_visibility_unconfirmed"
    assert calls[0][1]["timeout"] == 30
    script = base64.b64decode(calls[0][0][-1]).decode("utf-16le")
    assert "$(danger)" not in script
    assert "Set-ItemProperty" not in script
    assert "Register-" not in script
    assert "Start-Sleep -Seconds 5" in script
    assert "-NoProfile" in calls[0][0]


@pytest.mark.parametrize(
    "failure", ["exception", "returncode", "invalid_json", "array"]
)
def test_notification_errors_return_without_raising(
    module: ModuleType,
    environment: tuple,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    _, proc, _ = environment
    value = module.sample(
        binding(module, environment), proc=proc, observation_seconds=0
    )

    def run(*args: object, **kwargs: object) -> SimpleNamespace:
        if failure == "exception":
            raise FileNotFoundError("PowerShell unavailable")
        return SimpleNamespace(
            returncode=1 if failure == "returncode" else 0,
            stdout="[]" if failure == "array" else "not json",
            stderr="notify error",
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    result = module.windows_notification(value)
    assert result["submitted"] is False
    assert result["status"] in {
        "notification_failed_monitor_continues",
        "rejected_or_unconfirmed",
    }


def test_watch_records_failed_notification_and_terminal_result(
    module: ModuleType, environment: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, proc, _ = environment
    frozen = binding(module, environment)
    value = module.sample(frozen, proc=proc, observation_seconds=0)
    value["status"] = "complete"
    monkeypatch.setattr(module, "bind", lambda *args: frozen)
    monkeypatch.setattr(module, "sample", lambda *args: value)
    monkeypatch.setattr(
        module,
        "windows_notification",
        lambda record: {"submitted": False, "status": "api_error"},
    )
    module.watch(root, 42, 7200, windows_notify=True)
    path = next((root / "health_runs").glob("*/latest.json"))
    result = json.loads(path.read_text())
    assert result["status"] == "complete"
    assert not result["notification"]["submitted"]
