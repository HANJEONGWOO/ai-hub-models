# SPDX-License-Identifier: BSD-3-Clause
"""One-shot expanded-swap recovery; fake host/processes and no device access."""

import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"))
    return importlib.import_module("niah_swap_resume")


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


@pytest.fixture
def prepared_fixture(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, str]:
    monkeypatch.setattr(module, "ensure_idle", lambda root: None)
    monkeypatch.setattr(module, "host_state", lambda: {"boot_id": "expanded-boot", "SwapTotal_bytes": 128 * 1024**3})
    monkeypatch.setattr(module.original, "verify_frozen", lambda root: {})
    marker = tmp_path / "previous-recovery.json"
    write(marker, "unchanged serial failure")
    monkeypatch.setattr(module, "prior_failure", lambda root, python: {"previous-recovery.json": module.original.digest(marker)})
    monkeypatch.setattr(module.previous, "allowed_links", lambda root: {})
    complete = "builds/cl16384/shared/part1"
    write(tmp_path / complete / "part.bin", "complete")
    monkeypatch.setattr(module.previous, "failed_scope", lambda root: ([], [{"output": str(root / complete)}]))
    for name in module.TARGETS:
        write(tmp_path / name / "partial.onnx" if "/part" in name and not name.endswith(".log") else tmp_path / name, "failed")
    return tmp_path, complete


def test_prepare_preserves_exact_scope_once(module: ModuleType, prepared_fixture: tuple[Path, str]) -> None:
    root, complete = prepared_fixture
    old = {name: module.inventory(root / name, {}) for name in module.TARGETS}
    archive = module.prepare(root, Path(sys.executable), "재개: expanded swap, same serial 16K")
    intent = module.verify_prepared(root, archive, Path(sys.executable))
    assert intent["workers_before"] == intent["workers_after"] == 1
    assert intent["capacity_unchanged"] == 16384
    assert intent["host"]["SwapTotal_bytes"] == 128 * 1024**3
    for name in module.TARGETS:
        assert not (root / name).exists()
        assert module.inventory(archive / "preserved" / name, {}) == old[name]
    assert (root / complete / "part.bin").read_text() == "complete"
    assert (root / "previous-recovery.json").read_text() == "unchanged serial failure"
    with pytest.raises(FileExistsError, match="already exists"):
        module.prepare(root, Path(sys.executable), "again")


def test_authorization_required(module: ModuleType, prepared_fixture: tuple[Path, str]) -> None:
    root, _ = prepared_fixture
    with pytest.raises(ValueError, match="authorization"):
        module.prepare(root, Path(sys.executable), "  ")
    assert all((root / name).exists() for name in module.TARGETS)
    assert not list(root.glob("recovery_swap_*"))


@pytest.mark.parametrize("field", ["boot_id", "SwapTotal_bytes"])
def test_host_change_refuses_launch(module: ModuleType, prepared_fixture: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    root, _ = prepared_fixture
    archive = module.prepare(root, Path(sys.executable), "approved")
    changed = dict(module.host_state())
    changed[field] = "different" if field == "boot_id" else 127 * 1024**3
    monkeypatch.setattr(module, "host_state", lambda: changed)
    with pytest.raises(ValueError, match="Host boot/swap"):
        module.verify_prepared(root, archive, Path(sys.executable))


@pytest.mark.parametrize("target", ["archive", "complete", "prior", "intent"])
def test_tamper_rejected(module: ModuleType, prepared_fixture: tuple[Path, str], target: str) -> None:
    root, complete = prepared_fixture
    archive = module.prepare(root, Path(sys.executable), "approved")
    path = {
        "archive": archive / "preserved" / module.previous.FAILED_LOGS[0],
        "complete": root / complete / "part.bin",
        "prior": root / "previous-recovery.json",
        "intent": archive / "intent.json",
    }[target]
    write(path, "changed")
    with pytest.raises(ValueError, match="changed|mismatch"):
        module.verify_prepared(root, archive, Path(sys.executable))


def test_partial_archive_is_preserved_and_not_retried(module: ModuleType, prepared_fixture: tuple[Path, str], monkeypatch: pytest.MonkeyPatch) -> None:
    root, _ = prepared_fixture
    real = module.previous.archive_one
    def injected(root: Path, archive: Path, name: str, links: dict, expected: list) -> None:
        if name == module.TARGETS[1]:
            raise OSError("injected")
        real(root, archive, name, links, expected)
    monkeypatch.setattr(module.previous, "archive_one", injected)
    with pytest.raises(OSError, match="injected"):
        module.prepare(root, Path(sys.executable), "approved")
    archive = next(root.glob("recovery_swap_*"))
    assert (archive / "prepare_failed.json").is_file()
    assert not (archive / "prepared.json").exists()
    assert (archive / "preserved" / module.TARGETS[0]).exists()
    assert (root / module.TARGETS[1]).exists()
    with pytest.raises(FileExistsError):
        module.prepare(root, Path(sys.executable), "again")


def test_run_consumes_before_controller_once(module: ModuleType, prepared_fixture: tuple[Path, str], monkeypatch: pytest.MonkeyPatch) -> None:
    root, _ = prepared_fixture
    archive = module.prepare(root, Path(sys.executable), "approved")
    calls = []
    class Controller:
        def __init__(self, root: Path, python: Path, archive: Path) -> None:
            self.run_dir = root / "mock"
        def run(self) -> None:
            assert (archive / "run_consumed.json").exists()
            calls.append("run")
    monkeypatch.setattr(module, "RecoveryController", Controller)
    module.run(root, Path(sys.executable), archive)
    with pytest.raises(FileExistsError):
        module.run(root, Path(sys.executable), archive)
    assert calls == ["run"]


def test_launch_releases_locks_before_child_and_is_one_shot(module: ModuleType, prepared_fixture: tuple[Path, str], monkeypatch: pytest.MonkeyPatch) -> None:
    root, _ = prepared_fixture
    archive = module.prepare(root, Path(sys.executable), "approved")
    calls = []
    def popen(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        with module.locks(root):
            pass
        assert kwargs["start_new_session"] is True
        assert "--detach" not in argv
        calls.append(argv)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    module.launch(root, Path(sys.executable), archive)
    with pytest.raises(FileExistsError):
        module.launch(root, Path(sys.executable), archive)
    assert len(calls) == 1


def test_host_requires_expanded_swap(module: ModuleType, tmp_path: Path) -> None:
    write(tmp_path / "meminfo", "MemTotal: 77594624 kB\nMemAvailable: 70000000 kB\nSwapTotal: 25165824 kB\nSwapFree: 25165824 kB\n")
    write(tmp_path / "sys/kernel/random/boot_id", "boot")
    write(tmp_path / "uptime", "60.2 0")
    with pytest.raises(RuntimeError, match="Expanded swap"):
        module.host_state(tmp_path)
    write(tmp_path / "meminfo", "MemTotal: 77594624 kB\nSwapTotal: 134217728 kB\n")
    assert module.host_state(tmp_path)["SwapTotal_bytes"] == 128 * 1024**3


def test_new_helper_detected_but_own_detach_parent_allowed(module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module.previous, "ensure_idle", lambda root: None)
    argv = ("python", "niah_swap_resume.py", "run", "--root", str(tmp_path))
    process = module.resume.Process(123456, argv)
    monkeypatch.setattr(module.resume, "processes", lambda: [process])
    with pytest.raises(RuntimeError, match="already active"):
        module.ensure_idle(tmp_path)
    process = module.resume.Process(os.getppid(), (*argv, "--detach"))
    module.ensure_idle(tmp_path)


def test_controller_keeps_single_worker_and_original_contract(module: ModuleType, prepared_fixture: tuple[Path, str]) -> None:
    root, _ = prepared_fixture
    archive = module.prepare(root, Path(sys.executable), "approved")
    write(root / "continuation_contract.json", "original workers2 remains")
    controller = module.RecoveryController(root, Path(sys.executable), archive)
    controller.contract()
    assert controller.workers == 1
    assert (root / "continuation_contract.json").read_text() == "original workers2 remains"
    amendment = json.loads((controller.run_dir / "recovery_amendment.json").read_text())
    assert amendment["scientific_contract_unchanged"]
    assert amendment["capacity_unchanged"] == 16384


@pytest.fixture
def provenance_fixture(module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    archive = tmp_path / module.PRIOR_ARCHIVE
    run_dir = tmp_path / "continuation_runs" / module.PRIOR_RUN
    monkeypatch.setattr(module.previous, "verify_prepared", lambda *args: {"authorization": "prior approved"})
    python = Path(sys.executable)
    write(archive / "intent.json", "intent")
    write(archive / "prepared.json", "prepared")
    write(archive / "controller.json", {"run_directory": str(run_dir), "pid": module.PRIOR_PID})
    write(archive / "run_consumed.json", {"pid": module.PRIOR_PID})
    write(archive / "launch.json", {"pid": module.PRIOR_PID, "argv": [str(python), str(Path(module.previous.__file__).resolve()), "run", "--root", str(tmp_path), "--python", str(python), "--archive", str(archive)]})
    build_argv = [str(python), str(module.original.HERE / "niah_resume_build.py"), "resume", "--root", str(tmp_path), "--capacity", "16384", "--group", *module.original.GROUPS, "--workers", "1"]
    write(run_dir / "status.json", {"status": "failed", "stage": "build_cl16384", "pid": module.PRIOR_PID, "error": f"CalledProcessError(1, {build_argv!r})"})
    write(run_dir / "recovery_amendment.json", {
        "workers": 1, "scientific_contract_unchanged": True, "archive": str(archive),
        "archive_intent_sha256": module.original.digest(archive / "intent.json"),
        "archive_prepared_sha256": module.original.digest(archive / "prepared.json"),
        "helper_sha256": module.original.digest(Path(module.previous.__file__)),
        "authorization": "prior approved", "original_contract_sha256": module.previous.ORIGINAL_CONTRACT_SHA256,
    })
    return tmp_path


def test_serial_retry_chain_verified(module: ModuleType, provenance_fixture: Path) -> None:
    assert len(module.prior_failure(provenance_fixture, Path(sys.executable))) == 7


@pytest.mark.parametrize("target", ["workers", "source", "failure"])
def test_wrong_prior_retry_rejected(module: ModuleType, provenance_fixture: Path, target: str) -> None:
    run_dir = provenance_fixture / "continuation_runs" / module.PRIOR_RUN
    path = run_dir / ("status.json" if target == "failure" else "recovery_amendment.json")
    data = json.loads(path.read_text())
    if target == "workers":
        data["workers"] = 2
    elif target == "source":
        data["helper_sha256"] = "changed"
    else:
        data["stage"] = "measure_cl8192"
    write(path, data)
    with pytest.raises(ValueError, match="provenance mismatch"):
        module.prior_failure(provenance_fixture, Path(sys.executable))
