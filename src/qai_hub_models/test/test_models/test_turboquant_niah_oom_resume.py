# SPDX-License-Identifier: BSD-3-Clause
"""Explicit OOM recovery tests; no real model/device/process execution."""

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
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("niah_oom_resume")


def write(path: Path, text: str = "preserved") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def prepared_fixture(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, str]:
    monkeypatch.setattr(module, "ensure_idle", lambda root: None)
    monkeypatch.setattr(module, "validate_original", lambda root, python: {})
    monkeypatch.setattr(module.original, "verify_frozen", lambda root: {})
    failure = "continuation_runs/old/status.json"
    write(tmp_path / failure, '{"status":"failed"}')
    monkeypatch.setattr(
        module,
        "prior_failure",
        lambda root: {
            "path": failure,
            "sha256": module.original.digest(root / failure),
        },
    )
    monkeypatch.setattr(module, "allowed_links", lambda root: {})
    complete = "builds/cl16384/shared/part1"
    write(tmp_path / complete / "part1_of_4.bin", "complete binary")
    monkeypatch.setattr(
        module, "failed_scope", lambda root: ([], [{"output": str(root / complete)}])
    )
    for path in module.FAILED_PARTS:
        write(tmp_path / path / "incomplete.onnx")
    for path in module.FAILED_LOGS:
        write(tmp_path / path, "failed converter log")
    return tmp_path, complete


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../../outside"])
def test_path_escape_rejected(module: ModuleType, tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        module.safe_path(tmp_path, name, exists=False)


def test_symlink_parent_rejected(module: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "actual").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "actual", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked"):
        module.safe_path(tmp_path, "alias/child", exists=False)


def test_inventory_preserves_known_external_weight_link(
    module: ModuleType, tmp_path: Path
) -> None:
    target = tmp_path / "weight.data"
    write(target, "weight")
    directory = tmp_path / "part"
    directory.mkdir()
    (directory / "weight.data").symlink_to(target)
    values = module.inventory(directory, {"weight.data": str(target)})
    entry = next(v for v in values if v["kind"] == "symlink")
    assert entry["link"] == str(target)
    assert entry["sha256"] == module.original.digest(target)
    with pytest.raises(ValueError, match="Unapproved"):
        module.inventory(directory, {})


def test_prepare_exact_archive_and_one_shot(
    module: ModuleType, prepared_fixture: tuple[Path, str]
) -> None:
    root, complete = prepared_fixture
    old = {p: module.inventory(root / p, {}) for p in module.TARGETS}
    archive = module.prepare(root, Path(sys.executable), "User: 재개; workers 2→1")
    for name in module.TARGETS:
        assert not (root / name).exists()
        assert module.inventory(archive / "preserved" / name, {}) == old[name]
    assert (root / complete / "part1_of_4.bin").read_text() == "complete binary"
    intent = module.verify_prepared(root, archive, Path(sys.executable))
    assert intent["workers_before"] == 2
    assert intent["workers_after"] == 1
    assert intent["capacity_unchanged"] == 16384
    with pytest.raises(FileExistsError, match="already exists"):
        module.prepare(root, Path(sys.executable), "retry")


def test_no_authorization_no_changes(
    module: ModuleType, prepared_fixture: tuple[Path, str]
) -> None:
    root, _ = prepared_fixture
    with pytest.raises(ValueError, match="authorization"):
        module.prepare(root, Path(sys.executable), "  ")
    assert (root / module.FAILED_PARTS[0]).exists()
    assert not list(root.glob("recovery_oom_*"))


def test_mid_archive_failure_preserves_everything(
    module: ModuleType,
    prepared_fixture: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = prepared_fixture
    original_rename = Path.rename

    def fail_second(path: Path, destination: Path) -> Path:
        if path == root / module.FAILED_PARTS[1]:
            raise OSError("injected rename failure")
        return original_rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_second)
    with pytest.raises(OSError, match="injected"):
        module.prepare(root, Path(sys.executable), "approved")
    archive = next(root.glob("recovery_oom_*"))
    assert (archive / "prepare_failed.json").is_file()
    assert not (archive / "prepared.json").exists()
    assert (
        archive / "preserved" / module.FAILED_PARTS[0] / "incomplete.onnx"
    ).is_file()
    assert (root / module.FAILED_PARTS[1] / "incomplete.onnx").is_file()
    assert all((root / path).exists() for path in module.FAILED_LOGS)


@pytest.mark.parametrize("target", ["archive", "complete", "status", "intent"])
def test_prepared_tamper_rejected(
    module: ModuleType, prepared_fixture: tuple[Path, str], target: str
) -> None:
    root, complete = prepared_fixture
    archive = module.prepare(root, Path(sys.executable), "approved")
    paths = {
        "archive": archive / "preserved" / module.FAILED_LOGS[0],
        "complete": root / complete / "part1_of_4.bin",
        "status": root / "continuation_runs/old/status.json",
        "intent": archive / "intent.json",
    }
    write(paths[target], "changed")
    with pytest.raises(ValueError, match=r"changed|mismatch"):
        module.verify_prepared(root, archive, Path(sys.executable))


def test_run_consumed_before_controller_and_no_second_run(
    module: ModuleType,
    prepared_fixture: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = prepared_fixture
    archive = module.prepare(root, Path(sys.executable), "approved")
    calls = []

    class FakeController:
        def __init__(self, root: Path, python: Path, archive: Path) -> None:
            self.run_dir = root / "mock_run"

        def run(self) -> None:
            assert (archive / "run_consumed.json").is_file()
            calls.append("run")

    monkeypatch.setattr(module, "RecoveryController", FakeController)
    module.run(root, Path(sys.executable), archive)
    assert calls == ["run"]
    with pytest.raises(FileExistsError):
        module.run(root, Path(sys.executable), archive)
    assert calls == ["run"]


def test_launch_releases_locks_before_detached_child(
    module: ModuleType,
    prepared_fixture: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = prepared_fixture
    archive = module.prepare(root, Path(sys.executable), "approved")
    calls = []

    def popen(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        with module.locks(root):
            pass
        assert "--detach" not in argv
        assert kwargs["start_new_session"] is True
        calls.append(argv)
        return SimpleNamespace(pid=99)

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    module.launch(root, Path(sys.executable), archive)
    with pytest.raises(FileExistsError):
        module.launch(root, Path(sys.executable), archive)
    assert len(calls) == 1


def test_active_controller_stops_prepare(
    module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = module.resume.Process(
        100, ("python", "niah_continue.py", "--root", str(tmp_path))
    )
    monkeypatch.setattr(module.resume, "processes", lambda: [process])
    monkeypatch.setattr(module.resume, "build_processes", lambda *args: [])
    with pytest.raises(RuntimeError, match="controller/recovery"):
        module.ensure_idle(tmp_path)


def test_other_resume_lock_rejected(module: ModuleType, tmp_path: Path) -> None:
    with (
        module.build_lock(tmp_path),
        pytest.raises(RuntimeError, match="active"),
        module.locks(tmp_path),
    ):
        pass


def test_original_contract_hash_is_not_rewritten(
    module: ModuleType, tmp_path: Path
) -> None:
    contract = tmp_path / "continuation_contract.json"
    write(contract, '{"workers":1}')
    with pytest.raises(ValueError, match="workers=2"):
        module.validate_original(tmp_path, Path(sys.executable))
    assert contract.read_text() == '{"workers":1}'


def test_subclass_has_single_worker_and_preserves_contract(
    module: ModuleType, prepared_fixture: tuple[Path, str]
) -> None:
    root, _ = prepared_fixture
    archive = module.prepare(root, Path(sys.executable), "approved")
    write(root / "continuation_contract.json", "original immutable contract")
    controller = module.RecoveryController(root, Path(sys.executable), archive)
    controller.contract()
    assert controller.workers == 1
    assert (
        root / "continuation_contract.json"
    ).read_text() == "original immutable contract"
    amendment = json.loads((controller.run_dir / "recovery_amendment.json").read_text())
    assert amendment["workers"] == 1
    assert amendment["scientific_contract_unchanged"]


def test_special_file_inventory_rejected(module: ModuleType, tmp_path: Path) -> None:
    directory = tmp_path / "part"
    directory.mkdir()
    os.mkfifo(directory / "fifo")
    with pytest.raises(ValueError, match="Special file"):
        module.inventory(directory, {})
