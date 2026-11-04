# SPDX-License-Identifier: BSD-3-Clause
"""Scoped 2K supplement + preserved 1K/4K/8K scheduling, without a device."""

import contextlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"))
    return importlib.import_module("niah_8k_continue")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def data(capacities: tuple[int, ...], pilot_capacity: int) -> dict:
    cases = []
    for capacity in capacities:
        cases += [{"case_id": f"test_{capacity}_{n}", "split": "test", "capacity": capacity,
                   "unchanged_hash": f"hash_{capacity}_{n}"} for n in range(50)]
        cases.append({"case_id": f"smoke_{capacity}", "split": "smoke", "capacity": capacity})
    cases += [{"case_id": f"pilot_{pilot_capacity}_{n}", "split": "pilot", "capacity": pilot_capacity} for n in range(2)]
    return {"cases": cases, "protocol": {"capacities": list(capacities), "seed": 123}}


def setup_controller(module: ModuleType, tmp_path: Path) -> Any:
    side = tmp_path / module.SUPPLEMENT_NAME
    side.mkdir()
    scope = tmp_path / module.SCOPE_NAME
    scope.mkdir()
    old = data((1024, 4096, 8192, 16384), 1024)
    new = data((2048,), 2048)
    dump(tmp_path / "data/data_manifest.json", old)
    dump(side / "data/data_manifest.json", new)
    controller = module.Controller(tmp_path, side, Path(sys.executable), scope)
    return controller, old, new


def test_manifest_contains_800_runs_and_unchanged_existing_case_objects(module: ModuleType) -> None:
    old = data((1024, 4096, 8192, 16384), 1024)
    new = data((2048,), 2048)
    old_text, new_text = json.dumps(old), json.dumps(new)
    merged = module.selected_manifest(old, new, {"old": "a", "new": "b"})
    assert merged["case_counts"] == {"test": 200, "pilot": 4, "smoke": 4}
    assert merged["protocol"]["capacities"] == [1024, 2048, 4096, 8192]
    assert merged["scope_amendment"]["main_runs"] == 800
    assert merged["scope_amendment"]["excluded_capacity"] == 16384
    assert len(merged["cases"]) == 208
    assert all(c["capacity"] != 16384 for c in merged["cases"])
    assert merged["cases"][0] is old["cases"][0]
    assert merged["cases"][-1] is new["cases"][-1]
    assert json.dumps(old) == old_text and json.dumps(new) == new_text


@pytest.mark.parametrize("mutation", ["missing", "wrong_capacity", "duplicate"])
def test_invalid_supplement_rejected(module: ModuleType, mutation: str) -> None:
    old, new = data((1024, 4096, 8192, 16384), 1024), data((2048,), 2048)
    if mutation == "missing":
        new["cases"].pop()
    elif mutation == "wrong_capacity":
        new["cases"][0]["capacity"] = 16384
    else:
        new["cases"][0]["case_id"] = old["cases"][0]["case_id"]
    with pytest.raises(ValueError):
        module.selected_manifest(old, new, {})


def test_routes_original_and_supplement_separately(module: ModuleType, tmp_path: Path) -> None:
    controller, _, _ = setup_controller(module, tmp_path)
    for capacity in module.CAPACITIES:
        expected = controller.supplement_root if capacity == 2048 else tmp_path
        assert controller.target_root(capacity) == expected
        assert controller.args(capacity, "fp16") == ["--root", str(expected), "--capacity", str(capacity), "--group", "fp16"]
    with pytest.raises(ValueError):
        controller.target_root(16384)


def fake_schedule(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple:
    controller, old, _ = setup_controller(module, tmp_path)
    complete = {("pilot", 1024, "fp16")} | {("probe", c, g) for c in module.PRESERVED_CAPACITIES for g in module.GROUPS}
    events = []
    monkeypatch.setattr(module.original, "verify_frozen", lambda root: old)
    monkeypatch.setattr(controller, "contract", lambda: None)
    monkeypatch.setattr(module.original, "verify_audit", lambda root, c, g: events.append(("audit_verified", c, g)))
    monkeypatch.setattr(module.original, "stage_complete", lambda root, d, s, c, g: (s, c, g) in complete)
    monkeypatch.setattr(controller, "deploy", lambda c, g: events.append(("deploy", c, g)))
    monkeypatch.setattr(controller, "release", lambda c, g: events.append(("release", c, g)))
    monkeypatch.setattr(controller, "report", lambda: events.append(("report",)))

    def command(label: str, script: str, *options: str) -> None:
        capacity = int(options[options.index("--capacity") + 1])
        group = options[options.index("--group") + 1] if "--group" in options else None
        stage = options[0]
        events.append((stage, capacity, group, script))
        if stage == "measure":
            assert all(("probe", c, g) in complete for c in module.CAPACITIES for g in module.GROUPS)
            assert ("pilot", 2048, "fp16") in complete
        if stage in ("pilot", "probe", "measure"):
            complete.add((stage, capacity, group))

    monkeypatch.setattr(controller, "command", command)
    return controller, events, complete


def test_only_2k_built_all_16_probes_before_fixed_800_run_order(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller, events, _ = fake_schedule(module, monkeypatch, tmp_path)
    controller.run()
    assert [e[1] for e in events if e[0] == "build"] == [2048]
    assert [(e[1], e[2]) for e in events if e[0] == "pilot"] == [(2048, "fp16")]
    assert [(e[1], e[2]) for e in events if e[0] == "probe"] == [(2048, g) for g in module.GROUPS]
    assert [(e[1], e[2]) for e in events if e[0] == "measure"] == [(c, g) for c in module.CAPACITIES for g in module.GROUPS]
    assert not any(len(e) > 1 and e[1] == 16384 for e in events)
    gate = module.original.read(controller.run_dir / "verified_gates.json")
    assert len(gate["probes"]) == 16 and gate["main_runs"] == 800
    assert module.original.read(controller.run_dir / "status.json")["status"] == "complete"


def test_missing_preserved_probe_stops_before_any_build_or_main(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller, events, complete = fake_schedule(module, monkeypatch, tmp_path)
    complete.remove(("probe", 8192, "boundary_structured"))
    with pytest.raises(ValueError, match="12 preserved"):
        controller.run()
    assert not any(e[0] in ("build", "pilot", "probe", "measure") for e in events)


def test_failed_2k_probe_stops_without_main_or_retry(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller, events, _ = fake_schedule(module, monkeypatch, tmp_path)
    previous = controller.command

    def fail(label: str, script: str, *options: str) -> None:
        if label == "probe_cl2048_k4_v4":
            raise subprocess.CalledProcessError(1, [script])
        previous(label, script, *options)

    monkeypatch.setattr(controller, "command", fail)
    with pytest.raises(subprocess.CalledProcessError):
        controller.run()
    assert not any(e[0] == "measure" for e in events)
    assert module.original.read(controller.run_dir / "status.json")["status"] == "failed"


@pytest.mark.parametrize("capacity,script", [(1024, "niah_device_copies.py"), (2048, "niah_2k_device_copies.py")])
def test_device_copies_routed_to_scoped_adapter(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capacity: int, script: str) -> None:
    controller, _, _ = setup_controller(module, tmp_path)
    dump(controller.target_root(capacity) / "deployments" / f"cl{capacity}_fp16.json", {})
    calls = []
    monkeypatch.setattr(controller, "command", lambda *args: calls.append(args))
    controller.deploy(capacity, "fp16")
    controller.release(capacity, "fp16")
    assert calls[0][1:3] == (script, "restore")
    assert calls[1][1:3] == (script, "release")
    assert calls[1][-1] == "--if-present"
    assert str(controller.target_root(capacity)) in calls[0]


@pytest.mark.parametrize("script,stage,capacity,workers", [
    ("niah_build.py", "build", 2048, "1"),
    ("niah_2k_supplement.py", "build", 8192, "1"),
    ("niah_2k_supplement.py", "build", 2048, "2"),
    ("benchmark_niah.py", "probe", 8192, "1"),
    ("niah_device_copies.py", "release", 2048, "1"),
])
def test_command_guard_forbids_old_builds_repeats_or_parallel_2k(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: str, stage: str, capacity: int, workers: str) -> None:
    controller, _, _ = setup_controller(module, tmp_path)
    monkeypatch.setattr(module.original.Controller, "command", lambda *args: pytest.fail("Unapproved command escaped guard"))
    with pytest.raises(ValueError):
        controller.command("bad", script, stage, *controller.args(capacity), "--workers", workers)


def test_command_guard_accepts_serial_2k_build(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller, _, _ = setup_controller(module, tmp_path)
    calls = []
    monkeypatch.setattr(module.original.Controller, "command", lambda *args: calls.append(args))
    controller.command("build_cl2048", "niah_2k_supplement.py", "build", *controller.args(2048), "--workers", "1")
    assert len(calls) == 1


def test_completed_main_not_repeated(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller, events, complete = fake_schedule(module, monkeypatch, tmp_path)
    complete.add(("measure", 2048, "fp16"))
    controller.block("measure", 2048, "fp16")
    assert events == [("release", 2048, "fp16")]


def test_successful_exit_without_stage_completion_rejected(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller, _, _ = fake_schedule(module, monkeypatch, tmp_path)
    monkeypatch.setattr(controller, "command", lambda *args: None)
    with pytest.raises(ValueError, match="omitted verified"):
        controller.block("probe", 2048, "fp16")


def test_report_combines_both_roots_with_provenance(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reporting = importlib.import_module("niah_report")
    controller, old, new = setup_controller(module, tmp_path)
    selected = module.selected_manifest(old, new, {})
    dump(controller.scope / "selected_manifest.json", selected)
    dump(controller.scope / "amendment.json", {})
    dump(tmp_path / "results/a.json", {"split": "test", "capacity": 1024, "runner_report": "reports/a.json"})
    dump(controller.supplement_root / "results/b.json", {"split": "test", "capacity": 2048, "runner_report": "reports/b.json"})
    observed = {}

    def aggregate(records: list, manifest: dict, **kwargs: Any) -> dict:
        observed.update(records=records, manifest=manifest, options=kwargs)
        return {"status": "complete"}

    def write(value: dict, output: Path) -> None:
        dump(output / "summary.json", value)

    monkeypatch.setattr(reporting, "aggregate_records", aggregate)
    monkeypatch.setattr(reporting, "write_report", write)
    controller.report()
    assert observed["options"]["capacities"] == (1024, 2048, 4096, 8192)
    assert observed["manifest"] == selected
    assert {r["source_root"] for r in observed["records"]} == {str(tmp_path), str(controller.supplement_root)}
    assert all(len(r["source_result_sha256"]) == 64 for r in observed["records"])
    assert (tmp_path / "continuation_report_8k.json").is_file()
    assert not (tmp_path / "report").exists()
    with pytest.raises(FileExistsError):
        controller.report()


def test_excluded_16k_main_results_are_not_silently_dropped(module: ModuleType, tmp_path: Path) -> None:
    dump(tmp_path / "results/excluded.json", {"capacity": 16384, "split": "test"})
    with pytest.raises(ValueError, match="outside"):
        module.reject_excluded_results(tmp_path, module.PRESERVED_CAPACITIES)


def test_scope_authorization_and_one_shot_creation(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    side = tmp_path / module.SUPPLEMENT_NAME
    side.mkdir()
    monkeypatch.setattr(module.recovery, "validate_original", lambda *args: None)
    monkeypatch.setattr(module, "verify_supplement", lambda *args: None)
    monkeypatch.setattr(module, "merged_manifest", lambda *args: {"cases": []})
    monkeypatch.setattr(module, "source_hashes", lambda *args: {})
    with pytest.raises(ValueError, match="authorization"):
        module.create_scope(tmp_path, side, Path(sys.executable), "")
    scope = module.create_scope(tmp_path, side, Path(sys.executable), "User approved 2K and max8K")
    assert module.original.read(scope / "amendment.json")["main_runs"] == 800
    with pytest.raises(FileExistsError):
        module.create_scope(tmp_path, side, Path(sys.executable), "User approved 2K and max8K")


def test_launch_releases_locks_before_detaching_child(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scope = tmp_path / module.SCOPE_NAME
    scope.mkdir()
    state = {"locked": False}

    @contextlib.contextmanager
    def locks(root: Path):
        state["locked"] = True
        try:
            yield
        finally:
            state["locked"] = False

    def popen(argv: list[str], **kwargs: Any) -> Any:
        assert not state["locked"]
        assert kwargs["start_new_session"] is True
        assert "--supplement-root" in argv
        return SimpleNamespace(pid=99999)

    monkeypatch.setattr(module.recovery, "locks", locks)
    monkeypatch.setattr(module, "ensure_idle", lambda *args: None)
    monkeypatch.setattr(module, "create_scope", lambda *args: scope)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    result = module.launch(tmp_path, tmp_path / module.SUPPLEMENT_NAME, Path(sys.executable), "Approved")
    assert result["pid"] == 99999
    assert (scope / "launch_claim.json").is_file()
    with pytest.raises(FileExistsError):
        module.launch(tmp_path, tmp_path / module.SUPPLEMENT_NAME, Path(sys.executable), "Approved")


def test_run_authorization_consumed_once(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scope = tmp_path / module.SCOPE_NAME
    scope.mkdir()
    monkeypatch.setattr(module, "ensure_idle", lambda *args: None)
    monkeypatch.setattr(module, "verify_scope", lambda *args: {})
    monkeypatch.setattr(module.Controller, "run", lambda self: None)
    module.run(tmp_path, tmp_path / module.SUPPLEMENT_NAME, Path(sys.executable), scope)
    assert module.original.read(scope / "run_consumed.json")["pid"] == os.getpid()
    with pytest.raises(FileExistsError):
        module.run(tmp_path, tmp_path / module.SUPPLEMENT_NAME, Path(sys.executable), scope)


def test_idle_check_covers_both_roots_and_new_helpers(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    side = tmp_path / module.SUPPLEMENT_NAME
    seen = []
    process = SimpleNamespace(pid=999, argv=("python", "niah_2k_device_copies.py", "release", "--root", str(side)))
    monkeypatch.setattr(module.resume, "processes", lambda: [process])
    monkeypatch.setattr(module.resume, "build_processes", lambda root, procs: seen.append(root) or [])
    with pytest.raises(RuntimeError, match="remains active"):
        module.ensure_idle(tmp_path, side)
    assert seen == [tmp_path, side]
