# SPDX-License-Identifier: BSD-3-Clause
"""One explicitly approved, serial retry of the two failed 16K NIAH parts.

The original scientific/execution contracts are never edited. ``prepare`` moves
only the two known failed outputs and their parent logs into a hash-inventoried
archive. ``run`` consumes that authorization once and calls the unchanged NIAH
controller with one build worker. An interrupted prepare/run is not auto-retried.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import niah_build as build
import niah_continue as original
import niah_resume_build as resume

ORIGINAL_RUN = "20261102T110628_214172Z_1201201"
ORIGINAL_CONTRACT_SHA256 = (
    "6c747484933da75f8762732c932c5cb8f74bfc6306fc7558f2d601b75cbd57d5"
)
FAILED_PARTS = tuple(f"builds/cl16384/k4_v4/part{n}" for n in (2, 3))
FAILED_LOGS = tuple(f"logs/build_cl16384_k4_v4_part{n}.log" for n in (2, 3))
TARGETS = FAILED_PARTS + FAILED_LOGS
POLICY = "all probes first; fixed capacity/profile test blocks; no inference retries or capacity fallback"


def safe_path(root: Path, relative: str, *, exists: bool = True) -> Path:
    """Reject symlinked archive roots/parents, traversal and external targets."""
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    path = root
    for component in rel.parts:
        path /= component
        if path.is_symlink():
            raise ValueError(f"Symlinked archival path: {path}")
    if exists and not path.exists():
        raise FileNotFoundError(path)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Archive target escapes root: {path}")
    return path


@contextlib.contextmanager
def build_lock(root: Path) -> Iterator[None]:
    path = safe_path(root, ".niah_build_resume.lock", exists=False)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Build-resume helper is active") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def locks(root: Path) -> Iterator[None]:
    safe_path(root, ".niah_continue.lock", exists=False)
    with original.exclusive(root), build_lock(root):
        yield


def validate_original(root: Path, python: Path) -> dict:
    path = safe_path(root, "continuation_contract.json")
    if original.digest(path) != ORIGINAL_CONTRACT_SHA256:
        raise ValueError("Original workers=2 continuation contract changed")
    contract = original.read(path)
    expected_files = {
        str((original.HERE / name).resolve())
        for name in (
            "niah_continue.py",
            "niah_resume_build.py",
            "niah_device_copies.py",
        )
    } | {
        str((root / name).resolve())
        for name in (
            "protocol.json",
            "data/data_manifest.json",
            "execution_freeze.json",
        )
    }
    if (
        set(contract["files_sha256"]) != expected_files
        or contract["workers"] != 2
        or contract["python"] != str(python.resolve(strict=True))
        or contract["capacities"] != list(original.CAPACITIES)
        or contract["groups_in_fixed_order"] != list(original.GROUPS)
        or contract["policy"] != POLICY
    ):
        raise ValueError("Original continuation scope/configuration changed")
    for name, sha in contract["files_sha256"].items():
        if original.digest(Path(name)) != sha:
            raise ValueError(f"Original frozen source changed: {name}")
    return contract


def ensure_idle(root: Path) -> None:
    observed = resume.processes()
    if resume.build_processes(root, observed):
        raise RuntimeError("Host converter/QNN writers remain active")
    for process in observed:
        script = Path(process.argv[1]).name if len(process.argv) > 1 else ""
        target = resume.option(process.argv, "--root")
        if script not in ("niah_continue.py", "niah_oom_resume.py") or not target:
            continue
        if Path(target).resolve() != root:
            continue
        # The short-lived detach launcher may still be recording the child PID.
        if process.pid == os.getppid() and "--detach" in process.argv:
            continue
        raise RuntimeError(f"NIAH controller/recovery is active: PID {process.pid}")


def prior_failure(root: Path) -> dict:
    path = safe_path(root, f"continuation_runs/{ORIGINAL_RUN}/status.json")
    status = original.read(path)
    if status.get("status") != "failed" or status.get("stage") != "build_cl16384":
        raise ValueError("Original controller is not the known failed 16K run")
    if any(p.pid == status["pid"] for p in resume.processes()):
        raise RuntimeError("Original failed-controller PID is still live")
    return {
        "path": str(path.relative_to(root)),
        "sha256": original.digest(path),
        "status": status,
    }


def allowed_links(root: Path) -> dict[str, str]:
    protocol = original.read(root / "protocol.json")
    split = original.read(Path(protocol["split"]) / "split_manifest.json")
    return {
        f"{part['class']}.data": str(
            (Path(part["bundle_dir"]) / f"{part['class']}.data").resolve(strict=True)
        )
        for part in split["parts"].values()
    }


def inventory(path: Path, links: dict[str, str]) -> list[dict]:
    """Hash regular contents; preserve only known external weight links verbatim."""
    result = []
    entries = [path] if path.is_file() else [path, *sorted(path.rglob("*"))]
    for entry in entries:
        info = entry.lstat()
        item: dict[str, Any] = {
            "path": str(entry.relative_to(path)) if entry != path else "."
        }
        if stat.S_ISLNK(info.st_mode):
            target = entry.resolve(strict=True)
            if links.get(entry.name) != str(target) or not target.is_file():
                raise ValueError(f"Unapproved internal symlink: {entry}")
            item.update(
                kind="symlink",
                link=os.readlink(entry),
                target=str(target),
                bytes=target.stat().st_size,
                sha256=original.digest(target),
            )
        elif stat.S_ISREG(info.st_mode):
            item.update(kind="file", bytes=info.st_size, sha256=original.digest(entry))
        elif stat.S_ISDIR(info.st_mode):
            item.update(kind="directory")
        else:
            raise ValueError(f"Special file in recovery inventory: {entry}")
        result.append(item)
    return result


def failed_scope(root: Path) -> tuple[list[dict], list[dict]]:
    states = [
        state
        for capacity in original.CAPACITIES[1:]
        for state in resume.part_states(root, capacity, list(original.GROUPS))
    ]
    partial = {
        str(Path(s["output"]).relative_to(root))
        for s in states
        if s["state"] == "partial"
    }
    if partial != set(FAILED_PARTS):
        raise ValueError(f"Unexpected inactive partial scope: {sorted(partial)}")
    for state in states:
        relative = str(Path(state["output"]).relative_to(root))
        if state["state"] == "missing" and state["log_exists"]:
            raise ValueError(f"Unexpected prior failed attempt: {relative}")
        required = (
            "/cl16384/" not in relative
            or "/shared/" in relative
            or "/fp16/" in relative
            or relative.endswith("k4_v4/part4")
        )
        if required and state["state"] != "complete":
            raise ValueError(f"Expected completed part is unavailable: {relative}")
    for number in (2, 3):
        path = safe_path(root, f"logs/build_cl16384_k4_v4_part{number}.log")
        text = path.read_text()
        argv = json.loads(text.splitlines()[0])
        expected = build.convert_options(
            root, 16384, "k4_v4", number, root / f"builds/cl16384/k4_v4/part{number}"
        )
        if (
            argv[2:] != expected
            or "RuntimeError: qairt-converter failed; see " not in text
        ):
            raise ValueError(f"Failed converter provenance mismatch: {path}")
    return states, [
        s for s in states if s["state"] == "complete" and "/cl16384/" in s["output"]
    ]


def archive_one(
    root: Path, archive: Path, name: str, links: dict[str, str], expected: list[dict]
) -> None:
    source = safe_path(root, name)
    if inventory(source, links) != expected:
        raise ValueError(f"Source changed during archival: {source}")
    destination = safe_path(archive, f"preserved/{name}", exists=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise ValueError("Archive must use same-filesystem atomic rename")
    source.rename(destination)
    if inventory(destination, links) != expected:
        raise ValueError(f"Archive verification failed: {destination}")


def prepare(root: Path, python: Path, authorization: str) -> Path:
    if not authorization.strip():
        raise ValueError("Explicit user authorization text is required")
    with locks(root):
        if list(root.glob("recovery_oom_*")):
            raise FileExistsError(
                "An OOM recovery already exists; no automatic prepare retry"
            )
        ensure_idle(root)
        validate_original(root, python)
        original.verify_frozen(root)
        failure = prior_failure(root)
        states, complete = failed_scope(root)
        links = allowed_links(root)
        targets = {name: inventory(safe_path(root, name), links) for name in TARGETS}
        completed = {
            str(Path(s["output"]).relative_to(root)): inventory(
                safe_path(root, str(Path(s["output"]).relative_to(root))), links
            )
            for s in complete
        }
        ensure_idle(root)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        archive = safe_path(root, f"recovery_oom_{stamp}_{os.getpid()}", exists=False)
        archive.mkdir(exist_ok=False)
        value = {
            "created_utc": original.now(),
            "authorization": authorization,
            "original_contract_sha256": ORIGINAL_CONTRACT_SHA256,
            "helper_sha256": original.digest(Path(__file__)),
            "python": str(python.absolute()),
            "failure": failure,
            "scope": states,
            "targets": targets,
            "completed_parts": completed,
            "allowed_weight_links": links,
            "workers_before": 2,
            "workers_after": 1,
            "capacity_unchanged": 16384,
            "policy": "One explicitly approved build retry; no inference retries, capacity reduction or scientific changes",
        }
        original.save(archive / "intent.json", value)
        try:
            for index, name in enumerate(TARGETS):
                archive_one(root, archive, name, links, targets[name])
                original.save(
                    archive / f"move_{index}.json",
                    {
                        "source": name,
                        "destination": f"preserved/{name}",
                        "verified": True,
                    },
                )
            original.save(
                archive / "prepared.json",
                {
                    "utc": original.now(),
                    "intent_sha256": original.digest(archive / "intent.json"),
                    "targets": list(TARGETS),
                },
            )
        except BaseException as error:
            original.save(
                archive / "prepare_failed.json",
                {
                    "utc": original.now(),
                    "error": repr(error),
                    "policy": "No rollback/deletion/retry; all partial moves and intent preserved",
                },
            )
            raise
        print(
            json.dumps(
                {"prepared_archive": str(archive), "workers": 1, "capacity": 16384}
            ),
            flush=True,
        )
        return archive


def verify_prepared(root: Path, archive: Path, python: Path) -> dict:
    if archive.parent != root or not archive.name.startswith("recovery_oom_"):
        raise ValueError("Unexpected recovery archive")
    safe_path(root, archive.name)
    prepared = original.read(safe_path(archive, "prepared.json"))
    intent_path = safe_path(archive, "intent.json")
    if prepared["intent_sha256"] != original.digest(intent_path):
        raise ValueError("Recovery intent hash mismatch")
    intent = original.read(intent_path)
    if (
        prepared["targets"] != list(TARGETS)
        or set(intent["targets"]) != set(TARGETS)
        or intent["helper_sha256"] != original.digest(Path(__file__))
        or intent["workers_before"] != 2
        or intent["workers_after"] != 1
        or intent["capacity_unchanged"] != 16384
        or intent["original_contract_sha256"] != ORIGINAL_CONTRACT_SHA256
        or Path(intent["python"]).resolve() != python.resolve()
    ):
        raise ValueError("Recovery amendment/source changed")
    validate_original(root, python)
    if (
        original.digest(safe_path(root, intent["failure"]["path"]))
        != intent["failure"]["sha256"]
    ):
        raise ValueError("Original failed-controller status changed")
    for name, expected in intent["targets"].items():
        if (
            inventory(
                safe_path(archive, f"preserved/{name}"), intent["allowed_weight_links"]
            )
            != expected
        ):
            raise ValueError(f"Preserved failed output changed: {name}")
    for name, expected in intent["completed_parts"].items():
        if inventory(safe_path(root, name), intent["allowed_weight_links"]) != expected:
            raise ValueError(f"Preserved completed output changed: {name}")
    return intent


class RecoveryController(original.Controller):
    def __init__(self, root: Path, python: Path, archive: Path) -> None:
        super().__init__(root, python, workers=1)
        self.archive = archive

    def contract(self) -> None:
        intent = verify_prepared(self.root, self.archive, self.python)
        original.save(
            self.run_dir / "recovery_amendment.json",
            {
                "utc": original.now(),
                "original_contract_sha256": ORIGINAL_CONTRACT_SHA256,
                "archive": str(self.archive),
                "archive_intent_sha256": original.digest(self.archive / "intent.json"),
                "archive_prepared_sha256": original.digest(
                    self.archive / "prepared.json"
                ),
                "helper_sha256": original.digest(Path(__file__)),
                "authorization": intent["authorization"],
                "workers": 1,
                "scientific_contract_unchanged": True,
            },
        )


def run(root: Path, python: Path, archive: Path) -> None:
    # Holding the build-resume lock through Controller.run would block its child.
    safe_path(root, ".niah_continue.lock", exists=False)
    with original.exclusive(root):
        with build_lock(root):
            ensure_idle(root)
            verify_prepared(root, archive, python)
            for relative in TARGETS:
                if safe_path(root, relative, exists=False).exists():
                    raise FileExistsError(
                        "Retry output already exists; second retry forbidden"
                    )
            original.save(
                archive / "run_consumed.json",
                {"utc": original.now(), "pid": os.getpid()},
            )
        controller = RecoveryController(root, python, archive)
        original.save(
            archive / "controller.json",
            {"run_directory": str(controller.run_dir), "pid": os.getpid()},
        )
        controller.run()


def launch(root: Path, python: Path, archive: Path) -> dict:
    with locks(root):
        ensure_idle(root)
        verify_prepared(root, archive, python)
        if (archive / "run_consumed.json").exists():
            raise FileExistsError("Recovery already consumed; no second retry")
        original.save(
            archive / "launch_claim.json", {"utc": original.now(), "pid": os.getpid()}
        )
    # Release both nonblocking locks before spawning; the child acquires them.
    argv = [
        str(python),
        str(Path(__file__).resolve()),
        "run",
        "--root",
        str(root),
        "--python",
        str(python),
        "--archive",
        str(archive),
    ]
    with (archive / "controller.log").open("x") as log:
        child = subprocess.Popen(
            argv,
            cwd=original.REPO,
            env={
                **os.environ,
                "PYTHONPATH": str(original.REPO / "src"),
                "PYTHONUNBUFFERED": "1",
            },
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {
        "utc": original.now(),
        "pid": child.pid,
        "argv": argv,
        "log": str(archive / "controller.log"),
        "status": "launched_not_verified_complete",
    }
    original.save(archive / "launch.json", record)
    print(json.dumps(record, indent=2), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--authorization")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    root = args.root.absolute()
    if root != root.resolve(strict=True):
        raise ValueError("Recovery root/parents must not be symlinks")
    safe_path(root, ".niah_continue.lock", exists=False)
    python = args.python.absolute()
    if args.stage == "prepare":
        if args.archive or args.detach or not args.authorization:
            parser.error("prepare requires --authorization, without --archive/--detach")
        prepare(root, python, args.authorization)
    else:
        if args.archive is None or args.authorization:
            parser.error("run requires --archive and uses its preserved authorization")
        archive = args.archive.absolute()
        (launch if args.detach else run)(root, python, archive)


if __name__ == "__main__":
    main()
