# SPDX-License-Identifier: BSD-3-Clause
"""One approved serial 16K retry after the WSL swap expansion to 128 GiB.

The previous recovery and all scientific contracts remain immutable. Only the
second failed K4/V4 part2/part3 outputs are moved into a separately hashed archive.
This helper cannot retry itself, reduce context, or change inference settings.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import niah_oom_resume as previous

original = previous.original
resume = previous.resume
safe_path = previous.safe_path
inventory = previous.inventory
locks = previous.locks
TARGETS = previous.TARGETS
PRIOR_ARCHIVE = "recovery_oom_20261103T070729_621977Z_263768"
PRIOR_RUN = "20261103T071059_020299Z_266185"
PRIOR_PID = 266185
MIN_SWAP_BYTES = 120 * 1024**3


def host_state(proc: Path = Path("/proc")) -> dict:
    memory = {}
    for line in (proc / "meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            memory[key + "_bytes"] = int(value.split()[0]) * 1024
    if memory.get("SwapTotal_bytes", 0) < MIN_SWAP_BYTES:
        raise RuntimeError("Expanded swap is unavailable; at least 120 GiB required")
    return {
        "utc": original.now(),
        "boot_id": (proc / "sys/kernel/random/boot_id").read_text().strip(),
        "uptime_seconds": float((proc / "uptime").read_text().split()[0]),
        **memory,
    }


def ensure_idle(root: Path) -> None:
    previous.ensure_idle(root)
    for process in resume.processes():
        script = Path(process.argv[1]).name if len(process.argv) > 1 else ""
        target = resume.option(process.argv, "--root")
        if script != Path(__file__).name or not target or Path(target).resolve() != root:
            continue
        if process.pid == os.getppid() and "--detach" in process.argv:
            continue
        raise RuntimeError(f"Swap recovery is already active: PID {process.pid}")


def prior_failure(root: Path, python: Path) -> dict:
    """Verify the immutable first retry, not just an arbitrary failed status."""
    archive = safe_path(root, PRIOR_ARCHIVE)
    intent = previous.verify_prepared(root, archive, python)
    run_dir = safe_path(root, f"continuation_runs/{PRIOR_RUN}")
    status = original.read(safe_path(run_dir, "status.json"))
    amendment = original.read(safe_path(run_dir, "recovery_amendment.json"))
    controller = original.read(safe_path(archive, "controller.json"))
    consumed = original.read(safe_path(archive, "run_consumed.json"))
    launch = original.read(safe_path(archive, "launch.json"))
    expected_argv = [
        str(python), str(Path(previous.__file__).resolve()), "run", "--root",
        str(root), "--python", str(python), "--archive", str(archive),
    ]
    build_argv = [
        str(python), str(original.HERE / "niah_resume_build.py"), "resume",
        "--root", str(root), "--capacity", "16384", "--group",
        *original.GROUPS, "--workers", "1",
    ]
    if (
        status.get("status") != "failed"
        or status.get("stage") != "build_cl16384"
        or status.get("pid") != PRIOR_PID
        or status.get("error") != f"CalledProcessError(1, {build_argv!r})"
        or controller != {"run_directory": str(run_dir), "pid": PRIOR_PID}
        or consumed.get("pid") != PRIOR_PID
        or launch.get("pid") != PRIOR_PID
        or launch.get("argv") != expected_argv
        or amendment.get("workers") != 1
        or amendment.get("scientific_contract_unchanged") is not True
        or amendment.get("archive") != str(archive)
        or amendment.get("archive_intent_sha256") != original.digest(archive / "intent.json")
        or amendment.get("archive_prepared_sha256") != original.digest(archive / "prepared.json")
        or amendment.get("helper_sha256") != original.digest(Path(previous.__file__))
        or amendment.get("authorization") != intent["authorization"]
        or amendment.get("original_contract_sha256") != previous.ORIGINAL_CONTRACT_SHA256
    ):
        raise ValueError("Previous serial-retry provenance mismatch")
    paths = [
        archive / name for name in (
            "intent.json", "prepared.json", "controller.json", "run_consumed.json", "launch.json",
        )
    ] + [run_dir / "status.json", run_dir / "recovery_amendment.json"]
    return {str(path.relative_to(root)): original.digest(path) for path in paths}


def prepare(root: Path, python: Path, authorization: str) -> Path:
    if not authorization.strip():
        raise ValueError("Explicit user authorization text is required")
    with locks(root):
        if list(root.glob("recovery_swap_*")):
            raise FileExistsError("Swap recovery already exists; no prepare retry")
        ensure_idle(root)
        host = host_state()
        original.verify_frozen(root)
        provenance = prior_failure(root, python)
        states, complete = previous.failed_scope(root)
        links = previous.allowed_links(root)
        targets = {name: inventory(safe_path(root, name), links) for name in TARGETS}
        completed = {
            str(Path(state["output"]).relative_to(root)): inventory(
                safe_path(root, str(Path(state["output"]).relative_to(root))), links
            ) for state in complete
        }
        ensure_idle(root)
        if host_state()["boot_id"] != host["boot_id"]:
            raise RuntimeError("Host rebooted during preparation")
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        archive = safe_path(root, f"recovery_swap_{stamp}_{os.getpid()}", exists=False)
        archive.mkdir(exist_ok=False)
        original.save(archive / "intent.json", {
            "created_utc": original.now(), "authorization": authorization,
            "helper_sha256": original.digest(Path(__file__)),
            "previous_helper_sha256": original.digest(Path(previous.__file__)),
            "original_contract_sha256": previous.ORIGINAL_CONTRACT_SHA256,
            "python": str(python.absolute()), "prior_provenance": provenance,
            "host": host, "scope": states, "targets": targets,
            "completed_parts": completed, "allowed_weight_links": links,
            "workers_before": 1, "workers_after": 1, "capacity_unchanged": 16384,
            "swap_change": "24 GiB to 128 GiB, configured by user before retry",
            "policy": "One approved build retry; no inference retries, context reduction or scientific changes",
        })
        try:
            for index, name in enumerate(TARGETS):
                previous.archive_one(root, archive, name, links, targets[name])
                original.save(archive / f"move_{index}.json", {
                    "source": name, "destination": f"preserved/{name}", "verified": True,
                })
            original.save(archive / "prepared.json", {
                "utc": original.now(), "intent_sha256": original.digest(archive / "intent.json"),
                "targets": list(TARGETS),
            })
        except BaseException as error:
            original.save(archive / "prepare_failed.json", {
                "utc": original.now(), "error": repr(error),
                "policy": "No rollback/deletion/retry; partial moves and intent preserved",
            })
            raise
        print(json.dumps({"prepared_archive": str(archive), "workers": 1, "capacity": 16384}), flush=True)
        return archive


def verify_prepared(root: Path, archive: Path, python: Path) -> dict:
    if archive.parent != root or not archive.name.startswith("recovery_swap_"):
        raise ValueError("Unexpected swap recovery archive")
    safe_path(root, archive.name)
    prepared = original.read(safe_path(archive, "prepared.json"))
    intent_path = safe_path(archive, "intent.json")
    if prepared["intent_sha256"] != original.digest(intent_path):
        raise ValueError("Swap recovery intent hash mismatch")
    intent = original.read(intent_path)
    if (
        prepared["targets"] != list(TARGETS)
        or set(intent["targets"]) != set(TARGETS)
        or intent["helper_sha256"] != original.digest(Path(__file__))
        or intent["previous_helper_sha256"] != original.digest(Path(previous.__file__))
        or intent["original_contract_sha256"] != previous.ORIGINAL_CONTRACT_SHA256
        or intent["workers_before"] != 1 or intent["workers_after"] != 1
        or intent["capacity_unchanged"] != 16384
        or Path(intent["python"]).resolve() != python.resolve()
        or intent["prior_provenance"] != prior_failure(root, python)
    ):
        raise ValueError("Swap recovery source/provenance changed")
    host = host_state()
    if any(host[key] != intent["host"][key] for key in ("boot_id", "SwapTotal_bytes")):
        raise ValueError("Host boot/swap configuration changed since preparation")
    for name, expected in intent["targets"].items():
        if inventory(safe_path(archive, f"preserved/{name}"), intent["allowed_weight_links"]) != expected:
            raise ValueError(f"Archived failed output changed: {name}")
    for name, expected in intent["completed_parts"].items():
        if inventory(safe_path(root, name), intent["allowed_weight_links"]) != expected:
            raise ValueError(f"Completed output changed: {name}")
    return intent


class RecoveryController(original.Controller):
    def __init__(self, root: Path, python: Path, archive: Path) -> None:
        super().__init__(root, python, workers=1)
        self.archive = archive

    def contract(self) -> None:
        intent = verify_prepared(self.root, self.archive, self.python)
        original.save(self.run_dir / "recovery_amendment.json", {
            "utc": original.now(), "archive": str(self.archive),
            "archive_intent_sha256": original.digest(self.archive / "intent.json"),
            "archive_prepared_sha256": original.digest(self.archive / "prepared.json"),
            "original_contract_sha256": previous.ORIGINAL_CONTRACT_SHA256,
            "helper_sha256": original.digest(Path(__file__)),
            "authorization": intent["authorization"], "workers": 1,
            "capacity_unchanged": 16384, "host": host_state(),
            "scientific_contract_unchanged": True,
        })


def run(root: Path, python: Path, archive: Path) -> None:
    safe_path(root, ".niah_continue.lock", exists=False)
    with original.exclusive(root):
        with previous.build_lock(root):
            ensure_idle(root)
            verify_prepared(root, archive, python)
            for name in TARGETS:
                if safe_path(root, name, exists=False).exists():
                    raise FileExistsError("Retry output exists; second retry forbidden")
            original.save(archive / "run_consumed.json", {
                "utc": original.now(), "pid": os.getpid(), "host": host_state(),
            })
        controller = RecoveryController(root, python, archive)
        original.save(archive / "controller.json", {
            "run_directory": str(controller.run_dir), "pid": os.getpid(),
        })
        controller.run()


def launch(root: Path, python: Path, archive: Path) -> dict:
    with locks(root):
        ensure_idle(root)
        verify_prepared(root, archive, python)
        if (archive / "run_consumed.json").exists():
            raise FileExistsError("Swap recovery already consumed")
        original.save(archive / "launch_claim.json", {"utc": original.now(), "pid": os.getpid()})
    # The detached child must acquire both locks after the launcher releases them.
    argv = [str(python), str(Path(__file__).resolve()), "run", "--root", str(root),
            "--python", str(python), "--archive", str(archive)]
    with (archive / "controller.log").open("x") as log:
        child = subprocess.Popen(
            argv, cwd=original.REPO,
            env={**os.environ, "PYTHONPATH": str(original.REPO / "src"), "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {"utc": original.now(), "pid": child.pid, "argv": argv,
              "log": str(archive / "controller.log"), "status": "launched_not_verified_complete"}
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
    root, python = args.root.absolute(), args.python.absolute()
    if root != root.resolve(strict=True):
        raise ValueError("Recovery root/parents must not be symlinks")
    if args.stage == "prepare":
        if args.archive or args.detach or not args.authorization:
            parser.error("prepare requires --authorization without --archive/--detach")
        prepare(root, python, args.authorization)
    else:
        if args.archive is None or args.authorization:
            parser.error("run requires --archive and its preserved authorization")
        (launch if args.detach else run)(root, python, args.archive.absolute())


if __name__ == "__main__":
    main()
