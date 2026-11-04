# SPDX-License-Identifier: BSD-3-Clause
"""Resume the frozen NIAH experiment without retries or context fallback.

All 16 operational probes must pass before any main test is started. This
controller does not change the frozen build, inference, data, or report code.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

CAPACITIES = (1024, 4096, 8192, 16384)
GROUPS = ("fp16", "k4_v4", "boundary_lm", "boundary_structured")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path: Path, value: Any, *, replace: bool = False) -> None:
    """Durable exclusive records; only this run's live status is replaceable."""
    target = path.with_suffix(".pending") if replace else path
    with target.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(target, path)


@contextlib.contextmanager
def exclusive(root: Path) -> Iterator[None]:
    with (root / ".niah_continue.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another NIAH continuation controller is active"
            ) from error
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()} utc={now()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verify_frozen(root: Path) -> dict:
    import benchmark_niah as runtime
    import niah_build as build

    if not (root / "execution_freeze.json").is_file():
        raise ValueError("Run the manually supervised FP16 pilot/freeze first")
    protocol = build.check(root, CAPACITIES[0])
    runtime.freeze_sources(root, CAPACITIES[0])
    data = read(root / "data/data_manifest.json")
    if (
        protocol["context_capacities"] != list(CAPACITIES)
        or set(protocol["profiles"]) != set(GROUPS)
        or data["protocol"]["capacities"] != list(CAPACITIES)
        or data["case_counts"] != {"test": 200, "pilot": 2, "smoke": 4}
    ):
        raise ValueError("Frozen NIAH scope differs from the approved 16K plan")
    for case in data["cases"]:
        for kind in ("tokens", "prompt"):
            path = (root / "data" / case[f"{kind}_file"]).resolve(strict=True)
            if not path.is_relative_to((root / "data").resolve()):
                raise ValueError("Dataset path escapes its frozen directory")
            if digest(path) != case[f"{kind}_sha256"]:
                raise ValueError(f"Frozen dataset changed: {path}")
        if case["input_tokens"] + 64 > case["capacity"]:
            raise ValueError("Frozen prompt exceeds the context/generation budget")
    return data


def verify_audit(root: Path, capacity: int, group: str) -> None:
    path = root / "audits" / f"cl{capacity}" / f"{group}.json"
    audit = read(path)
    if (
        not audit.get("passed")
        or not audit.get("compiled_io", {}).get("passed")
        or not audit.get("checks")
        or not all(audit["checks"].values())
        or not audit.get("graphs")
        or any(g.get("violations") for g in audit["graphs"].values())
        or Path(audit["bundle"]).resolve()
        != (root / "bundles" / f"cl{capacity}" / group).resolve()
    ):
        raise ValueError(f"Preserved compiled audit failed: {path}")


def stage_complete(
    root: Path, data: dict, stage: str, capacity: int, group: str
) -> bool:
    """Never trust a controller checkpoint alone to skip an inference stage."""
    import benchmark_niah as runtime

    split = {"pilot": "pilot", "probe": "smoke", "measure": "test"}[stage]
    cases = [
        c for c in data["cases"] if c["split"] == split and c["capacity"] == capacity
    ]
    expected_count = {"pilot": 2, "probe": 1, "measure": 50}[stage]
    if len(cases) != expected_count:
        raise ValueError(f"Unexpected frozen {stage} case count")
    completion_file = root / "stage_completion" / f"{stage}_cl{capacity}_{group}.json"
    # An earlier failed attempt is not silently retried, even without completion.
    for case in cases:
        result = runtime.result_path(root, group, case)
        if result.exists() and read(result).get("status") != "ok":
            raise ValueError(f"Failed result preserved; no automatic retry: {result}")
    if not completion_file.exists():
        return False
    completion = read(completion_file)
    dep_file = runtime.deployment_path(root, capacity, group)
    dep_hash, dep = digest(dep_file), read(dep_file)
    if (
        not completion.get("pre_post_host_device_hashes_verified")
        or completion.get("cases") != [c["case_id"] for c in cases]
        or completion.get("deployment_sha256") != dep_hash
        or digest(root / dep["compiled_audit_file"]) != dep["compiled_audit_sha256"]
        or digest(root / "execution_freeze.json") != dep["execution_freeze_sha256"]
        or digest(root / "bundles" / f"cl{capacity}" / group / "convert_report.json")
        != dep["convert_report_sha256"]
    ):
        raise ValueError(f"Stage completion provenance mismatch: {completion_file}")
    verify_audit(root, capacity, group)
    for case in cases:
        result = runtime.check_existing(root, group, case, dep_hash)
        if result is None:
            raise ValueError("Completed stage is missing a result")
        runtime.validate_report(
            read(root / result["runner_report"]),
            case,
            dep,
            runtime.name(root, capacity, group),
        )
        if stage == "pilot" and not result.get("exact_match"):
            raise ValueError("FP16 pilot did not pass exact-match gate")
    return True


class Controller:
    def __init__(self, root: Path, python: Path, workers: int = 2) -> None:
        self.root, self.python, self.workers = root, python, workers
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_dir = root / "continuation_runs" / f"{stamp}_{os.getpid()}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.counter = 0
        self.current = "verify"
        self.data: dict = {}

    def status(self, status: str, **extra: Any) -> None:
        save(
            self.run_dir / "status.json",
            {
                "utc": now(),
                "status": status,
                "stage": self.current,
                "pid": os.getpid(),
                **extra,
            },
            replace=True,
        )

    def contract(self) -> None:
        files = [
            HERE / name
            for name in (
                "niah_continue.py",
                "niah_resume_build.py",
                "niah_device_copies.py",
            )
        ]
        files += [
            self.root / name
            for name in (
                "protocol.json",
                "data/data_manifest.json",
                "execution_freeze.json",
            )
        ]
        value = {
            "files_sha256": {str(p.resolve()): digest(p) for p in files},
            "python": str(self.python.resolve()),
            "capacities": list(CAPACITIES),
            "groups_in_fixed_order": list(GROUPS),
            "workers": self.workers,
            "policy": "all probes first; fixed capacity/profile test blocks; no inference retries or capacity fallback",
        }
        target = self.root / "continuation_contract.json"
        if target.exists():
            if read(target) != value:
                raise ValueError(
                    "Continuation contract changed; preserve results and stop"
                )
        else:
            save(target, value)

    def command(self, label: str, script: str, *options: str) -> None:
        self.current = label
        self.counter += 1
        prefix = self.run_dir / f"{self.counter:03d}_{label}"
        argv = [str(self.python), str(HERE / script), *options]
        save(
            prefix.with_suffix(".start.json"),
            {"utc": now(), "argv": argv, "cwd": str(REPO)},
        )
        self.status("running", command=argv)
        print(f"{now()} START {label}", flush=True)
        env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONUNBUFFERED": "1"}
        with prefix.with_suffix(".log").open("x") as log:
            subprocess.run(
                argv,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            log.flush()
            os.fsync(log.fileno())
        save(prefix.with_suffix(".complete.json"), {"utc": now(), "returncode": 0})
        print(f"{now()} DONE  {label}", flush=True)

    def args(self, capacity: int, group: str | None = None) -> list[str]:
        result = ["--root", str(self.root), "--capacity", str(capacity)]
        return result + (["--group", group] if group is not None else [])

    def assets(self, capacity: int) -> None:
        self.command(
            f"assets_cl{capacity}", "benchmark_niah.py", "assets", *self.args(capacity)
        )

    def audit(self, capacity: int, group: str) -> None:
        if not (self.root / "audits" / f"cl{capacity}" / f"{group}.json").exists():
            self.command(
                f"audit_cl{capacity}_{group}",
                "niah_build.py",
                "audit",
                *self.args(capacity, group),
            )
        verify_audit(self.root, capacity, group)

    def deploy(self, capacity: int, group: str) -> None:
        existing = self.root / "deployments" / f"cl{capacity}_{group}.json"
        if existing.exists():
            self.command(
                f"restore_cl{capacity}_{group}",
                "niah_device_copies.py",
                "restore",
                *self.args(capacity, group),
            )
        else:
            self.command(
                f"deploy_cl{capacity}_{group}",
                "benchmark_niah.py",
                "deploy",
                *self.args(capacity, group),
            )

    def release(self, capacity: int, group: str) -> None:
        self.command(
            f"release_cl{capacity}_{group}",
            "niah_device_copies.py",
            "release",
            *self.args(capacity, group),
            "--if-present",
        )

    def block(self, stage: str, capacity: int, group: str) -> None:
        if not stage_complete(self.root, self.data, stage, capacity, group):
            self.deploy(capacity, group)
            self.command(
                f"{stage}_cl{capacity}_{group}",
                "benchmark_niah.py",
                stage,
                *self.args(capacity, group),
            )
            if not stage_complete(self.root, self.data, stage, capacity, group):
                raise ValueError(
                    f"Successful command omitted verified {stage} completion"
                )
        else:
            print(
                f"{now()} VERIFIED completed {stage} CL{capacity} {group}; no repeat",
                flush=True,
            )
        if stage != "pilot":
            self.release(capacity, group)

    def report(self) -> None:
        output = self.root / "report"
        marker = self.root / "continuation_report.json"
        inputs = {
            str(p.relative_to(self.root)): digest(p)
            for p in sorted((self.root / "results").glob("*.json"))
        }
        if marker.exists():
            prior = read(marker)
            if prior["inputs_sha256"] != inputs or any(
                digest(output / name) != sha
                for name, sha in prior["files_sha256"].items()
            ):
                raise ValueError("Final report or its inputs changed")
            return
        if output.exists():
            raise FileExistsError(
                "Uncertified report preserved; refusing overwrite or automatic retry"
            )
        self.command(
            "report",
            "niah_report.py",
            "--root",
            str(self.root),
            "--output-dir",
            str(output),
        )
        if read(output / "summary.json").get("status") != "complete":
            raise ValueError("Final report has incomplete coverage")
        save(
            marker,
            {
                "utc": now(),
                "inputs_sha256": inputs,
                "files_sha256": {
                    p.name: digest(p) for p in output.iterdir() if p.is_file()
                },
            },
        )

    def all_probes(self) -> None:
        self.current = "all_probes_gate"
        if not all(
            stage_complete(self.root, self.data, "probe", c, g)
            for c in CAPACITIES
            for g in GROUPS
        ):
            raise ValueError(
                "All 16 operational probes must pass before any main evaluation"
            )

    def run(self) -> None:
        try:
            self.status("running")
            self.data = verify_frozen(self.root)
            self.contract()
            self.audit(1024, "fp16")
            self.assets(1024)
            self.block("pilot", 1024, "fp16")
            for capacity in CAPACITIES:
                if capacity != 1024:
                    self.command(
                        f"build_cl{capacity}",
                        "niah_resume_build.py",
                        "resume",
                        *self.args(capacity),
                        "--group",
                        *GROUPS,
                        "--workers",
                        str(self.workers),
                    )
                    self.assets(capacity)
                for group in GROUPS:
                    self.audit(capacity, group)
                    self.block("probe", capacity, group)
            self.all_probes()
            for capacity in CAPACITIES:
                for group in GROUPS:
                    self.block("measure", capacity, group)
            self.report()
            self.current = "complete"
            self.status("complete")
        except BaseException as error:
            self.status(
                "failed",
                error=repr(error),
                policy="Stopped without automatic retry or context reduction",
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Launch a disconnect-safe child; this does not mean the experiment completed",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    python = args.python.expanduser().absolute()
    if args.detach:
        launch(root, python, args.workers)
        return
    with exclusive(root):
        Controller(root, python, args.workers).run()


def launch(root: Path, python: Path, workers: int) -> dict:
    """The child takes the process lock; its PID is not a completion claim."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root / "continuation_launches" / f"{stamp}_{os.getpid()}"
    directory.mkdir(parents=True, exist_ok=False)
    log_path = directory / "controller.log"
    argv = [
        str(python),
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        "--python",
        str(python),
        "--workers",
        str(workers),
    ]
    with log_path.open("x") as log:
        child = subprocess.Popen(
            argv,
            cwd=REPO,
            env={
                **os.environ,
                "PYTHONPATH": str(REPO / "src"),
                "PYTHONUNBUFFERED": "1",
            },
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {
        "utc": now(),
        "pid": child.pid,
        "argv": argv,
        "log": str(log_path),
        "status": "launched_not_verified_complete",
        "note": "Inspect child log and continuation_runs status; duplicate children fail the exclusive lock",
    }
    save(directory / "launch.json", record)
    print(json.dumps(record, indent=2), flush=True)
    return record


if __name__ == "__main__":
    main()
