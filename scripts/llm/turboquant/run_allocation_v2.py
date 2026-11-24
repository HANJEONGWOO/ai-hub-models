# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Ordered experiment continuation, stopping on errors without altering budgets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark_asymmetric_once import execute, read


class EventJournal:
    """Serialize concurrent appends, including on WSL's mounted Windows drive."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def __call__(self, stage: str, state: str, **extra: object) -> None:
        with self.lock:
            value = {"time": time.time(), "stage": stage, "state": state, **extra}
            with self.path.open("a") as stream:
                stream.write(json.dumps(value) + "\n")
            print(json.dumps(value), flush=True)


def run(root: Path, search_pid: int | None) -> None:
    # Append-only execution journal; result-producing stages refuse overwrites.
    event = EventJournal(root / "logs/pipeline_events.jsonl")

    def stage(script: str, name: str, options: list[str]) -> None:
        log = root / "logs" / (name + ".log")
        attempt = 1
        while log.exists():
            attempt += 1
            log = root / "logs" / f"{name}_attempt{attempt}.log"
        event(name, "start", log=str(log))
        try:
            execute(script, [*options, "--root", str(root)], log)
        except Exception as exc:
            event(name, "failed", error=str(exc))
            raise
        event(name, "complete")

    stage("benchmark_allocation_v2.py", "static_probe", ["probe"])
    if not (root / "nominations.json").exists():
        if search_pid is None:
            stage("search_allocation_v2.py", "search", ["search"])
        else:
            event("search", "wait_for_existing", pid=search_pid)
            while True:
                path = Path(f"/proc/{search_pid}/cmdline")
                command = (
                    path.read_bytes().replace(b"\0", b" ") if path.exists() else b""
                )
                if (
                    b"search_allocation_v2.py search" not in command
                    or str(root).encode() not in command
                ):
                    if (root / "nominations.json").exists():
                        # Existence alone is not completion: wait for the writer
                        # to exit, then parse the complete immutable result.
                        read(root / "nominations.json")
                        break
                    event("search", "failed", error="Search exited without nominations")
                    raise RuntimeError("Existing search stopped before completion")
                time.sleep(30)
            event("search", "complete")
    if not (root / "physical_groups.json").exists():
        stage("benchmark_allocation_v2.py", "candidate_builds", ["candidates"])
    if not (root / "selection.json").exists():
        stage("benchmark_allocation_v2.py", "htp_calibration", ["calibrate"])

    def certificate() -> None:
        if not (root / "local_certificate.json").exists():
            stage("search_allocation_v2.py", "full_swap_certificate", ["certificate"])

    # Selection is already immutable. Independent GPU/HTP resources can overlap.
    with ThreadPoolExecutor(max_workers=2) as pool:
        cert = pool.submit(certificate)
        stage("benchmark_allocation_v2.py", "htp_evaluation", ["evaluate"])
        cert.result()
    if not (root / "regression_final.xml").exists():
        event("regression", "start")
        files = sorted(
            Path("src/qai_hub_models/test/test_models").glob("test_turboquant*.py")
        )
        with (root / "logs/regression_final.log").open("x") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    *map(str, files),
                    "--junitxml=" + str(root / "regression_final.xml"),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        event("regression", "complete")
    if not (root / "comparison.json").exists():
        stage("report_allocation_v2.py", "comparison", ["summary"])
    event(
        "pipeline",
        "measurements_complete",
        selected=read(root / "selection.json")["selected_group"],
        remaining="Review evidence, update README, run final verification; no commit",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--wait-for-search-pid", type=int)
    a = p.parse_args()
    run(a.root.resolve(), a.wait_for_search_pid)
