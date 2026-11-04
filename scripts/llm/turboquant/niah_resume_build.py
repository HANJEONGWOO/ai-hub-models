# SPDX-License-Identifier: BSD-3-Clause
"""Resume NIAH orchestration after disconnect, never retry a partial converter.

Run outside the sandbox PID namespace so /proc exposes existing host converters.
`inspect` is read-only. `resume` waits for matching live converters, then invokes
the unchanged niah_build.build only when every existing part is complete. No
process is killed, no partial directory is removed, and no context is reduced.
Do not launch the original build command concurrently with this helper.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import niah_build as build


@dataclass(frozen=True)
class Process:
    pid: int
    argv: tuple[str, ...]


def processes(proc_root: Path = Path("/proc")) -> list[Process]:
    """Read same-user live processes; inaccessible same-user state is an error."""
    init = (proc_root / "1/cmdline").read_bytes().split(b"\0")[0]
    if b"codex-linux-sandbox" in init:
        raise RuntimeError("Host processes are hidden; run with require_escalated")
    result = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            stat = (entry / "stat").read_text()
            if stat[stat.rfind(")") + 2 :].split()[0] == "Z":
                continue
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        argv = tuple(os.fsdecode(v) for v in raw.split(b"\0") if v)
        if argv:
            result.append(Process(int(entry.name), argv))
    return result


def option(argv: tuple[str, ...], name: str) -> str | None:
    if name not in argv:
        return None
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise ValueError(f"Missing value for {name}")
    return argv[index + 1]


def in_tree(value: str, root: Path) -> bool:
    return value.startswith("/") and Path(value).is_relative_to(root)


def build_processes(root: Path, observed: list[Process]) -> list[dict]:
    """Identify real Python entrypoints / QNN tools, not shell command text."""
    base = root / "builds"
    result = []
    for p in observed:
        executable = Path(p.argv[0]).name
        # Existing commands use Python followed immediately by the script path.
        script = Path(p.argv[1]).name if len(p.argv) > 1 else ""
        python = executable.startswith("python")
        if python and script in ("niah_build.py", "niah_resume_build.py"):
            target = option(p.argv, "--root")
            stages = {"build"} if script == "niah_build.py" else {"resume"}
            if target and Path(target).resolve() == root and stages & set(p.argv):
                raise RuntimeError(f"Another build orchestrator is active: PID {p.pid}")
        if python and script == "convert_parts.py":
            target = option(p.argv, "--out")
            if not target or not in_tree(target, base):
                continue
            output = Path(target)
            parts = output.relative_to(base).parts
            if len(parts) != 3 or not parts[0].startswith("cl"):
                raise ValueError(f"Unexpected converter output: {output}")
            capacity = int(parts[0][2:])
            group = "fp16" if parts[1] == "shared" else parts[1]
            number = int(parts[2].removeprefix("part"))
            if (
                capacity not in build.CAPACITIES
                or group not in build.GROUPS
                or number not in (1, 2, 3, 4)
                or (number == 1) != (parts[1] == "shared")
            ):
                raise ValueError(f"Unexpected converter target: {output}")
            expected = build.convert_options(root, capacity, group, number, output)
            if list(p.argv[2:]) != expected:
                raise ValueError(f"Active converter configuration mismatch: PID {p.pid}")
            result.append(
                {"pid": p.pid, "kind": "converter", "output": str(output)}
            )
        else:
            # A converter parent may die while a QNN child is still writing.
            tool = script if python else executable
            if tool.startswith(("qairt-", "qnn-")) and any(
                in_tree(a, base) for a in p.argv
            ):
                result.append({"pid": p.pid, "kind": "qnn_child", "tool": tool})
    return result


def part_states(root: Path, capacity: int, groups: list[str]) -> list[dict]:
    tasks = [("shared", 1)] + [(g, n) for g in groups for n in (2, 3, 4)]
    result = []
    for group, number in tasks:
        output = root / "builds" / f"cl{capacity}" / group / f"part{number}"
        effective = "fp16" if group == "shared" else group
        try:
            complete = build.completed_part(output, number, capacity, effective)
        except json.JSONDecodeError:
            # Existing converter writes its report only after context compilation.
            # A read during that brief write is not proof that the writer failed.
            complete = False
        state = (
            "complete" if complete else "partial" if output.exists() else "missing"
        )
        log = root / "logs" / f"build_cl{capacity}_{group}_part{number}.log"
        result.append(
            {"output": str(output), "state": state, "log_exists": log.exists()}
        )
    return result


def ready_parts(states: list[dict]) -> None:
    for part in states:
        if part["state"] == "partial":
            raise FileExistsError(
                f"Inactive partial build retained; automatic retry forbidden: {part['output']}"
            )
        if part["state"] == "missing" and part["log_exists"]:
            raise FileExistsError(
                f"Previous attempt log retained without completed output: {part['output']}"
            )


def snapshot(root: Path, capacity: int, groups: list[str]) -> dict:
    return {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "active": build_processes(root, processes()),
        "parts": part_states(root, capacity, groups),
    }


def resume(root: Path, capacity: int, groups: list[str], workers: int, poll: int) -> None:
    # Never trust another process's command line instead of the frozen inputs.
    build.check(root, capacity)
    with (root / ".niah_build_resume.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another NIAH resume helper holds the lock") from exc
        run = (
            dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:8]
        )
        ledger = root / "recovery" / run
        ledger.mkdir(parents=True, exist_ok=False)
        first = snapshot(root, capacity, groups)
        build.save_new(
            ledger / "start.json",
            {
                "helper_sha256": build.digest(Path(__file__)),
                "capacity": capacity,
                "groups": groups,
                "workers": workers,
                "policy": "Wait for existing writers; preserve completed parts; stop on inactive partials.",
                "snapshot": first,
            },
        )
        current = first
        while current["active"]:
            print(
                json.dumps({"waiting_for": current["active"], "utc": current["utc"]}),
                flush=True,
            )
            time.sleep(poll)
            current = snapshot(root, capacity, groups)
        ready_parts(current["parts"])
        build.save_new(ledger / "ready.json", current)
        # Existing implementation independently rechecks frozen source hashes,
        # reuses complete parts, refuses partial directories and exclusive logs.
        build.build(root, capacity, groups, workers)
        build.save_new(ledger / "complete.json", snapshot(root, capacity, groups))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "resume"))
    parser.add_argument("--root", type=Path, default=build.DEFAULT_ROOT)
    parser.add_argument(
        "--capacity", type=int, choices=build.CAPACITIES[1:], required=True
    )
    parser.add_argument(
        "--group", nargs="+", choices=build.GROUPS, default=list(build.GROUPS)
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--poll-seconds", type=int, choices=range(1, 61), default=30)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    groups = list(dict.fromkeys(args.group))
    if args.stage == "inspect":
        print(json.dumps(snapshot(root, args.capacity, groups), indent=2))
    else:
        resume(root, args.capacity, groups, args.workers, args.poll_seconds)


if __name__ == "__main__":
    main()
