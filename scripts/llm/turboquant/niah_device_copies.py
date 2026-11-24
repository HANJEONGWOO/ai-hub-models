# SPDX-License-Identifier: BSD-3-Clause
"""Release/restore only hash-verified copies of this NIAH experiment's bundles.

This storage helper does not modify a deployment manifest, model, shared runtime,
input, result or benchmark script. Every deletion has a verified PC restore path.
Only use while the experiment runner is stopped; no automatic retries are made.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

DEVICE_ROOT = "/data/local/tmp/qaihm_turboquant/llm"
GROUPS = ("fp16", "k4_v4", "boundary_lm", "boundary_structured")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def manifest(root: Path, capacity: int, group: str) -> tuple[Path, dict, str]:
    if not re.fullmatch(r"niah_[A-Za-z0-9_.-]+", root.name):
        raise ValueError("Only a named niah_ experiment may be managed")
    if group not in GROUPS or capacity not in (1024, 4096, 8192, 16384):
        raise ValueError("Unknown NIAH capacity/profile")
    if capacity not in read(root / "protocol.json")["context_capacities"]:
        raise ValueError("Capacity is absent from the frozen protocol")
    path = root / "deployments" / f"cl{capacity}_{group}.json"
    data = read(path)
    name = f"{root.name}_cl{capacity}_{group}"
    if (data["name"], data["capacity"], data["group"]) != (name, capacity, group):
        raise ValueError("Deployment identity does not match this experiment")
    bundle = f"{DEVICE_ROOT}/bundles/{name}"
    files = data["files"]
    seen = set()
    for item in files:
        remote, local = item["remote"], item["local"]
        p = PurePosixPath(remote)
        if (
            not p.is_absolute()
            or str(p) != remote
            or any(part in (".", "..") for part in p.parts)
            or not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote)
            or remote in seen
        ):
            raise ValueError(f"Unsafe or duplicate remote path: {remote!r}")
        seen.add(remote)
        if not (
            p.parent == PurePosixPath(bundle)
            or p.parent == PurePosixPath(f"{DEVICE_ROOT}/bin")
            or (
                p.parent.parent == PurePosixPath(f"{DEVICE_ROOT}/assets")
                and re.fullmatch(r"cl\d+_[0-9a-f]{16}", p.parent.name)
            )
        ):
            raise ValueError(f"Unexpected deployment path: {remote}")
        if (
            not Path(local).is_absolute()
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            or type(item["bytes"]) is not int
            or item["bytes"] < 1
        ):
            raise ValueError("Invalid local backup metadata")
    selected = [
        item
        for item in files
        if PurePosixPath(item["remote"]).parent == PurePosixPath(bundle)
    ]
    names = {PurePosixPath(item["remote"]).name for item in selected}
    required = {f"part{i}_of_4.bin" for i in range(1, 5)} | {"runtime_manifest.json"}
    if data.get("native_decoder"):
        native = data["native_decoder"]["library"]
        if not re.fullmatch(r"lib[A-Za-z0-9_.-]+\.so", native):
            raise ValueError("Unsafe Native library filename")
        required.add(native)
    if names != required:
        raise ValueError("Deployment does not list the exact expected bundle files")
    if data.get("device", {}).get("serial") != "R5KYB0654BW":
        raise ValueError("Deployment is not for the frozen NIAH device")
    return path, data, bundle


class AdbDevice:
    def __init__(self, adb: Path, serial: str) -> None:
        self.adb = adb
        self.serial = serial

    def command(self, *args: str) -> str:
        result = subprocess.run(
            [str(self.adb), "-s", self.serial, *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def idle(self) -> None:
        output = self.command(
            "shell", "if pidof qnn-llm-runner; then true; else echo __IDLE__; fi"
        )
        if output != "__IDLE__":
            raise RuntimeError(f"Runner active or device unavailable: {output!r}")

    def inventory(self, bundle: str) -> set[str] | None:
        quoted = shlex.quote(bundle)
        # Canonical path rejects symbolic links at any ancestor as well as at
        # the bundle itself. find also reveals unknown files, dirs and links.
        script = (
            f"if [ -L {quoted} ]; then exit 70; fi; "
            f"if [ ! -e {quoted} ]; then echo __MISSING__; exit 0; fi; "
            f'[ -d {quoted} ] && [ "$(readlink -f {quoted})" = {quoted} ] || exit 71; '
            f"echo __DIRECTORY__; find {quoted} -mindepth 1 -maxdepth 1 -print"
        )
        lines = self.command("shell", script).splitlines()
        if lines == ["__MISSING__"]:
            return None
        if (
            not lines
            or lines[0] != "__DIRECTORY__"
            or len(lines[1:]) != len(set(lines[1:]))
        ):
            raise RuntimeError("Invalid device inventory")
        return set(lines[1:])

    def file_info(self, remote: str) -> tuple[int, str]:
        quoted = shlex.quote(remote)
        output = self.command(
            "shell",
            f"[ -f {quoted} ] && [ ! -L {quoted} ] && "
            f'[ "$(readlink -f {quoted})" = {quoted} ] || exit 72; '
            f"stat -c %s {quoted}; sha256sum {quoted}",
        ).splitlines()
        if len(output) != 2 or not output[0].isdigit():
            raise RuntimeError(f"Invalid file metadata: {remote}")
        fields = output[1].split(maxsplit=1)
        if (
            len(fields) != 2
            or fields[1].lstrip("*") != remote
            or not re.fullmatch(r"[0-9a-f]{64}", fields[0])
        ):
            raise RuntimeError(f"Invalid file digest: {remote}")
        return int(output[0]), fields[0]

    def remove(self, remote: str) -> None:
        self.command("shell", "rm -- " + shlex.quote(remote))

    def ensure_bundle(self, bundle: str) -> None:
        # Never create/repoint the shared parent. A canonical existing parent
        # is required even if the leaf was removed by an external operator.
        parent = shlex.quote(str(PurePosixPath(bundle).parent))
        leaf = shlex.quote(bundle)
        self.command(
            "shell",
            f'[ -d {parent} ] && [ "$(readlink -f {parent})" = {parent} ] || exit 73; '
            f"if [ ! -e {leaf} ]; then mkdir -- {leaf}; fi",
        )
        self.inventory(bundle)

    def push(self, local: Path, remote: str) -> None:
        self.command(
            "shell", f"[ ! -e {shlex.quote(remote)} ] && [ ! -L {shlex.quote(remote)} ]"
        )
        if self.adb.suffix.lower() == ".exe":
            source = subprocess.run(
                ["wslpath", "-w", str(local.resolve(strict=True))],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        else:
            source = str(local.resolve(strict=True))
        self.command("push", source, remote)


def verify_local(files: list[dict]) -> None:
    for item in files:
        path = Path(item["local"])
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or digest(path) != item["sha256"]
        ):
            raise ValueError(f"Local backup size/hash mismatch: {path}")


def verify_remote(device: Any, files: list[dict]) -> None:
    for item in files:
        if device.file_info(item["remote"]) != (item["bytes"], item["sha256"]):
            raise ValueError(f"Device size/hash mismatch: {item['remote']}")


def manage(
    root: Path,
    capacity: int,
    group: str,
    action: str,
    device: Any,
    *,
    if_present: bool = False,
) -> Path:
    if action not in ("release", "restore"):
        raise ValueError("Unknown copy-management action")
    if if_present and action != "release":
        raise ValueError("--if-present is only valid for release")
    path, data, bundle = manifest(root, capacity, group)
    expected_hash = digest(path)
    files = data["files"]
    selected = [
        item
        for item in files
        if PurePosixPath(item["remote"]).parent == PurePosixPath(bundle)
    ]
    targets = {item["remote"] for item in selected}
    audit_root = root / "device_copy_audits"
    audit_root.mkdir(exist_ok=True)
    # Single-host serialization is scoped to this experiment, not systemwide.
    with (audit_root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_local(files)
        device.idle()
        inventory = device.inventory(bundle)
        present = inventory or set()
        if not present <= targets:
            raise ValueError(
                f"Unknown entries in bundle; untouched: {sorted(present - targets)}"
            )
        already_released = action == "release" and if_present and not present
        if action == "release" and present != targets and not already_released:
            raise ValueError(
                "Release requires every original bundle file to be present"
            )
        verify_remote(
            device,
            [
                item
                for item in files
                if item["remote"] not in targets or item["remote"] in present
            ],
        )
        if digest(path) != expected_hash:
            raise ValueError("Deployment changed during verification")
        audit = (
            audit_root
            / f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{action}_cl{capacity}_{group}_{uuid.uuid4().hex[:8]}.jsonl"
        )
        with audit.open("x") as stream:

            def event(kind: str, **values: Any) -> None:
                stream.write(
                    json.dumps({"utc": now(), "event": kind, **values}, sort_keys=True)
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())

            event(
                "verified_plan",
                action=action,
                deployment=str(path),
                deployment_sha256=expected_hash,
                helper_sha256=digest(Path(__file__)),
                bundle=bundle,
                files=selected,
                shared_files_untouched=len(files) - len(selected),
                initially_present=sorted(present),
            )
            try:
                if already_released:
                    event(
                        "already_released",
                        deployment_unchanged=True,
                        recoverable_from_local=True,
                        managed_bytes=0,
                    )
                    return audit
                if action == "restore":
                    device.idle()
                    device.ensure_bundle(bundle)
                for item in selected:
                    remote = item["remote"]
                    device.idle()
                    if action == "release":
                        # Recheck each file immediately before its exact,
                        # nonrecursive deletion; the verified host is retained.
                        verify_local([item])
                        verify_remote(device, [item])
                        device.remove(remote)
                        event("deleted_recoverable_copy", **item)
                    elif remote not in present:
                        verify_local([item])
                        device.push(Path(item["local"]), remote)
                        verify_remote(device, [item])
                        event("restored_verified_copy", **item)
                    else:
                        event("kept_matching_copy", **item)
                remaining = device.inventory(bundle)
                if action == "release" and remaining != set():
                    raise RuntimeError("Bundle not empty after exact-file release")  # noqa: TRY301 -- Log final-state failures too.
                if action == "restore":
                    if remaining != targets:
                        raise RuntimeError("Unexpected restored bundle inventory")  # noqa: TRY301
                    verify_remote(device, files)
                if digest(path) != expected_hash:
                    raise RuntimeError("Deployment manifest changed")  # noqa: TRY301
                event(
                    "complete",
                    recoverable_from_local=True,
                    deployment_unchanged=True,
                    managed_bytes=sum(item["bytes"] for item in selected),
                )
            except BaseException as error:
                event("error_partial_state_preserved", error=repr(error))
                raise
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("release", "restore"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--adb", type=Path, default=Path("/mnt/c/adb/adb.exe"))
    parser.add_argument(
        "--if-present",
        action="store_true",
        help="Release only: accept an already empty/absent bundle; partial or unknown files still fail",
    )
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    _, data, _ = manifest(root, args.capacity, args.group)
    device = AdbDevice(args.adb, data["device"]["serial"])
    audit = manage(
        root, args.capacity, args.group, args.action, device, if_present=args.if_present
    )
    print(
        json.dumps(
            {
                "action": args.action,
                "audit": str(audit),
                "local_backups_preserved": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
