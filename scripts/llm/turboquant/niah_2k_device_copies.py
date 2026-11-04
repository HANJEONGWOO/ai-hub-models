# SPDX-License-Identifier: BSD-3-Clause
"""Admit only the authorized 2K sidecar to the original copy-safety workflow.

The manifest checks below mirror niah_device_copies.manifest, with admission
restricted to CL2048 and its specific sidecar name. A private module instance
reuses the original idle checks, hashes, exact-file release/restore and logs.
Neither the original module nor its frozen source is modified.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path, PurePosixPath

import niah_device_copies as original

SIDECAR_NAME = "niah_supplement_cl2048_20261103"
DEVICE_ROOT = original.DEVICE_ROOT
GROUPS = original.GROUPS
digest = original.digest
read = original.read


def manifest(root: Path, capacity: int, group: str) -> tuple[Path, dict, str]:
    if root.name != SIDECAR_NAME or root != root.resolve(strict=True):
        raise ValueError("Only the canonical authorized 2K sidecar may be managed")
    if capacity != 2048 or group not in GROUPS:
        raise ValueError("Only the four frozen profiles at CL2048 are admitted")
    if read(root / "protocol.json")["context_capacities"] != [2048]:
        raise ValueError("Sidecar protocol must authorize only CL2048")
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
        item for item in files
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


def private_helper():
    spec = importlib.util.spec_from_file_location(
        "_niah_2k_private_device_copies", original.__file__
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the frozen copy-safety helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.manifest = manifest
    # The audit identifies this admission adapter; the parent scope also freezes
    # the unchanged original implementation's SHA-256.
    module.__file__ = __file__
    return module


helper = private_helper()
manage = helper.manage
AdbDevice = helper.AdbDevice

if __name__ == "__main__":
    helper.main()
