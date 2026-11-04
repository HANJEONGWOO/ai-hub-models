# SPDX-License-Identifier: BSD-3-Clause
"""Only simulated device storage is mutated by these safety regression tests."""

import copy
import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def helper(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("niah_device_copies")


class FakeDevice:
    def __init__(self, files: list[dict], bundle: str) -> None:
        self.files = {item["remote"]: (item["bytes"], item["sha256"]) for item in files}
        self.bundle = bundle
        self.exists = True
        self.busy = False
        self.deleted = []
        self.pushed = []
        self.fail_delete = None

    def idle(self) -> None:
        if self.busy:
            raise RuntimeError("Runner active")

    def inventory(self, bundle: str) -> set[str] | None:
        assert bundle == self.bundle
        if not self.exists:
            return None
        return {p for p in self.files if str(Path(p).parent) == bundle}

    def file_info(self, remote: str) -> tuple[int, str]:
        return self.files[remote]

    def remove(self, remote: str) -> None:
        if remote == self.fail_delete:
            raise RuntimeError("Simulated disconnect")
        del self.files[remote]
        self.deleted.append(remote)

    def ensure_bundle(self, bundle: str) -> None:
        assert bundle == self.bundle
        self.exists = True

    def push(self, local: Path, remote: str) -> None:
        assert remote not in self.files
        import hashlib

        self.files[remote] = (
            local.stat().st_size,
            hashlib.sha256(local.read_bytes()).hexdigest(),
        )
        self.pushed.append(remote)


State = tuple[Path, Path, dict, FakeDevice]


@pytest.fixture
def state(tmp_path: Path, helper: ModuleType) -> State:
    root = tmp_path / "niah_unit"
    root.mkdir()
    (root / "protocol.json").write_text(
        json.dumps({"context_capacities": [1024, 4096]})
    )
    (root / "deployments").mkdir()
    name = "niah_unit_cl1024_fp16"
    bundle = helper.DEVICE_ROOT + "/bundles/" + name
    remotes = [f"{bundle}/part{i}_of_4.bin" for i in range(1, 5)]
    remotes += [
        bundle + "/runtime_manifest.json",
        helper.DEVICE_ROOT + "/bin/qnn-llm-runner",
        helper.DEVICE_ROOT + "/assets/cl1024_0123456789abcdef/prompt.bin",
    ]
    files = []
    for i, remote in enumerate(remotes):
        local = root / f"local{i}.bin"
        local.write_bytes(f"frozen file {i}".encode())
        files.append(
            {
                "local": str(local),
                "remote": remote,
                "sha256": helper.digest(local),
                "bytes": local.stat().st_size,
            }
        )
    manifest = root / "deployments/cl1024_fp16.json"
    data = {
        "name": name,
        "capacity": 1024,
        "group": "fp16",
        "files": files,
        "device": {"serial": "R5KYB0654BW"},
    }
    manifest.write_text(json.dumps(data))
    return root, manifest, data, FakeDevice(files, bundle)


def test_release_restore_roundtrip_preserves_shared_and_manifest(
    helper: ModuleType, state: State
) -> None:
    root, path, data, device = state
    original = path.read_bytes()
    shared = {
        p: info
        for p, info in device.files.items()
        if not p.startswith(device.bundle + "/")
    }
    release = helper.manage(root, 1024, "fp16", "release", device)
    assert len(device.deleted) == 5
    assert device.files == shared
    assert device.exists  # Directory is intentionally left empty.
    assert path.read_bytes() == original
    assert all(Path(f["local"]).exists() for f in data["files"])
    records = [json.loads(line) for line in release.read_text().splitlines()]
    assert records[0]["event"] == "verified_plan"
    assert records[-1]["event"] == "complete"
    assert records[-1]["recoverable_from_local"]
    restore = helper.manage(root, 1024, "fp16", "restore", device)
    assert restore != release
    assert len(device.pushed) == 5
    assert path.read_bytes() == original
    helper.verify_remote(device, data["files"])
    helper.manage(root, 1024, "fp16", "restore", device)
    assert len(device.pushed) == 5  # Matching files never overwritten.


@pytest.mark.parametrize(
    "kind",
    [
        "unknown",
        "busy",
        "local_size",
        "local_hash",
        "remote_hash",
        "shared_hash",
        "missing_bundle_file",
    ],
)
def test_release_rejects_unsafe_preflight_without_deletion(
    helper: ModuleType, state: State, kind: str
) -> None:
    root, _, data, device = state
    item = data["files"][0]
    if kind == "unknown":
        device.files[device.bundle + "/untracked.bin"] = (1, "a" * 64)
    elif kind == "busy":
        device.busy = True
    elif kind == "local_size":
        Path(item["local"]).write_bytes(b"x")
    elif kind == "local_hash":
        Path(item["local"]).write_bytes(b"x" * item["bytes"])
    elif kind == "remote_hash":
        device.files[item["remote"]] = (item["bytes"], "a" * 64)
    elif kind == "shared_hash":
        device.files[data["files"][-1]["remote"]] = (1, "a" * 64)
    else:
        del device.files[item["remote"]]
    with pytest.raises((ValueError, RuntimeError)):
        helper.manage(root, 1024, "fp16", "release", device)
    assert not device.deleted
    assert not device.pushed


def test_restore_partial_missing_and_absent_directory(
    helper: ModuleType, state: State
) -> None:
    root, _, data, device = state
    targets = [i for i in data["files"] if i["remote"].startswith(device.bundle + "/")]
    del device.files[targets[0]["remote"]]
    helper.manage(root, 1024, "fp16", "restore", device)
    assert device.pushed == [targets[0]["remote"]]
    helper.manage(root, 1024, "fp16", "release", device)
    device.exists = False
    helper.manage(root, 1024, "fp16", "restore", device)
    assert device.exists
    helper.verify_remote(device, data["files"])


def test_restore_never_overwrites_mismatched_existing_file(
    helper: ModuleType, state: State
) -> None:
    root, _, data, device = state
    remote = data["files"][0]["remote"]
    device.files[remote] = (123, "0" * 64)
    before = copy.deepcopy(device.files)
    with pytest.raises(ValueError, match="Device size/hash mismatch"):
        helper.manage(root, 1024, "fp16", "restore", device)
    assert device.files == before
    assert not device.pushed


def test_release_if_present_only_accepts_fully_absent_or_empty(
    helper: ModuleType, state: State
) -> None:
    root, _, data, device = state
    helper.manage(root, 1024, "fp16", "release", device)
    for exists in (True, False):
        device.exists = exists
        audit = helper.manage(root, 1024, "fp16", "release", device, if_present=True)
        records = [json.loads(line) for line in audit.read_text().splitlines()]
        assert records[-1]["event"] == "already_released"
    assert len(device.deleted) == 5
    device.exists = True
    item = data["files"][0]
    device.files[item["remote"]] = (item["bytes"], item["sha256"])
    with pytest.raises(ValueError, match="every original bundle"):
        helper.manage(root, 1024, "fp16", "release", device, if_present=True)
    with pytest.raises(ValueError, match="only valid for release"):
        helper.manage(root, 1024, "fp16", "restore", device, if_present=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "dotdot",
        "other_bundle",
        "nested_bundle",
        "wrong_identity",
        "wrong_serial",
        "missing_part",
        "unsafe_native",
    ],
)
def test_manifest_rejects_unsafe_targets(
    helper: ModuleType, state: State, mutation: str
) -> None:
    root, path, data, device = state
    if mutation == "duplicate":
        data["files"].append(copy.deepcopy(data["files"][0]))
    elif mutation == "dotdot":
        data["files"][0]["remote"] = device.bundle + "/../old/part1.bin"
    elif mutation == "other_bundle":
        data["files"][0]["remote"] = helper.DEVICE_ROOT + "/bundles/old/part1.bin"
    elif mutation == "nested_bundle":
        data["files"][0]["remote"] = device.bundle + "/nested/part1.bin"
    elif mutation == "wrong_identity":
        data["name"] = "old_experiment"
    elif mutation == "wrong_serial":
        data["device"]["serial"] = "OTHER"
    elif mutation == "missing_part":
        data["files"].pop(0)
    else:
        data["native_decoder"] = {"library": "../../evil.so"}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r"Unsafe|Unexpected|Deployment"):
        helper.manage(root, 1024, "fp16", "release", device)
    assert not device.deleted


def test_partial_deletion_is_audited_and_restorable(
    helper: ModuleType, state: State
) -> None:
    root, path, data, device = state
    before = path.read_bytes()
    device.fail_delete = data["files"][1]["remote"]
    with pytest.raises(RuntimeError, match="Simulated disconnect"):
        helper.manage(root, 1024, "fp16", "release", device)
    assert len(device.deleted) == 1
    audit = next((root / "device_copy_audits").glob("*.jsonl"))
    events = [json.loads(line) for line in audit.read_text().splitlines()]
    assert events[-1]["event"] == "error_partial_state_preserved"
    assert events[1]["event"] == "deleted_recoverable_copy"
    helper.manage(root, 1024, "fp16", "restore", device)
    helper.verify_remote(device, data["files"])
    assert path.read_bytes() == before


def test_device_remove_is_exact_nonrecursive_and_quoted(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = helper.AdbDevice(Path("adb"), "SERIAL")
    calls = []
    monkeypatch.setattr(device, "command", lambda *args: calls.append(args) or "")
    device.remove("/data/local/tmp/qaihm_turboquant/llm/bundles/niah_test/part1.bin")
    assert calls == [
        (
            "shell",
            "rm -- /data/local/tmp/qaihm_turboquant/llm/bundles/niah_test/part1.bin",
        )
    ]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("__MISSING__", None),
        ("__DIRECTORY__", set()),
        ("__DIRECTORY__\n/a/b/file", {"/a/b/file"}),
    ],
)
def test_inventory_parser(
    helper: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    output: str,
    expected: set[str] | None,
) -> None:
    device = helper.AdbDevice(Path("adb"), "SERIAL")
    monkeypatch.setattr(device, "command", lambda *args: output)
    assert device.inventory("/a/b") == expected


def test_file_info_rejects_wrong_remote(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = helper.AdbDevice(Path("adb"), "SERIAL")
    monkeypatch.setattr(device, "command", lambda *args: "10\n" + "a" * 64 + "  /wrong")
    with pytest.raises(RuntimeError, match="Invalid file digest"):
        device.file_info("/expected")
