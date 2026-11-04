# SPDX-License-Identifier: BSD-3-Clause
"""2K admission retains the original exact-file copy-safety checks."""

import importlib
import json
from pathlib import Path

import pytest

from qai_hub_models.test.test_models.test_turboquant_niah_device_copies import FakeDevice


@pytest.fixture
def helper(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("niah_2k_device_copies")


@pytest.fixture
def state(tmp_path, helper):
    root = tmp_path / helper.SIDECAR_NAME
    (root / "deployments").mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps({"context_capacities": [2048]}))
    name = root.name + "_cl2048_fp16"
    bundle = helper.DEVICE_ROOT + "/bundles/" + name
    remotes = [f"{bundle}/part{i}_of_4.bin" for i in range(1, 5)]
    remotes += [bundle + "/runtime_manifest.json", helper.DEVICE_ROOT + "/bin/qnn-llm-runner"]
    files = []
    for i, remote in enumerate(remotes):
        local = root / f"backup{i}.bin"
        local.write_bytes(f"preserved {i}".encode())
        files.append({"remote": remote, "local": str(local), "bytes": local.stat().st_size,
                      "sha256": helper.digest(local)})
    data = {"name": name, "capacity": 2048, "group": "fp16", "files": files,
            "device": {"serial": "R5KYB0654BW"}}
    path = root / "deployments/cl2048_fp16.json"
    path.write_text(json.dumps(data))
    return root, path, data, FakeDevice(files, bundle)


def test_only_private_admission_changes(helper, state):
    root, _, _, _ = state
    assert helper.manifest(root, 2048, "fp16")[1]["capacity"] == 2048
    with pytest.raises(ValueError, match="Unknown NIAH"):
        helper.original.manifest(root, 2048, "fp16")
    assert helper.original.manifest is not helper.helper.manifest
    assert helper.manage.__code__.co_code == helper.original.manage.__code__.co_code


def test_release_restore_preserves_source_shared_and_manifest(helper, state):
    root, path, data, device = state
    before = path.read_bytes()
    shared = device.files[helper.DEVICE_ROOT + "/bin/qnn-llm-runner"]
    audit = helper.manage(root, 2048, "fp16", "release", device)
    assert len(device.deleted) == 5
    assert device.files == {helper.DEVICE_ROOT + "/bin/qnn-llm-runner": shared}
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert records[0]["helper_sha256"] == helper.digest(Path(helper.__file__))
    assert records[-1]["recoverable_from_local"]
    helper.manage(root, 2048, "fp16", "restore", device)
    assert len(device.pushed) == 5
    assert path.read_bytes() == before
    assert all(Path(item["local"]).exists() for item in data["files"])


@pytest.mark.parametrize("capacity", [1024, 4096, 8192, 16384])
def test_other_capacities_rejected(helper, state, capacity):
    with pytest.raises(ValueError):
        helper.manifest(state[0], capacity, "fp16")


@pytest.mark.parametrize("kind", ["unknown", "busy", "local_hash", "remote_hash", "shared_hash", "missing"])
def test_unsafe_state_prevents_deletion(helper, state, kind):
    root, _, data, device = state
    item = data["files"][0]
    if kind == "unknown":
        device.files[device.bundle + "/unknown.bin"] = (1, "a" * 64)
    elif kind == "busy":
        device.busy = True
    elif kind == "local_hash":
        Path(item["local"]).write_bytes(b"x" * item["bytes"])
    elif kind == "remote_hash":
        device.files[item["remote"]] = (item["bytes"], "a" * 64)
    elif kind == "shared_hash":
        device.files[data["files"][-1]["remote"]] = (1, "a" * 64)
    else:
        del device.files[item["remote"]]
    with pytest.raises((ValueError, RuntimeError)):
        helper.manage(root, 2048, "fp16", "release", device)
    assert not device.deleted and not device.pushed


@pytest.mark.parametrize("kind", ["path", "duplicate", "serial", "identity", "protocol"])
def test_manifest_tampering_rejected(helper, state, kind):
    root, path, data, device = state
    if kind == "path":
        data["files"][0]["remote"] = "/data/local/tmp/other.bin"
    elif kind == "duplicate":
        data["files"].append(data["files"][0])
    elif kind == "serial":
        data["device"]["serial"] = "other"
    elif kind == "identity":
        data["capacity"] = 8192
    else:
        (root / "protocol.json").write_text(json.dumps({"context_capacities": [2048, 16384]}))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        helper.manage(root, 2048, "fp16", "release", device)
    assert not device.deleted
