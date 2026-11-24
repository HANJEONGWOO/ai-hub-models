# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Host-only safety checks for execute-only Native decoder replacements."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts/llm/turboquant"


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("run_device_llm")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def packages(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    sdk = tmp_path / "qairt"
    sdk.mkdir()

    def create(name: str, execute: bytes) -> tuple[dict[str, Any], Path]:
        root = tmp_path / name
        root.mkdir()
        manifest: dict[str, Any] = {
            "package": "TurboQuantNative",
            "interface": "TurboQuantInterfaceProvider",
            "operations": [f"Decode{bits}" for bits in (2, 3, 4, 5, 6)],
            "qairt_sdk": str(sdk),
            "source_sha256": digest(b"unchanged decoder.cpp"),
            "source_files": {
                "decoder.cpp": digest(b"unchanged decoder.cpp"),
                "Decode4.xml": digest(b"unchanged operation ABI"),
                "hvx_decode.h": digest(execute),
            },
            "libraries": {},
        }
        for target, data in (
            ("x86_64-linux-clang", b"unchanged prepare library"),
            ("hexagon-v81", execute),
        ):
            directory = root / target
            directory.mkdir()
            library = directory / "libTurboQuantNative.so"
            library.write_bytes(data)
            manifest["libraries"][target] = {
                "path": str(library),
                "sha256": digest(data),
            }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return manifest, path

    compiled, _ = create("compiled", b"original execute kernel")
    _, override = create("override", b"optimized execute kernel")
    return compiled, override, sdk


def test_native_runtime_default_keeps_compiled_package(
    runner: ModuleType, packages: tuple[dict[str, Any], Path, Path]
) -> None:
    compiled, _, sdk = packages
    selected, provenance = runner.native_runtime_package(compiled, None, sdk)
    assert selected == compiled
    assert provenance == {}


def test_native_runtime_accepts_execute_only_change_with_provenance(
    runner: ModuleType, packages: tuple[dict[str, Any], Path, Path]
) -> None:
    compiled, override, sdk = packages
    original = json.loads(json.dumps(compiled))
    selected, provenance = runner.native_runtime_package(compiled, override.parent, sdk)
    assert selected == json.loads(override.read_text())
    assert compiled == original
    assert (
        selected["source_files"]["hvx_decode.h"]
        != compiled["source_files"]["hvx_decode.h"]
    )
    assert provenance["compiled_native_decoder"] == original
    audit = provenance["native_runtime_override"]
    assert audit["manifest"] == str(override.resolve())
    assert audit["manifest_sha256"] == digest(override.read_bytes())
    assert isinstance(audit["abi_checks"], list)
    assert audit["abi_checks"]
    assert isinstance(audit["scope"], str) and audit["scope"]


@pytest.mark.parametrize("compiled", [None, {}])
def test_native_runtime_rejects_override_without_compiled_package(
    runner: ModuleType,
    packages: tuple[dict[str, Any], Path, Path],
    compiled: dict[str, Any] | None,
) -> None:
    _, override, sdk = packages
    with pytest.raises(ValueError, match="requires a compiled Native bundle"):
        runner.native_runtime_package(compiled, override.parent, sdk)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("package", "DifferentPackage"),
        ("interface", "DifferentInterfaceProvider"),
        ("operations", ["Decode6", "Decode5", "Decode4", "Decode3", "Decode2"]),
        ("operations", ["Decode4"]),
        ("qairt_sdk", "/different/qairt"),
    ],
)
def test_native_runtime_rejects_manifest_abi_changes(
    runner: ModuleType,
    packages: tuple[dict[str, Any], Path, Path],
    field: str,
    value: Any,
) -> None:
    compiled, override, sdk = packages
    changed = json.loads(override.read_text())
    changed[field] = value
    override.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match=f"ABI mismatch: {field}"):
        runner.native_runtime_package(compiled, override.parent, sdk)


@pytest.mark.parametrize("source", ["decoder.cpp", "Decode4.xml"])
def test_native_runtime_rejects_prepare_or_xml_source_changes(
    runner: ModuleType,
    packages: tuple[dict[str, Any], Path, Path],
    source: str,
) -> None:
    compiled, override, sdk = packages
    changed = json.loads(override.read_text())
    changed["source_files"][source] = digest(b"changed source")
    override.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="registration mismatch"):
        runner.native_runtime_package(compiled, override.parent, sdk)


def test_native_runtime_rejects_changed_prepare_library_even_with_valid_hash(
    runner: ModuleType, packages: tuple[dict[str, Any], Path, Path]
) -> None:
    compiled, override, sdk = packages
    changed = json.loads(override.read_text())
    library = changed["libraries"]["x86_64-linux-clang"]
    Path(library["path"]).write_bytes(b"different prepare implementation")
    library["sha256"] = digest(b"different prepare implementation")
    override.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed the x86 prepare library"):
        runner.native_runtime_package(compiled, override.parent, sdk)


@pytest.mark.parametrize(
    ("target", "use_override"),
    [("hexagon-v81", False), ("hexagon-v81", True), ("x86_64-linux-clang", True)],
)
def test_native_runtime_rejects_library_contents_not_matching_manifest(
    runner: ModuleType,
    packages: tuple[dict[str, Any], Path, Path],
    target: str,
    use_override: bool,
) -> None:
    compiled, override, sdk = packages
    manifest = json.loads(override.read_text()) if use_override else compiled
    Path(manifest["libraries"][target]["path"]).write_bytes(b"corrupted binary")
    with pytest.raises(ValueError, match="library changed since its manifest"):
        runner.native_runtime_package(
            compiled, override.parent if use_override else None, sdk
        )


def test_native_runtime_rejects_different_runtime_sdk(
    runner: ModuleType,
    packages: tuple[dict[str, Any], Path, Path],
) -> None:
    compiled, override, sdk = packages
    with pytest.raises(ValueError, match="SDK differs from the active SDK"):
        runner.native_runtime_package(
            compiled, override.parent, sdk.parent / "other-sdk"
        )


@pytest.mark.parametrize("compiled", [None, {}])
def test_native_runtime_without_native_has_no_override(
    runner: ModuleType, tmp_path: Path, compiled: dict[str, Any] | None
) -> None:
    assert runner.native_runtime_package(compiled, None, tmp_path) == (None, {})


def test_native_runtime_preserves_legacy_default_without_new_abi_metadata(
    runner: ModuleType, packages: tuple[dict[str, Any], Path, Path]
) -> None:
    compiled, _, sdk = packages
    del compiled["operations"]
    del compiled["source_files"]
    assert runner.native_runtime_package(compiled, None, sdk) == (compiled, {})


@pytest.mark.parametrize("prepare_state", ["missing-file", "missing-entry", "corrupt"])
def test_native_runtime_default_does_not_require_prepare_library(
    runner: ModuleType,
    packages: tuple[dict[str, Any], Path, Path],
    prepare_state: str,
) -> None:
    compiled, _, sdk = packages
    prepare = compiled["libraries"]["x86_64-linux-clang"]
    if prepare_state == "missing-file":
        prepare["path"] = str(Path(prepare["path"]).with_name("unavailable.so"))
    elif prepare_state == "missing-entry":
        del compiled["libraries"]["x86_64-linux-clang"]
    else:
        Path(prepare["path"]).write_bytes(b"prepare not needed at runtime")
    assert runner.native_runtime_package(compiled, None, sdk) == (compiled, {})


@pytest.mark.parametrize("missing_from", ["compiled", "override"])
def test_native_runtime_override_requires_prepare_files(
    runner: ModuleType,
    packages: tuple[dict[str, Any], Path, Path],
    missing_from: str,
) -> None:
    compiled, override, sdk = packages
    changed = (
        compiled if missing_from == "compiled" else json.loads(override.read_text())
    )
    prepare = changed["libraries"]["x86_64-linux-clang"]
    prepare["path"] = str(Path(prepare["path"]).with_name("unavailable.so"))
    if missing_from == "override":
        override.write_text(json.dumps(changed))
    with pytest.raises(FileNotFoundError):
        runner.native_runtime_package(compiled, override.parent, sdk)


@pytest.mark.parametrize("field", ["operations", "source_files"])
def test_native_runtime_override_requires_recorded_abi_metadata(
    runner: ModuleType,
    packages: tuple[dict[str, Any], Path, Path],
    field: str,
) -> None:
    compiled, override, sdk = packages
    del compiled[field]
    with pytest.raises(ValueError, match=r"Native runtime.*mismatch"):
        runner.native_runtime_package(compiled, override.parent, sdk)


@pytest.mark.parametrize("target", ["x86_64-linux-clang", "hexagon-v81"])
def test_native_runtime_override_still_verifies_original_compiled_libraries(
    runner: ModuleType,
    packages: tuple[dict[str, Any], Path, Path],
    target: str,
) -> None:
    compiled, override, sdk = packages
    Path(compiled["libraries"][target]["path"]).write_bytes(b"corrupted original")
    with pytest.raises(ValueError, match="library changed since its manifest"):
        runner.native_runtime_package(compiled, override.parent, sdk)
