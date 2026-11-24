# SPDX-License-Identifier: BSD-3-Clause
"""Offline 2K sidecar invariants; no QNN conversion, device or external writes."""

import copy
import importlib
import json
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from .test_turboquant_niah_data import CharacterTokenizer


@pytest.fixture
def supplement(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/llm/turboquant")
    )
    return importlib.import_module("niah_2k_supplement")


@pytest.fixture
def original(tmp_path: Path, supplement: ModuleType) -> tuple[Path, dict, dict]:
    root = tmp_path / "niah_original"
    (root / "data/backgrounds").mkdir(parents=True)
    bases = []
    for i, (base_id, split) in enumerate(
        [(f"test_{i:02d}", "test") for i in range(10)]
        + [("pilot_00", "pilot"), ("smoke_00", "smoke")]
    ):
        text = f"Natural document {i}. " + "A natural sentence about science and history. " * 100
        filename = f"backgrounds/{base_id}.txt"
        (root / "data" / filename).write_text(text)
        bases.append(
            {
                "base_id": base_id,
                "split": split,
                "seed": 9000 + i,
                "root_seed": 2026110201,
                "project": f"PROJECT{chr(65+i)}",
                "expected_answer": f"{420+i:06d}",
                "background_tokens": len(text),
                "background_sha256": supplement.data.text_digest(text),
                "source_documents": [{"title": f"Doc {i}", "sha256": str(i)}],
                "file": filename,
            }
        )
    source = root / "fixture.parquet"
    source.write_bytes(b"source fixture")
    tokenizer_file = root / "tokenizer.json"
    tokenizer_file.write_text("fixture")
    manifest = {
        "schema_version": 1,
        "benchmark": "single needle fixture",
        "source": {"path": str(source), "sha256": supplement.data.digest(source)},
        "tokenizer": {
            "checkpoint_dir": str(root),
            "files_sha256": {str(tokenizer_file): supplement.data.digest(tokenizer_file)},
            "chat_template_sha256": supplement.data.text_digest("fixture"),
            "eos_token_id": 42,
        },
        "protocol": {
            "capacities": supplement.ORIGINAL_CAPACITIES,
            "depths_percent": [0, 25, 50, 75, 100],
            "base_count": 10,
            "test_seed": 2026110201,
            "pilot_seed": 2026110202,
            "smoke_seed": 2026110203,
            "max_new_tokens": 64,
        },
        "base_examples": bases,
        "case_counts": {"test": 200, "pilot": 2, "smoke": 4},
        "cases": [],
    }
    protocol = {
        "context_capacities": supplement.ORIGINAL_CAPACITIES,
        "groups": list(supplement.GROUPS),
        "mask": supplement.BOUNDARY_MASK,
        "profiles": {g: {"config_hash": f"hash-{g}"} for g in supplement.GROUPS},
        "frozen_inputs": {},
        "max_generated_tokens": 64,
    }
    for name, value in (
        ("protocol.json", protocol),
        ("data/data_manifest.json", manifest),
        ("execution_freeze.json", {"files_sha256": {}}),
        ("boundary_layer_plan.json", {"fixture": "unchanged boundary mask"}),
    ):
        (root / name).write_text(json.dumps(value))
    return root, protocol, manifest


def test_derives_exact_original_bases_and_53_cases(
    supplement: ModuleType, original: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _, manifest = original
    before = {str(p): supplement.data.digest(p) for p in source.rglob("*") if p.is_file()}
    monkeypatch.setattr(supplement.data, "base_examples", lambda *a, **kw: pytest.fail("reselection"))
    result = supplement.derive_dataset(
        source / supplement.SIDECAR_NAME / "data", source, manifest, CharacterTokenizer()
    )
    assert result["case_counts"] == {"test": 50, "pilot": 2, "smoke": 1}
    assert result["base_examples"] == manifest["base_examples"]
    assert result["source"] == manifest["source"]
    assert result["tokenizer"] == manifest["tokenizer"]
    assert manifest["protocol"]["capacities"] == supplement.ORIGINAL_CAPACITIES
    assert result["protocol"]["capacities"] == [2048]
    assert {c["depth"] for c in result["cases"] if c["split"] == "test"} == {0, 25, 50, 75, 100}
    assert len({c["case_id"] for c in result["cases"]}) == 53
    for filename, expected in before.items():
        assert supplement.data.digest(Path(filename)) == expected
    for row in result["cases"]:
        target = source / supplement.SIDECAR_NAME / "data"
        raw = (target / row["tokens_file"]).read_bytes()
        prompt = (target / row["prompt_file"]).read_text()
        assert list(struct.unpack(f"<{row['input_tokens']}i", raw)) == CharacterTokenizer().encode(prompt)
        assert row["input_tokens"] + 64 <= 2048
        assert prompt.count(row["expected_answer"]) == 1
    with pytest.raises(FileExistsError):
        supplement.derive_dataset(source / supplement.SIDECAR_NAME / "data", source, manifest, CharacterTokenizer())


@pytest.mark.parametrize("changed", ["background", "tokenizer", "source"])
def test_source_tampering_fails(
    supplement: ModuleType, original: tuple, changed: str
) -> None:
    root, _, manifest = original
    path = {
        "background": root / "data" / manifest["base_examples"][0]["file"],
        "tokenizer": root / "tokenizer.json",
        "source": root / "fixture.parquet",
    }[changed]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        supplement.source_snapshot(root)


def test_known_mask_and_original_scope_required(supplement: ModuleType, original: tuple) -> None:
    root, protocol, _ = original
    protocol["mask"] = list(range(14))
    (root / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="scope"):
        supplement.source_snapshot(root)


def test_source_snapshot_freezes_all_dependencies(supplement: ModuleType, original: tuple) -> None:
    root, protocol, manifest = original
    actual_protocol, actual_manifest, hashes = supplement.source_snapshot(root)
    assert actual_protocol == protocol and actual_manifest == manifest
    assert str(root / "protocol.json") in hashes
    assert str(root / "execution_freeze.json") in hashes
    assert str(root / "data/data_manifest.json") in hashes
    assert str(root / "boundary_layer_plan.json") in hashes
    assert str(root / "tokenizer.json") in hashes
    assert str(root / "fixture.parquet") in hashes
    assert all(str(root / "data" / b["file"]) in hashes for b in manifest["base_examples"])


def test_rejects_wrong_or_symlink_sidecar(supplement: ModuleType, original: tuple, tmp_path: Path) -> None:
    source, _, _ = original
    with pytest.raises(ValueError, match="direct-child"):
        supplement.sidecar_path(source, source)
    with pytest.raises(ValueError, match="direct-child"):
        supplement.sidecar_path(tmp_path / supplement.SIDECAR_NAME, source)
    (source / supplement.SIDECAR_NAME).symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        supplement.sidecar_path(source / supplement.SIDECAR_NAME, source)


def test_private_module_changes_only_admission_without_shared_mutation(supplement: ModuleType) -> None:
    original = importlib.import_module("niah_build")
    before = tuple(original.CAPACITIES)
    sha = supplement.data.digest(Path(original.__file__))
    private = supplement.private_build()
    assert private is not original
    assert private.CAPACITIES == (2048,)
    assert original.CAPACITIES == before == tuple(supplement.ORIGINAL_CAPACITIES)
    assert private.convert_options.__code__.co_code == original.convert_options.__code__.co_code
    assert private.build.__code__.co_code == original.build.__code__.co_code
    assert private.audit.__code__.co_code == original.audit.__code__.co_code
    assert supplement.data.digest(Path(original.__file__)) == sha
    assert "_niah_build_cl2048_private" not in sys.modules


def test_prepare_and_verify_preserve_source_and_freeze_adapter(
    supplement: ModuleType, original: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _ = original
    target = root / supplement.SIDECAR_NAME
    fake_build = SimpleNamespace(
        config=lambda g: SimpleNamespace(config_hash=lambda: f"hash-{g}"),
        memory_estimate=lambda capacity, compressed: {"capacity": capacity, "compressed": compressed},
    )
    monkeypatch.setattr(supplement, "private_build", lambda: fake_build)
    tokenizer = CharacterTokenizer()
    tokenizer.is_fast, tokenizer.chat_template, tokenizer.eos_token_id = True, "fixture", 42
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)))
    supplement.prepare(target, root, "User approved 2K addition and 16K exclusion")
    provenance = supplement.verify_supplement(target)
    assert provenance["source_mutations"] == []
    assert provenance["main_cases_all_profiles"] == 800
    assert str(Path(supplement.__file__).resolve()) in provenance["source_files_sha256"]
    assert supplement.read(target / "protocol.json")["profiles"] == supplement.read(root / "protocol.json")["profiles"]
    assert (target / "boundary_layer_plan.json").read_bytes() == (root / "boundary_layer_plan.json").read_bytes()
    with pytest.raises(FileExistsError):
        supplement.prepare(target, root, "same approval")
    (target / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        supplement.verify_supplement(target)


def test_requires_authorization_before_writes(supplement: ModuleType, original: tuple) -> None:
    root, _, _ = original
    target = root / supplement.SIDECAR_NAME
    with pytest.raises(ValueError, match="authorization"):
        supplement.prepare(target, root, "  ")
    assert not target.exists()


@pytest.mark.parametrize("workers,groups", [(2, ["fp16"]), (1, []), (1, ["unknown"])])
def test_build_rejects_unsupported_work_before_loading(
    supplement: ModuleType, tmp_path: Path, workers: int, groups: list[str]
) -> None:
    with pytest.raises(ValueError, match="serial"):
        supplement.run_build(tmp_path, groups, workers)


@pytest.mark.parametrize("audit", [False, True])
def test_build_dispatch_reuses_original_implementation(
    supplement: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, audit: bool
) -> None:
    calls = []
    monkeypatch.setattr(supplement, "verify_supplement", lambda root: calls.append(("verify", root)))
    monkeypatch.setattr(supplement, "private_build", lambda: SimpleNamespace(
        build=lambda *args: calls.append(("build", args)),
        audit=lambda *args: calls.append(("audit", args)),
    ))
    supplement.run_build(tmp_path, ["fp16"], 1, audit=audit)
    assert calls[0] == ("verify", tmp_path)
    assert calls[1] == (("audit", (tmp_path, 2048, ["fp16"])) if audit else ("build", (tmp_path, 2048, ["fp16"], 1)))


def test_escaping_background_is_rejected(supplement: ModuleType, original: tuple, tmp_path: Path) -> None:
    root, _, manifest = original
    escaped = copy.deepcopy(manifest)
    escaped["base_examples"][0]["file"] = "../fixture.parquet"
    with pytest.raises(ValueError, match="escapes"):
        supplement.derive_dataset(tmp_path / "derived", root, escaped, CharacterTokenizer())
