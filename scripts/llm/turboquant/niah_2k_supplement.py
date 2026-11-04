# SPDX-License-Identifier: BSD-3-Clause
"""Add an immutable CL2048 NIAH sidecar without modifying the original freeze.

Only the capacity admission list of a private ``niah_build`` module instance
changes. Its conversion, graph surgery, profile, assembly and audit functions
are reused verbatim. The original module and every original artifact remain
untouched. Failed or partial sidecars/builds are never automatically retried.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib.util
import json
import struct
from pathlib import Path
from types import ModuleType
from typing import Any

import niah_data as data

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path("/mnt/d/ai-hub-models/binaries/turboquant/niah_20261102")
SIDECAR_NAME = "niah_supplement_cl2048_20261103"
CAPACITY = 2048
ORIGINAL_CAPACITIES = [1024, 4096, 8192, 16384]
GROUPS = ("fp16", "k4_v4", "boundary_lm", "boundary_structured")
BOUNDARY_MASK = [0, 1, 2, 3, 4, 5, 6, 21, 22, 23, 24, 25, 26, 27]


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hashes(files: dict[str, str]) -> None:
    for filename, expected in files.items():
        if data.digest(Path(filename)) != expected:
            raise ValueError(f"Frozen input changed: {filename}")


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"Dataset path escapes its frozen directory: {relative}")
    return path


def check_cases(root: Path, manifest: dict) -> None:
    for case in manifest["cases"]:
        for kind in ("tokens", "prompt"):
            path = contained(root, case[f"{kind}_file"])
            if data.digest(path) != case[f"{kind}_sha256"]:
                raise ValueError(f"Frozen case changed: {case['case_id']}")
            if kind == "tokens" and path.stat().st_size != case["input_tokens"] * 4:
                raise ValueError("Token binary length differs from the manifest")
        if (
            case["max_new_tokens"] != 64
            or case["input_tokens"] + 64 > case["capacity"]
        ):
            raise ValueError("Frozen generation budget changed")


def source_snapshot(source_root: Path) -> tuple[dict, dict, dict[str, str]]:
    """Validate the original source, execution freeze and every original input."""
    protocol = read(source_root / "protocol.json")
    manifest = read(source_root / "data/data_manifest.json")
    execution = read(source_root / "execution_freeze.json")
    if (
        protocol["context_capacities"] != ORIGINAL_CAPACITIES
        or list(protocol["groups"]) != list(GROUPS)
        or protocol["mask"] != BOUNDARY_MASK
        or set(protocol["profiles"]) != set(GROUPS)
        or protocol["max_generated_tokens"] != 64
        or manifest["protocol"]["capacities"] != ORIGINAL_CAPACITIES
        or manifest["protocol"]["depths_percent"] != list(data.DEPTHS)
        or manifest["protocol"]["base_count"] != 10
        or manifest["case_counts"] != {"test": 200, "pilot": 2, "smoke": 4}
    ):
        raise ValueError("Original frozen NIAH scope differs from the approved source")
    files = dict(protocol["frozen_inputs"])
    for filename, expected in execution["files_sha256"].items():
        if filename in files and files[filename] != expected:
            raise ValueError("Conflicting original source/execution hashes")
        files[filename] = expected
    for filename, expected in manifest["tokenizer"]["files_sha256"].items():
        if filename in files and files[filename] != expected:
            raise ValueError("Conflicting original tokenizer hashes")
        files[filename] = expected
    files[str(Path(manifest["source"]["path"]).resolve(strict=True))] = manifest[
        "source"
    ]["sha256"]
    verify_hashes(files)
    check_cases(source_root / "data", manifest)
    for name in (
        "protocol.json",
        "data/data_manifest.json",
        "execution_freeze.json",
        "boundary_layer_plan.json",
    ):
        path = (source_root / name).resolve(strict=True)
        files[str(path)] = data.digest(path)
    expected_bases = {f"test_{i:02d}" for i in range(10)} | {
        "pilot_00",
        "smoke_00",
    }
    if (
        len(manifest["base_examples"]) != 12
        or {b["base_id"] for b in manifest["base_examples"]} != expected_bases
    ):
        raise ValueError("Expected exactly the original ten test, pilot and smoke bases")
    for base in manifest["base_examples"]:
        path = contained(source_root / "data", base["file"])
        if data.digest(path) != base["background_sha256"]:
            raise ValueError(f"Original background changed: {base['base_id']}")
        files[str(path)] = base["background_sha256"]
    return protocol, manifest, files


def sidecar_path(root: Path, source_root: Path) -> Path:
    source_root = source_root.resolve(strict=True)
    root = root.absolute()
    if root.name != SIDECAR_NAME or root.parent.resolve() != source_root:
        raise ValueError("CL2048 supplement must be the dedicated direct-child sidecar")
    if root.is_symlink():
        raise ValueError("Supplement root must not be a symlink")
    return source_root / SIDECAR_NAME


def derive_dataset(
    root: Path, source_root: Path, original: dict, tokenizer: Any
) -> dict:
    """Cross the already frozen bases with 2K; never reselect source documents."""
    root.mkdir(parents=True, exist_ok=False)
    for name in ("assets", "prompts", "backgrounds"):
        (root / name).mkdir()
    cases = []
    for base_metadata in original["base_examples"]:
        base = copy.deepcopy(base_metadata)
        original_background = contained(source_root / "data", base["file"])
        raw_background = original_background.read_bytes()
        if data.digest(original_background) != base["background_sha256"]:
            raise ValueError("Original background changed during supplementation")
        with (root / base["file"]).open("xb") as stream:
            stream.write(raw_background)
        base["background"] = raw_background.decode("utf-8")
        if base["split"] == "test":
            settings = [(depth, False, False) for depth in data.DEPTHS]
        elif base["split"] == "pilot":
            settings = [(50, True, False), (50, False, True)]
        elif base["split"] == "smoke":
            settings = [(50, False, False)]
        else:
            raise ValueError("Unexpected original base split")
        for depth, empty, short in settings:
            record, prompt, ids = data.make_case(
                tokenizer,
                base,
                CAPACITY,
                depth,
                no_haystack=empty,
                short_haystack=short,
            )
            raw = struct.pack(f"<{len(ids)}i", *ids)
            with (root / record["tokens_file"]).open("xb") as stream:
                stream.write(raw)
            with (root / record["prompt_file"]).open("x", encoding="utf-8") as stream:
                stream.write(prompt)
            record["tokens_sha256"] = data.digest(root / record["tokens_file"])
            cases.append(record)
    manifest = copy.deepcopy(original)
    manifest["protocol"]["capacities"] = [CAPACITY]
    manifest["protocol"]["supplement"] = (
        "Only CL2048 added from the identical preselected bases and local tokenizer; "
        "two CL2048 pilot checks and one smoke are operational gates, not test cases. "
        "CL16384 is excluded from the combined 1024/2048/4096/8192 evaluation."
    )
    manifest["cases"] = cases
    manifest["case_counts"] = {
        split: sum(c["split"] == split for c in cases)
        for split in ("test", "pilot", "smoke")
    }
    if manifest["case_counts"] != {"test": 50, "pilot": 2, "smoke": 1}:
        raise ValueError("Supplement requires exactly 50 test + 2 pilot + 1 smoke")
    data.save_new(root / "data_manifest.json", manifest)
    check_cases(root, manifest)
    return manifest


def private_build() -> ModuleType:
    """Use an isolated module namespace; original niah_build.CAPACITIES is intact."""
    spec = importlib.util.spec_from_file_location(
        "_niah_build_cl2048_private", HERE / "niah_build.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load original build implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if list(module.CAPACITIES) != ORIGINAL_CAPACITIES:
        raise ValueError("Original build admission changed")
    module.CAPACITIES = (CAPACITY,)
    return module


def prepare(root: Path, source_root: Path, authorization: str) -> None:
    if not authorization.strip():
        raise ValueError("Explicit 2K supplement / 16K exclusion authorization required")
    source_root = source_root.resolve(strict=True)
    root = sidecar_path(root, source_root)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite or retry an existing sidecar: {root}")
    original_protocol, original_data, source_files = source_snapshot(source_root)
    build = private_build()
    for group in GROUPS:
        if build.config(group).config_hash() != original_protocol["profiles"][group]["config_hash"]:
            raise ValueError(f"Profile changed after original freeze: {group}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        original_data["tokenizer"]["checkpoint_dir"],
        local_files_only=True,
        use_fast=True,
    )
    if (
        not tokenizer.is_fast
        or data.text_digest(tokenizer.chat_template)
        != original_data["tokenizer"]["chat_template_sha256"]
        or tokenizer.eos_token_id != original_data["tokenizer"]["eos_token_id"]
    ):
        raise ValueError("Frozen local tokenizer differs")
    root.mkdir(exist_ok=False)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    protocol = copy.deepcopy(original_protocol)
    protocol["created_utc"] = created
    protocol["context_capacities"] = [CAPACITY]
    protocol["maximum_capacity_authorization"] = authorization
    protocol["supplement"] = {
        "source_root": str(source_root),
        "combined_capacities": [1024, 2048, 4096, 8192],
        "excluded_capacities": [16384],
        "reason": "User added 2K and stopped 16K; original artifacts retained",
        "build_admission": "Private original niah_build module; CAPACITIES=(2048,) only",
        "workers": 1,
    }
    source_files[str(Path(__file__).resolve())] = data.digest(Path(__file__))
    protocol["frozen_inputs"] = source_files
    data.save_new(root / "protocol.json", protocol)
    with (root / "boundary_layer_plan.json").open("xb") as stream:
        stream.write((source_root / "boundary_layer_plan.json").read_bytes())
    derive_dataset(root / "data", source_root, original_data, tokenizer)
    data.save_new(
        root / "memory_estimates.json",
        {
            "source": "Unchanged niah_build memory_estimate; not total device RAM",
            "capacities": {
                str(CAPACITY): {
                    g: build.memory_estimate(CAPACITY, g != "fp16") for g in GROUPS
                }
            },
        },
    )
    data.save_new(
        root / "supplement_provenance.json",
        {
            "created_utc": created,
            "authorization": authorization,
            "source_root": str(source_root),
            "source_files_sha256": source_files,
            "sidecar_files_sha256": {
                str(root / name): data.digest(root / name)
                for name in (
                    "protocol.json",
                    "data/data_manifest.json",
                    "boundary_layer_plan.json",
                    "memory_estimates.json",
                )
            },
            "data_derivation": "Same twelve frozen bases and seeds; no base_examples() reselection",
            "combined_capacities": [1024, 2048, 4096, 8192],
            "excluded_capacity": 16384,
            "main_cases_per_profile": 200,
            "main_cases_all_profiles": 800,
            "source_mutations": [],
        },
    )
    print(f"Prepared immutable CL2048 supplement: {root}", flush=True)


def verify_supplement(root: Path) -> dict:
    provenance = read(root / "supplement_provenance.json")
    source_root = Path(provenance["source_root"])
    root = sidecar_path(root, source_root)
    verify_hashes(provenance["source_files_sha256"])
    verify_hashes(provenance["sidecar_files_sha256"])
    protocol, manifest = read(root / "protocol.json"), read(root / "data/data_manifest.json")
    original = read(source_root / "data/data_manifest.json")
    if (
        protocol["context_capacities"] != [CAPACITY]
        or protocol["mask"] != BOUNDARY_MASK
        or protocol["profiles"] != read(source_root / "protocol.json")["profiles"]
        or manifest["protocol"]["capacities"] != [CAPACITY]
        or manifest["case_counts"] != {"test": 50, "pilot": 2, "smoke": 1}
        or manifest["base_examples"] != original["base_examples"]
        or manifest["source"] != original["source"]
        or manifest["tokenizer"] != original["tokenizer"]
    ):
        raise ValueError("Supplement scope or original bases/profiles changed")
    check_cases(root / "data", manifest)
    for base in manifest["base_examples"]:
        if data.digest(contained(root / "data", base["file"])) != base["background_sha256"]:
            raise ValueError("Supplement background changed")
    return provenance


def run_build(root: Path, groups: list[str], workers: int, audit: bool = False) -> None:
    if workers != 1 or not groups or any(g not in GROUPS for g in groups):
        raise ValueError("Only serial workers=1 and the four unchanged profiles are allowed")
    verify_supplement(root)
    module = private_build()
    if audit:
        module.audit(root, CAPACITY, groups)
    else:
        module.build(root, CAPACITY, groups, workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "verify", "build", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--authorization", default="")
    parser.add_argument("--capacity", type=int, choices=(CAPACITY,), default=CAPACITY)
    parser.add_argument("--group", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--workers", type=int, choices=(1,), default=1)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.root, args.source_root, args.authorization)
    elif args.stage == "verify":
        verify_supplement(args.root)
        print(f"Verified immutable CL2048 supplement: {args.root}", flush=True)
    else:
        run_build(args.root, args.group, args.workers, audit=args.stage == "audit")


if __name__ == "__main__":
    main()
