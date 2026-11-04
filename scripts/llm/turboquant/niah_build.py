# SPDX-License-Identifier: BSD-3-Clause
"""Immutable four-profile NIAH preparation/build; no device runs or fallback.

PYTHONPATH=src venv/bin/python scripts/llm/turboquant/niah_build.py prepare --root ROOT
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/niah_build.py build --root ROOT --capacity 4096 --group fp16
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from benchmark_asymmetric_once import ARTIFACTS, digest, execute, read, save_new
from benchmark_k8_repeat import MASKS, NATIVE, RUNNER, SDK, SPLIT
from benchmark_k8_repeat import config as repeat_config
from niah_audit import (
    audit_source_graph,
    compiled_io_contract,
    frozen_codec_constants,
    memory_estimate,
)

from qai_hub_models.models.templates.llm.turboquant.config import (
    TurboQuantConfig,
    get_profile,
    structured_allocation_plan,
)

GROUPS = ("fp16", "k4_v4", "boundary_lm", "boundary_structured")
CAPACITIES = (1024, 4096, 8192, 16384)
DEFAULT_ROOT = ARTIFACTS / "niah_20261102"
OLD_NATIVE = ARTIFACTS / "native_decoder_hvx_20260918"
REPEATED = ARTIFACTS / "iceic2027_k8_repeat_20261027"
SOURCES = {
    "fp16": ARTIFACTS / "qwen3_1_7b_fp16_attention_20260929/bundle",
    "k4_v4": ARTIFACTS / "asymmetric_kv_20261010/k4_v4",
    "boundary_lm": REPEATED / "boundary_lm",
    "boundary_structured": REPEATED / "boundary_structured",
}


def config(group: str) -> TurboQuantConfig:
    if group in ("fp16", "k4_v4"):
        return get_profile(
            {"fp16": "baseline_fp16_kv_fp16_attn", "k4_v4": "k4_v4_scaled"}[group]
        )
    if group not in GROUPS:
        raise ValueError(group)
    return repeat_config(group)


def native(group: str) -> Path | None:
    return None if group == "fp16" else OLD_NATIVE if group == "k4_v4" else NATIVE


def link(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source:
            raise ValueError(f"Conflicting immutable link: {target}")
    else:
        target.symlink_to(source)


def stage(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and (
            path.name.startswith(("part", "prompt_", "token_"))
            or path.suffix == ".data"
        ):
            link(path, target / path.name)
    # Deployment rewrites metadata. Keep wrapper-local copies, never symlinks.
    save_new(target / "convert_report.json", read(source / "convert_report.json"))


def prepare(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    frozen: dict[str, str] = {}

    def freeze(path: Path) -> None:
        key = str(path.resolve(strict=True))
        if key not in frozen:
            frozen[key] = digest(path)

    split = read(SPLIT / "split_manifest.json")
    freeze(SPLIT / "split_manifest.json")
    freeze(RUNNER)
    plan = structured_allocation_plan(MASKS["boundary"])
    save_new(root / "boundary_layer_plan.json", plan)
    sources_audit = {}
    for group in GROUPS:
        c = config(group)
        source = SOURCES[group]
        meta = read(source / "convert_report.json")
        if meta["config_hash"] != c.config_hash() or meta["context_buckets"] != [1024]:
            raise ValueError(f"Historical profile/context mismatch: {group}")
        package = native(group)
        if meta.get("native_decoder") != (
            read(package / "manifest.json") if package else None
        ):
            raise ValueError(f"Historical Native package mismatch: {group}")
        if (
            group == "boundary_structured"
            and meta.get("structured_k8_decoder") != "arithmetic"
        ):
            raise ValueError("Structured source must use the arithmetic decoder")
        freeze(source / "convert_report.json")
        if package:
            freeze(package / "manifest.json")
            for lib in read(package / "manifest.json")["libraries"].values():
                p = Path(lib["path"])
                freeze(p)
                if digest(p) != lib["sha256"]:
                    raise ValueError(f"Native library hash mismatch: {p}")
        checks = {}
        for part_name, entry in meta["parts"].items():
            freeze(source / f"{part_name}.bin")
            original = split["parts"][part_name]
            original_path = Path(original["bundle_dir"]) / (original["class"] + ".onnx")
            freeze(original_path)
            freeze(original_path.with_suffix(".encodings"))
            for data_file in original_path.parent.glob("*.data"):
                freeze(data_file)
            if (
                digest(original_path.with_suffix(".encodings"))
                != original["encodings_sha256"]
            ):
                raise ValueError(f"Original calibration changed: {part_name}")
            for graph, info in entry["graphs"].items():
                if not info.get("surgery"):
                    continue
                check = audit_source_graph(original_path, source / f"{graph}.onnx")
                checks[graph] = check
                if not check["passed"]:
                    raise ValueError(
                        f"Source preservation failed: {group}/{graph}: {check['violations']}"
                    )
                for file in check["external_weight_files"].values():
                    freeze(Path(file))
        sources_audit[group] = checks
        stage(source, root / "bundles/cl1024" / group)
    save_new(root / "source_audit.json", sources_audit)
    save_new(
        root / "memory_estimates.json",
        {
            "source": "Historical compiled CL1024 metadata plus current LlmSession allocation rules; not total device RAM.",
            "capacities": {
                str(c): {g: memory_estimate(c, g != "fp16") for g in GROUPS}
                for c in (*CAPACITIES, 32768)
            },
        },
    )
    save_new(
        root / "protocol.json",
        {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": "Qwen3-1.7B W4A16",
            "groups": GROUPS,
            "context_capacities": CAPACITIES,
            "depths_percent": [0, 25, 50, 75, 100],
            "cases_per_cell": 10,
            "max_generated_tokens": 64,
            "runs_per_case": 1,
            "batch_size": 1,
            "decoding": "greedy",
            "enable_thinking": False,
            "mask": MASKS["boundary"],
            "sdk": str(SDK),
            "runner": str(RUNNER),
            "split": str(SPLIT),
            "source_bundles": {g: str(p) for g, p in SOURCES.items()},
            "profiles": {
                g: {
                    "config": config(g).to_dict(),
                    "config_hash": config(g).config_hash(),
                    "native": str(native(g)) if native(g) else None,
                }
                for g in GROUPS
            },
            "maximum_capacity_authorization": "User approved common maximum 16384 after disclosure that FP16 CL32768 explicit KV+AR1/AR128 I/O alone is about 10.55 GiB; 32768 deferred, not executed.",
            "failure_policy": "No automatic context reduction, CPU fallback, discarded failures or retries. Stop and report capacity/device failures before changing conditions.",
            "fixed": "Existing codebooks, masks, weights, activation calibration, Dense QR, norm correction, FP16 scale, current-KV policy, Tile256, and historical per-profile Native packages.",
            "timing_scope": "NIAH quality evaluation only; no speed superiority claims from these runs.",
            "frozen_inputs": frozen,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )
    print(f"Prepared immutable NIAH protocol: {root}", flush=True)


def check(root: Path, capacity: int) -> dict:
    p = read(root / "protocol.json")
    if capacity not in p["context_capacities"] or capacity not in CAPACITIES:
        raise ValueError("Capacity not authorized by the frozen NIAH protocol")
    for g in GROUPS:
        if config(g).config_hash() != p["profiles"][g]["config_hash"]:
            raise ValueError(f"Profile changed after freeze: {g}")
    for file, expected in p["frozen_inputs"].items():
        if digest(Path(file)) != expected:
            raise ValueError(f"Frozen source changed: {file}")
    return p


def convert_options(
    root: Path, capacity: int, group: str, part: int, output: Path
) -> list[str]:
    c = config(group)
    args = [
        "--split-dir",
        str(SPLIT),
        "--out",
        str(output),
        "--profile",
        c.profile,
        "--context-length",
        str(capacity),
        "--sequence-lengths",
        "1",
        "128",
        "--parts",
        str(part),
    ]
    if group == "boundary_structured":
        args += [
            "--layer-plan",
            str(root / "boundary_layer_plan.json"),
            "--structured-k8-decoder",
            "arithmetic",
        ]
    if package := native(group):
        args += ["--native-decoder-package", str(package)]
    return args


def completed_part(path: Path, number: int, capacity: int, group: str) -> bool:
    report = path / "convert_report.json"
    if not report.exists():
        return False
    d = read(report)
    part = f"part{number}_of_4"
    if d["config_hash"] != config(group).config_hash() or d["context_buckets"] != [
        capacity
    ]:
        raise ValueError(f"Incompatible completed part: {path}")
    return (
        part in d["parts"]
        and "context_s" in d["parts"][part]
        and (path / f"{part}.bin").is_file()
    )


def assemble(root: Path, capacity: int, group: str) -> Path:
    base = root / "builds" / f"cl{capacity}"
    paths = {
        1: base / "shared/part1",
        **{i: base / group / f"part{i}" for i in (2, 3, 4)},
    }
    target = root / "bundles" / f"cl{capacity}" / group
    target.mkdir(parents=True, exist_ok=False)
    result = copy.deepcopy(read(paths[2] / "convert_report.json"))
    result["parts"] = {}
    for number, source in paths.items():
        if not completed_part(
            source, number, capacity, "fp16" if number == 1 else group
        ):
            raise ValueError(f"Incomplete part: {source}")
        name = f"part{number}_of_4"
        entry = read(source / "convert_report.json")["parts"][name]
        if number == 1 and any(g.get("surgery") for g in entry["graphs"].values()):
            raise ValueError("Shared part1 must be the unchanged embedding-only graph")
        result["parts"][name] = entry
        for path in source.iterdir():
            if path.is_file() and (
                path.name.startswith(name)
                or any(path.name.startswith(g + ".") for g in entry["graphs"])
                or path.suffix == ".data"
            ):
                link(path, target / path.name)
    result["part_sources"] = {f"part{i}_of_4": str(p) for i, p in paths.items()}
    save_new(target / "convert_report.json", result)
    return target


def build(root: Path, capacity: int, groups: list[str], workers: int) -> None:
    check(root, capacity)
    if not 1 <= workers <= 2:
        raise ValueError("At most two part converters are permitted")
    if capacity == 1024:
        print("CL1024 source bundles already staged; no rebuild", flush=True)
        return
    base = root / "builds" / f"cl{capacity}"

    def one(group: str, number: int) -> None:
        output = base / ("shared" if number == 1 else group) / f"part{number}"
        effective_group = "fp16" if number == 1 else group
        if completed_part(output, number, capacity, effective_group):
            return
        if output.exists():
            raise FileExistsError(
                f"Partial build retained; no automatic retry: {output}"
            )
        execute(
            "convert_parts.py",
            convert_options(root, capacity, effective_group, number, output),
            root
            / "logs"
            / f"build_cl{capacity}_{effective_group if number != 1 else 'shared'}_part{number}.log",
        )

    for group in groups:
        target = root / "bundles" / f"cl{capacity}" / group
        if target.exists():
            report = read(target / "convert_report.json")
            if report["config_hash"] != config(group).config_hash() or report[
                "context_buckets"
            ] != [capacity]:
                raise ValueError(f"Incompatible existing bundle: {target}")
            print(f"Preserved completed bundle: {target}", flush=True)
            continue
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # The first capacity build overlaps tiny shared part1 with part2;
            # later profiles reuse the completed embedding-only binary.
            list(pool.map(partial(one, group), (1, 2, 3, 4)))
        print(f"Assembled {assemble(root, capacity, group)}", flush=True)


def audit(root: Path, capacity: int, groups: list[str]) -> None:
    """Fail-closed source and compiled audit; never execute a device graph."""
    import re

    from verify_fp16_attention import verify_graph as verify_fp16
    from verify_rotated_attention import verify_graph as verify_rotated

    check(root, capacity)
    split = read(SPLIT / "split_manifest.json")
    for group in groups:
        bundle = root / "bundles" / f"cl{capacity}" / group
        meta = read(bundle / "convert_report.json")
        c = config(group)
        checks = {
            "config": meta["config_hash"] == c.config_hash()
            and meta["config"] == c.to_dict(),
            "context": meta["context_buckets"] == [capacity]
            and meta["context_length"] == capacity,
            "native": meta.get("native_decoder")
            == (read(native(group) / "manifest.json") if native(group) else None),
            "decoder": meta.get("structured_k8_decoder", "lut")
            == ("arithmetic" if group == "boundary_structured" else "lut"),
            "current_policy": meta["quantize_current_kv"] == (group != "fp16"),
            "attention_tile": meta["attention_tile"] == (0 if group == "fp16" else 256),
            "parts": set(meta["parts"]) == {f"part{i}_of_4" for i in range(1, 5)},
        }
        graphs = {}
        compiled_contexts = []
        observed_layers = {1: [], 128: []}
        for part, entry in meta["parts"].items():
            # Historical assembled bundles may omit the context-info symlink.
            info = bundle / f"{part}.json"
            if not info.exists():
                info = (bundle / f"{part}.bin").resolve().with_suffix(".json")
            compiled_contexts.append(read(info))
            original = split["parts"][part]
            original_path = Path(original["bundle_dir"]) / (original["class"] + ".onnx")
            for name, value in entry["graphs"].items():
                if not value.get("surgery"):
                    continue
                ar = int(re.search(r"_ar(\d+)_", name)[1])
                result = (verify_fp16 if group == "fp16" else verify_rotated)(
                    bundle, name
                )
                observed_layers[ar].extend(result["layers"])
                result["source_preservation"] = audit_source_graph(
                    original_path, bundle / f"{name}.onnx"
                )
                result["violations"].extend(result["source_preservation"]["violations"])
                if group != "fp16":
                    refname = name.replace(f"_cl{capacity}_", "_cl1024_")
                    result["fixed_constants"] = frozen_codec_constants(
                        SOURCES[group] / f"{refname}.onnx", bundle / f"{name}.onnx"
                    )
                    if not result["fixed_constants"]["passed"]:
                        result["violations"].append(
                            "Codebook/index/norm/rotation constants changed"
                        )
                    edits = read(bundle / f"{name}.kv_edits.json")
                    for item in edits["codec_io"]:
                        if (
                            item["past_tokens"] != capacity - ar
                            or item["new_tokens"] != ar
                        ):
                            result["violations"].append(
                                "Wrong static source codec shape"
                            )
                graphs[name] = result
                print(
                    f"Audit {group}/{name}: {'PASS' if not result['violations'] else 'FAIL'}",
                    flush=True,
                )
        checks["layer_coverage"] = all(
            sorted(layers) == list(range(28)) for layers in observed_layers.values()
        )
        io = compiled_io_contract(compiled_contexts, capacity, c)
        passed = (
            all(checks.values())
            and io["passed"]
            and bool(graphs)
            and all(not g["violations"] for g in graphs.values())
        )
        result = {
            "bundle": str(bundle),
            "checks": checks,
            "compiled_io": io,
            "graphs": graphs,
            "passed": passed,
        }
        save_new(root / "audits" / f"cl{capacity}" / f"{group}.json", result)
        if not passed:
            raise ValueError(f"NIAH bundle contract failed: {group}/CL{capacity}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "build", "audit"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--capacity", type=int, choices=CAPACITIES)
    parser.add_argument("--group", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.root)
    else:
        if args.capacity is None:
            parser.error("build requires --capacity")
        if args.stage == "build":
            build(args.root, args.capacity, args.group, args.workers)
        else:
            audit(args.root, args.capacity, args.group)


if __name__ == "__main__":
    main()
