# SPDX-License-Identifier: BSD-3-Clause
"""Single-run runtime-only HVX comparison; preserve compiled package provenance.

The context binaries and source graphs are reused byte-for-byte. Only the runtime
decoder library is overridden, after checking its unchanged QHPI registration
source and op definitions. This is not a recompilation with the new library.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import subprocess
from pathlib import Path
from typing import Any

from benchmark_asymmetric_once import (
    ARTIFACTS,
    digest,
    execute,
    read,
    save_new,
)
from summarize_native_results import performance

from qai_hub_models.models.templates.llm.turboquant.config import get_profile

GROUPS = ("k5_v3", "k6_v2")
CONDITIONS = {
    "short": (35, "prompt_ids.bin"),
    "long": (897, "boundary_prompt_cl1024.bin"),
}


def package_check(compiled: dict, runtime: dict) -> dict:
    """Conservative narrow override: only the implementation header may differ."""
    unchanged = ("package", "interface", "operations", "qairt_sdk", "hexagon_tools")
    checks = {key: compiled[key] == runtime[key] for key in unchanged}
    for name in ("decoder.cpp", "Decode4.xml"):
        checks[name] = compiled["source_files"][name] == runtime["source_files"][name]
    checks["prepare_library"] = (
        compiled["libraries"]["x86_64-linux-clang"]["sha256"]
        == runtime["libraries"]["x86_64-linux-clang"]["sha256"]
    )
    for package in (compiled, runtime):
        for value in package["libraries"].values():
            if digest(Path(value["path"])) != value["sha256"]:
                raise ValueError("Native package library hash changed")
    if not all(checks.values()):
        raise ValueError(
            f"Cannot reuse compiled contexts with changed QHPI contract: {checks}"
        )
    return checks


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": digest(path)}


def prepare(args: argparse.Namespace) -> None:
    root, previous = args.work_dir, args.previous
    if (root / "protocol.json").exists() or any(
        (root / group).exists() for group in GROUPS
    ):
        raise FileExistsError(
            "Refusing to reuse an existing experiment protocol or wrapper"
        )
    old_protocol = read(previous / "protocol.json")
    old_protocol.update(read(previous / "runner_amendment.json")["effective_runner"])
    runtime = read(args.native_package / "manifest.json")
    if digest(Path(old_protocol["runner"])) != old_protocol["runner_sha256"]:
        raise ValueError("Prior runner changed")
    frozen: dict[str, str] = {}

    def freeze(path: Path) -> None:
        frozen[str(path.resolve())] = digest(path)

    freeze(previous / "protocol.json")
    freeze(previous / "runner_amendment.json")
    freeze(previous / "comparison.json")
    freeze(args.native_package / "manifest.json")
    freeze(Path(old_protocol["runner"]))
    assets = Path(old_protocol["assets"])
    for name, expected in old_protocol["assets_manifest"]["sha256"].items():
        if digest(assets / name) != expected:
            raise ValueError(f"Legacy input changed: {name}")
        frozen[str((assets / name).resolve())] = expected
    sources = {}
    for group in GROUPS:
        source = previous / group
        metadata = read(source / "convert_report.json")
        expected_config = get_profile(group + "_scaled").config_hash()
        if (
            metadata["config_hash"] != expected_config
            or metadata["context_buckets"] != [1024]
            or not metadata["quantize_current_kv"]
            or not metadata["rotated_attention"]
            or metadata["attention_tile"] != 256
            or set(metadata["parts"]) != {f"part{i}_of_4" for i in range(1, 5)}
        ):
            raise ValueError(f"Incorrect compiled bundle: {group}")
        contract = package_check(metadata["native_decoder"], runtime)
        for kind in ("audit", "source_audit"):
            path = previous / "reports" / f"{kind}_{group}.json"
            if not read(path)["passed"]:
                raise ValueError(f"Prior audit failed: {path}")
            freeze(path)
        functions = [
            f"TurboQuantNative::tq_decode{b}" for b in (int(group[1]), int(group[4]))
        ]
        for part in range(2, 5):
            path = source / f"part{part}_of_4.bin"
            with (
                path.open("rb") as stream,
                mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data,
            ):
                if not all(data.find(name.encode()) >= 0 for name in functions):
                    raise ValueError(f"Expected serialized kernel names absent: {path}")
        for path in source.iterdir():
            if path.is_file() and (
                path.suffix in (".bin", ".onnx", ".encodings")
                or path.name.endswith(".kv_edits.json")
            ):
                freeze(path)
        freeze(source / "convert_report.json")
        sources[group] = {
            "bundle": str(source),
            "config_hash": expected_config,
            "compiled_native_decoder": metadata["native_decoder"],
            "runtime_contract_checks": contract,
            "serialized_kernel_names": functions,
            "binaries": {
                p.name: digest(p) for p in sorted(source.glob("part*_of_4.bin"))
            },
        }
    for group in ("k4_v4", *GROUPS):
        for label in ("perf_short", "perf_long", *(f"score_w{i}" for i in range(4))):
            freeze(previous / "reports" / f"{group}_{label}.json")
    for package in (runtime, *(s["compiled_native_decoder"] for s in sources.values())):
        for library in package["libraries"].values():
            freeze(Path(library["path"]))
    sdk_docs = (
        Path(runtime["qairt_sdk"]) / "docs/QAIRT-Docs/QNN/general/htp/htp_qhpi.html"
    )
    sdk_header = Path(runtime["qairt_sdk"]) / "include/QNN/HTP/core/qhpi.h"
    protocol = {
        "model": "Qwen3-1.7B W4A16",
        "context_length": 1024,
        "previous": str(previous),
        "sources": sources,
        "runtime_native_package": runtime,
        "runtime_native_package_directory": str(args.native_package),
        "runner": old_protocol["runner"],
        "runner_sha256": old_protocol["runner_sha256"],
        "assets": str(assets),
        "assets_manifest": old_protocol["assets_manifest"],
        "frozen_files": frozen,
        "performance": {
            "runs_per_condition": 1,
            "conditions": {
                key: {
                    "prompt_tokens": val[0],
                    "tokens_file": val[1],
                    "generated_tokens": 128,
                }
                for key, val in CONDITIONS.items()
            },
            "profiling": False,
        },
        "quality": {
            "windows": 4,
            "scored_tokens_per_window": 1023,
            "runs_per_window": 1,
            "aggregation": "exp(sum(nll_sum)/sum(scored_tokens))",
        },
        "group_order": list(GROUPS),
        "abi_evidence": {
            "sdk_document": file_record(sdk_docs),
            "sdk_header": file_record(sdk_header),
            "reasoning": "QHPI registers named runtime execution functions dynamically. Exact decoder.cpp and XML identity preserves operation order, names, signatures, layouts, cost and threading/resource flags; serialized context names and all source graph/context hashes are checked. Isolated HTP correctness plus full-model context loading must still pass before benchmarking.",
            "scope": "Only hvx_decode.h implementation changes; not a general ABI override facility.",
        },
        "notes": [
            "Old K4/V4 and asymmetric performance are historical controls, not contemporaneously remeasured.",
            "Same compiled encoder graphs, codebooks, weights, calibration, rotations, packing and scales.",
            "Single-run performance has no variance estimate; no statistical speedup claim.",
            "Existing encoder strict AR128 numerical validation limitation is unchanged.",
        ],
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
    }
    save_new(root / "protocol.json", protocol)
    for group, source in sources.items():
        wrapper = root / group
        wrapper.mkdir()
        original = Path(source["bundle"])
        for path in original.glob("part*_of_4.bin"):
            (wrapper / path.name).symlink_to(path.resolve())
        # Link the authentic report, rather than pretending that the new package compiled these contexts.
        (wrapper / "convert_report.json").symlink_to(
            (original / "convert_report.json").resolve()
        )
        save_new(
            wrapper / "native_runtime_override.json",
            {
                "compiled_native_decoder": source["compiled_native_decoder"],
                "runtime_native_decoder": runtime,
                "protocol": str(root / "protocol.json"),
                "context_recompiled": False,
            },
        )
    print(
        "Frozen runtime-only experiment; original contexts and compile metadata preserved."
    )


def validate(root: Path) -> dict[str, Any]:
    protocol = read(root / "protocol.json")
    for name, expected in protocol["frozen_files"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for group, source in protocol["sources"].items():
        if get_profile(group + "_scaled").config_hash() != source["config_hash"]:
            raise ValueError("Codec profile changed")
        package_check(
            source["compiled_native_decoder"], protocol["runtime_native_package"]
        )
        for name, expected in source["binaries"].items():
            if digest(root / group / name) != expected:
                raise ValueError("Wrapper context differs from frozen compiled binary")
    return protocol


def validate_probes(root: Path) -> None:
    for bits in (2, 3, 4, 5, 6):
        path = root / f"probe_b{bits}/correctness.json"
        data = read(path)
        if not {"native_t3", "native_t256"}.issubset(data) or not all(
            result["passed"] and result["bit_exact_fraction"] == 1.0
            for result in data.values()
        ):
            raise ValueError(
                f"Isolated HTP decoder correctness must pass first: {path}"
            )


def validate_report(path: Path, protocol: dict, group: str) -> dict:
    data = read(path)
    runtime = protocol["runtime_native_package"]
    if (
        data["config_hash"] != protocol["sources"][group]["config_hash"]
        or not data["quantize_current_kv"]
        or data.get("op_profiles")
        or data.get("diagnostic_only")
        or data["context_length"] != 1024
        or data["native_decoder"]["sha256"]
        != runtime["libraries"]["hexagon-v81"]["sha256"]
        or data.get("compiled_native_decoder")
        != protocol["sources"][group]["compiled_native_decoder"]
        or data.get("native_runtime_override", {}).get("manifest_sha256")
        != digest(Path(protocol["runtime_native_package_directory"]) / "manifest.json")
        or data["assets"]["tokens_sha256"]
        != protocol["assets_manifest"]["sha256"][data["assets"]["tokens_file"]]
    ):
        raise ValueError(f"Unexpected measurement configuration: {path}")
    return data


def measure(
    root: Path, group: str, label: str, options: list[str], protocol: dict
) -> dict:
    path = root / "reports" / f"{group}_{label}.json"
    if any(
        path.with_suffix(suffix).exists() for suffix in (".json", ".log", ".stdout.log")
    ):
        raise FileExistsError(f"Refusing repeated measurement: {path}")
    execute(
        "run_device_llm.py",
        [
            "run",
            "--name",
            root.name + "_" + group,
            "--assets",
            protocol["assets"],
            "--report",
            str(path),
            "--remote-report-tag",
            root.name + "_" + group + "_" + label,
            *options,
        ],
        path.with_suffix(".stdout.log"),
    )
    return validate_report(path, protocol, group)


def summarize(root: Path, protocol: dict) -> None:
    previous = Path(protocol["previous"])
    result: dict[str, Any] = {
        "protocol": file_record(root / "protocol.json"),
        "runtime_native_package": protocol["runtime_native_package"],
        "historical": read(previous / "comparison.json")["groups"],
        "optimized": {},
        "limitations": protocol["notes"],
    }
    rows = []
    for group in GROUPS:
        quality = [
            validate_report(
                root / "reports" / f"{group}_score_w{i}.json", protocol, group
            )
            for i in range(4)
        ]
        if any(
            d["scored_tokens"] != 1023
            or d["mode"] != "score"
            or d["prompt_tokens"] != 1024
            or d["assets"]["tokens_file"] != f"wikitext_w{i}.bin"
            for i, d in enumerate(quality)
        ):
            raise ValueError("Wrong quality token count")
        total_nll = sum(d["nll_sum"] for d in quality)
        total_tokens = sum(d["scored_tokens"] for d in quality)
        old_quality = [
            read(previous / "reports" / f"{group}_score_w{i}.json") for i in range(4)
        ]
        if any(
            new["device"] != old["device"]
            for new, old in zip(quality, old_quality, strict=True)
        ):
            raise ValueError("Quality device identity differs from historical control")
        old_perf = result["historical"][group]["performance"]
        value = {
            "config_hash": protocol["sources"][group]["config_hash"],
            "performance": {},
            "quality": {
                "nll_sum": total_nll,
                "scored_tokens": total_tokens,
                "mean_nll": total_nll / total_tokens,
                "ppl": math.exp(total_nll / total_tokens),
                "window_nll_deltas_vs_previous": [
                    new["nll_sum"] - old["nll_sum"]
                    for new, old in zip(quality, old_quality, strict=True)
                ],
                "note": "Equality of reported NLL is not a bit-exact-logit proof.",
            },
            "binaries_sha256": protocol["sources"][group]["binaries"],
            "context_recompiled": False,
        }
        for condition, (prompt, _) in CONDITIONS.items():
            path = root / "reports" / f"{group}_perf_{condition}.json"
            data = validate_report(path, protocol, group)
            old = read(previous / "reports" / f"{group}_perf_{condition}.json")
            if data["device"] != old["device"]:
                raise ValueError("Device identity differs from historical control")
            metrics = performance(path, prompt)
            metrics["generated_tokens_match_previous"] = (
                data["generated"] == old["generated"]
            )
            metrics["decode_speedup_vs_previous"] = (
                metrics["decode_tok_per_s"] / old_perf[condition]["decode_tok_per_s"]
            )
            metrics["ttft_ratio_vs_previous"] = (
                metrics["ttft_ms"] / old_perf[condition]["ttft_ms"]
            )
            value["performance"][condition] = metrics
            rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "ppl": value["quality"]["ppl"],
                    **{
                        name: metrics[name]
                        for name in (
                            "ttft_ms",
                            "prefill_tok_per_s",
                            "decode_tok_per_s",
                            "decode_speedup_vs_previous",
                            "host_kv_MiB",
                            "io_buffer_MiB",
                            "end_VmRSS_MiB",
                            "mean_decode_prepare_ms",
                            "mean_decode_qnn_ms",
                        )
                    },
                }
            )
        result["optimized"][group] = value
    save_new(root / "comparison.json", result)
    with (root / "comparison.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "push",
            "functional",
            "performance",
            "quality",
            "summarize",
        ),
    )
    parser.add_argument(
        "--work-dir", type=Path, default=ARTIFACTS / "asymmetric_hvx_20261010"
    )
    parser.add_argument(
        "--previous", type=Path, default=ARTIFACTS / "asymmetric_kv_20261010"
    )
    parser.add_argument("--native-package", type=Path)
    parser.add_argument("--groups", choices=GROUPS, nargs="+", default=list(GROUPS))
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    args.previous = args.previous.resolve()
    args.native_package = (args.native_package or args.work_dir / "native").resolve()
    if args.stage == "prepare":
        prepare(args)
        return
    protocol = validate(args.work_dir)
    if args.stage == "summarize":
        summarize(args.work_dir, protocol)
        return
    validate_probes(args.work_dir)
    for group in args.groups:
        root = args.work_dir
        if args.stage == "push":
            execute(
                "run_device_llm.py",
                [
                    "push",
                    "--bundle-dir",
                    str(root / group),
                    "--runner",
                    protocol["runner"],
                    "--name",
                    root.name + "_" + group,
                    "--native-runtime-package",
                    protocol["runtime_native_package_directory"],
                ],
                root / "logs" / f"push_{group}.log",
            )
        elif args.stage == "functional":
            data = measure(
                root,
                group,
                "reset_diagnostic",
                ["--mode", "generate", "--n-gen", "8", "--sessions", "2"],
                protocol,
            )
            if (
                len(data["sessions"]) != 2
                or data["sessions"][0]["generated"] != data["sessions"][1]["generated"]
            ):
                raise ValueError("Cache reset is not deterministic")
        else:
            diagnostic = root / "reports" / f"{group}_reset_diagnostic.json"
            data = validate_report(diagnostic, protocol, group)
            if (
                len(data["sessions"]) != 2
                or data["sessions"][0]["generated"] != data["sessions"][1]["generated"]
            ):
                raise ValueError(
                    "Full-model context load and reset check must pass first"
                )
            if args.stage == "performance":
                for condition, (_, tokens) in CONDITIONS.items():
                    measure(
                        root,
                        group,
                        f"perf_{condition}",
                        [
                            "--mode",
                            "generate",
                            "--n-gen",
                            "128",
                            "--sessions",
                            "1",
                            "--tokens",
                            tokens,
                        ],
                        protocol,
                    )
            elif args.stage == "quality":
                for i in range(4):
                    measure(
                        root,
                        group,
                        f"score_w{i}",
                        ["--mode", "score", "--tokens", f"wikitext_w{i}.bin"],
                        protocol,
                    )


if __name__ == "__main__":
    main()
