# SPDX-License-Identifier: BSD-3-Clause
"""Frozen decoder-only experiment: same package, same cache, one long run/group."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
from benchmark_asymmetric_once import ARTIFACTS, digest, execute, read, save_new
from benchmark_mixed_once import measure
from summarize_native_results import performance
from verify_kv_boundary import DlcInfo, parse_dlcinfo

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.native_decoder import (
    STRUCTURED_COEFFICIENTS,
    STRUCTURED_OP,
)
from qai_hub_models.models.templates.llm.turboquant.structured_k8 import (
    coefficients,
    metadata,
)

GROUPS = ("structured_lut", "structured_arithmetic")
PROFILE = "mixed_k6_v2_uniform_structured_k8"
PREVIOUS = ARTIFACTS / "structured_k8_20261013" / PROFILE


def link(source: Path, target: Path) -> None:
    if target.exists():
        if target.resolve() != source.resolve():
            raise ValueError(f"Conflicting file {target}")
    else:
        target.symlink_to(source.resolve(strict=True))


def prepare(root: Path) -> None:
    config = get_profile(PROFILE)
    previous = read(PREVIOUS / "convert_report.json")
    native = read(root / "native/manifest.json")
    if (
        previous["config_hash"] != config.config_hash()
        or "Decode8Structured" not in native["operations"]
    ):
        raise ValueError("Incompatible previous model or Native package")
    assets = ARTIFACTS / "device_assets_cl1024"
    manifest = read(assets / "assets.json")
    runner = ARTIFACTS / "asymmetric_kv_20261010/runner/qnn-llm-runner"
    frozen = [PREVIOUS / "convert_report.json", root / "native/manifest.json", runner]
    frozen += list(PREVIOUS.glob("*.dlc")) + list(PREVIOUS.glob("*.onnx"))
    frozen += list(PREVIOUS.glob("*.encodings")) + list(
        PREVIOUS.glob("*.kv_edits.json")
    )
    frozen += [Path(v["path"]) for v in native["libraries"].values()]
    frozen += [assets / name for name in manifest["sha256"]]
    save_new(
        root / "protocol.json",
        {
            "branch": "exp/structured-k8-decoder",
            "base_commit": "246a53417be071ecfc97b7a8b4f23e4c3013ce28",
            "model": "Qwen3-1.7B W4A16",
            "context_length": 1024,
            "groups": GROUPS,
            "profiles": {
                g: {"config": config.to_dict(), "config_hash": config.config_hash()}
                for g in GROUPS
            },
            "previous_bundle": str(PREVIOUS),
            "runner": str(runner),
            "split": str(ARTIFACTS / "qwen3_1_7b_w4a16_split"),
            "assets": str(assets),
            "assets_manifest": manifest,
            "native": str(root / "native"),
            "frozen_files": {str(p): digest(p) for p in frozen},
            "quantizer": metadata(),
            "performance": {
                "sessions_per_group": 1,
                "prompt": 897,
                "generated": 128,
                "profiled": False,
                "order": GROUPS,
            },
            "quality": {
                "windows": 4,
                "scored_tokens_per_window": 1023,
                "aggregation": "exp(total_nll/4092)",
            },
            "diagnostics": "one physical decoder probe per 1/3/256/1023 token shape; same input hashes; two-session 8-token reset check is not a performance sample",
            "build": "Recompile both LUT and arithmetic contexts with the same Native package; reuse unchanged LUT DLCs and the shared non-KV part1. Arithmetic parts 2/3/4 build independently, max three converters.",
            "fixed": "Encoder, codebook, K8 odd14/K4 even14, V2, Dense K42/V542, norm correction, FP16 scales, current/past policy, cache hash, weights/calibration and attention",
            "limits": "Single run; no variance/significance or busy-cycle-to-latency-share inference; no default promotion",
        },
    )
    # Only link immutable graph inputs. Context outputs/logs/configs must be new
    # files, never symlinks which could overwrite the old experiment.
    target = root / GROUPS[0]
    target.mkdir(exist_ok=False)
    for path in PREVIOUS.iterdir():
        if path.is_file() and (
            path.name.startswith(("token_", "prompt_")) or path.suffix == ".data"
        ):
            link(path, target / path.name)
    report = copy.deepcopy(previous)
    report["native_decoder"] = native
    report["decoder_experiment_source"] = str(PREVIOUS)
    save_new(target / "convert_report.json", report)


def validate(root: Path) -> dict:
    protocol = read(root / "protocol.json")
    for name, expected in protocol["frozen_files"].items():
        if digest(Path(name)) != expected:
            raise ValueError("Frozen file changed: " + name)
    if (
        get_profile(PROFILE).config_hash()
        != protocol["profiles"][GROUPS[0]]["config_hash"]
    ):
        raise ValueError("Cache/quantizer identity changed")
    return protocol


def build_lut(root: Path, protocol: dict) -> None:
    execute(
        "convert_parts.py",
        [
            "--split-dir",
            protocol["split"],
            "--out",
            str(root / GROUPS[0]),
            "--profile",
            PROFILE,
            "--context-length",
            "1024",
            "--sequence-lengths",
            "1",
            "128",
            "--native-decoder-package",
            protocol["native"],
            "--context-only",
        ],
        root / "logs/build_lut.log",
    )


def build_arithmetic(root: Path, protocol: dict) -> None:
    numeric = read(root / "probe_arithmetic/correctness.json")
    if len(numeric) != 4 or not all(
        x["passed"] and x["bit_exact_fraction"] == 1 for x in numeric.values()
    ):
        raise ValueError("Arithmetic HTP bit-exact gate is not complete")

    def part(number: int) -> None:
        execute(
            "convert_parts.py",
            [
                "--split-dir",
                protocol["split"],
                "--out",
                str(root / "arithmetic_parts" / f"part{number}"),
                "--profile",
                PROFILE,
                "--context-length",
                "1024",
                "--sequence-lengths",
                "1",
                "128",
                "--parts",
                str(number),
                "--structured-k8-decoder",
                "arithmetic",
                "--native-decoder-package",
                protocol["native"],
            ],
            root / "logs" / f"build_arithmetic_part{number}.log",
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(part, [2, 3, 4]))


def assemble(root: Path, protocol: dict) -> None:
    target = root / GROUPS[1]
    target.mkdir(exist_ok=False)
    reports = {
        i: read(root / "arithmetic_parts" / f"part{i}" / "convert_report.json")
        for i in (2, 3, 4)
    }
    combined = copy.deepcopy(reports[2])
    combined["parts"] = {}
    sources = {
        1: root / GROUPS[0],
        **{i: root / "arithmetic_parts" / f"part{i}" for i in (2, 3, 4)},
    }
    provenance = {}
    for i, source in sources.items():
        report = read(source / "convert_report.json")
        assert report["config_hash"] == protocol["profiles"][GROUPS[0]]["config_hash"]
        assert report["native_decoder"] == read(root / "native/manifest.json")
        name = f"part{i}_of_4"
        entry = report["parts"][name]
        assert "context_s" in entry
        if i == 1:
            assert all(
                not g.get("surgery", {}).get("codec_io")
                for g in entry["graphs"].values()
            )
        else:
            assert report["structured_k8_decoder"] == "arithmetic"
        combined["parts"][name] = entry
        link(source / f"{name}.bin", target / f"{name}.bin")
        for graph in entry["graphs"]:
            for file in source.glob(graph + ".*"):
                if file.is_file():
                    link(file, target / file.name)
        for file in source.glob("*.data"):
            link(file, target / file.name)
        provenance[name] = {
            "source": str(source),
            "report_sha256": digest(source / "convert_report.json"),
            "binary_sha256": digest(source / f"{name}.bin"),
        }
    combined["decoder_experiment_parts"] = provenance
    save_new(target / "convert_report.json", combined)


def audit(root: Path) -> None:
    def compiled_non_decoder(info: DlcInfo) -> dict:
        signatures = {}
        for op in info.ops:
            if op.op_type in ("Decode8", STRUCTURED_OP):
                continue
            value = asdict(op)
            value.pop("ident")
            # QAIRT gives fused RMS nodes random numeric names between runs.
            # Match their exact inputs/outputs/parameters, ignoring only this
            # non-semantic node label and the table row ID.
            if re.fullmatch(r"rms_norm_node___\d+", value["name"]):
                value["name"] = "SDK_RMS_AUTONAME"
            outputs = tuple(t.name for t in op.outputs)
            if outputs in signatures:
                raise ValueError("Ambiguous compiled outputs")
            signatures[outputs] = value
        return signatures

    checks = {}
    for path in (root / GROUPS[1]).glob("*.kv_edits.json"):
        name = path.name.removesuffix(".kv_edits.json")
        graphs = [
            onnx.load(root / group / (name + ".onnx"), load_external_data=False).graph
            for group in GROUPS
        ]
        left, right = graphs
        nodes = [list(g.node) for g in graphs]
        identical = len(nodes[0]) == len(nodes[1])
        replaced = 0
        for a, b in zip(*nodes, strict=True):
            if b.op_type == STRUCTURED_OP:
                normalized = copy.deepcopy(b)
                normalized.op_type = "Decode8"
                normalized.input[2] = "tq_native_structured_k8_centroids_fp16"
                identical &= a.SerializeToString() == normalized.SerializeToString()
                replaced += 1
            else:
                identical &= a.SerializeToString() == b.SerializeToString()
        constants = [
            {t.name: t.SerializeToString() for t in g.initializer} for g in graphs
        ]
        common = constants[0].keys() & constants[1].keys()
        values = next(t for t in right.initializer if t.name == STRUCTURED_COEFFICIENTS)
        checks[name] = {
            "only_decoder_nodes_changed": bool(identical and replaced),
            "arithmetic_ops": replaced,
            "shared_constants_identical": all(
                constants[0][k] == constants[1][k] for k in common
            ),
            "constant_changes_only_lut_to_coefficients": constants[0].keys()
            - constants[1].keys()
            == {"tq_native_structured_k8_centroids_fp16"}
            and constants[1].keys() - constants[0].keys() == {STRUCTURED_COEFFICIENTS},
            "coefficients_exact": bool(
                np.array_equal(
                    onnx.numpy_helper.to_array(values).ravel(),
                    coefficients().astype(np.float16),
                )
            ),
            "io_identical": all(
                [v.SerializeToString() for v in getattr(left, k)]
                == [v.SerializeToString() for v in getattr(right, k)]
                for k in ["input", "output", "value_info"]
            ),
            "all_encodings_identical": read(root / GROUPS[0] / (name + ".encodings"))
            == read(root / GROUPS[1] / (name + ".encodings")),
        }
        a, b = [
            parse_dlcinfo(root / group / (name + ".dlcinfo.txt")) for group in GROUPS
        ]
        checks[name].update(
            {
                "compiled_op_count_identical": len(a.ops) == len(b.ops),
                "compiled_io_identical": a.io_tables == b.io_tables,
                "compiled_non_decoder_signatures_identical": compiled_non_decoder(a)
                == compiled_non_decoder(b),
                "compiled_arithmetic_count_exact": sum(
                    op.op_type == STRUCTURED_OP for op in b.ops
                )
                == replaced,
            }
        )
    result = {
        "graphs": checks,
        "compiled_normalization": "Ignore numeric row IDs and SDK-generated RMS node labels only; tensor names, dtypes, dimensions and parameters must match",
        "passed": len(checks) == 6
        and all(all(bool(v) for v in row.values()) for row in checks.values()),
    }
    save_new(root / "reports/decoder_only_audit.json", result)
    if not result["passed"]:
        raise ValueError("Decoder-only graph audit failed")


def evaluate(root: Path, group: str, protocol: dict) -> None:
    if group == GROUPS[1]:
        audit(root)
    execute(
        "verify_rotated_attention.py",
        [
            "--bundle",
            str(root / group),
            "--report",
            str(root / "reports" / f"audit_{group}.json"),
        ],
        root / "logs" / f"audit_{group}.log",
    )
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
        ],
        root / "logs" / f"push_{group}.log",
    )
    reset = measure(
        root,
        group,
        "reset",
        ["--mode", "generate", "--n-gen", "8", "--sessions", "2"],
        protocol,
    )
    assert reset["sessions"][0]["generated"] == reset["sessions"][1]["generated"]
    measure(
        root,
        group,
        "perf_long",
        [
            "--mode",
            "generate",
            "--n-gen",
            "128",
            "--sessions",
            "1",
            "--tokens",
            "boundary_prompt_cl1024.bin",
        ],
        protocol,
    )
    for i in range(4):
        measure(
            root,
            group,
            f"score_w{i}",
            ["--mode", "score", "--tokens", f"wikitext_w{i}.bin"],
            protocol,
        )


def summarize(root: Path, protocol: dict) -> None:
    results = {}
    rows = []
    for group in GROUPS:
        quality = [
            read(root / "reports" / f"{group}_score_w{i}.json") for i in range(4)
        ]
        assert all(q["scored_tokens"] == 1023 for q in quality)
        nll = sum(q["nll_sum"] for q in quality)
        perf = performance(root / "reports" / f"{group}_perf_long.json", 897)
        assert perf["host_kv_MiB"] == 28.875
        assert (
            perf["native_decoder"]["sha256"]
            == read(root / "native/manifest.json")["libraries"]["hexagon-v81"]["sha256"]
        )
        results[group] = {
            "performance": perf,
            "ppl": math.exp(nll / 4092),
            "mean_nll": nll / 4092,
            "scored_tokens": 4092,
            "window_nll_sums": [q["nll_sum"] for q in quality],
            "window_ppl": [q["ppl"] for q in quality],
            "binaries": {
                p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                for p in (root / group).glob("part*_of_4.bin")
            },
        }
        rows.append(
            {
                "group": group,
                "ppl": results[group]["ppl"],
                **{
                    k: perf[k]
                    for k in [
                        "ttft_ms",
                        "prefill_tok_per_s",
                        "decode_tok_per_s",
                        "host_kv_MiB",
                        "io_buffer_MiB",
                        "end_VmRSS_MiB",
                        "mean_decode_prepare_ms",
                        "mean_decode_qnn_ms",
                    ]
                },
            }
        )
    save_new(
        root / "comparison.json",
        {
            "protocol_sha256": digest(root / "protocol.json"),
            "sessions_per_group": 1,
            "dispersion": None,
            "groups": results,
            "ppl_nll_sums_identical": results[GROUPS[0]]["window_nll_sums"]
            == results[GROUPS[1]]["window_nll_sums"],
        },
    )
    with (root / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "build-lut",
            "build-arithmetic",
            "assemble",
            "evaluate",
            "summarize",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        prepare(root)
        return
    protocol = validate(root)
    if args.stage == "evaluate":
        for group in args.groups:
            evaluate(root, group, protocol)
    else:
        {
            "build-lut": build_lut,
            "build-arithmetic": build_arithmetic,
            "assemble": assemble,
            "summarize": summarize,
        }[args.stage](root, protocol)


if __name__ == "__main__":
    main()
