# SPDX-License-Identifier: BSD-3-Clause
"""Single-run Qwen3-1.7B K4/V4, K5/V3, K6/V2 comparison on legacy WikiText windows.

Explicit stages, immutable inputs, exclusive attempt logs. No default promotion,
training, candidate selection, repeated performance runs or CPU inference fallback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import onnx
from summarize_native_results import performance

from qai_hub_models.models.templates.llm.turboquant.config import get_profile

SCRIPTS = Path(__file__).resolve().parent
ARTIFACTS = Path("/mnt/d/ai-hub-models/binaries/turboquant")
PROFILES = {"k4_v4": "k4_v4_scaled", "k5_v3": "k5_v3_scaled", "k6_v2": "k6_v2_scaled"}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def save_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def execute(script: str, options: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(SCRIPTS / script), *options]
    print("START", log.stem, flush=True)
    with log.open("x") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print("DONE", log.stem, flush=True)


def prepare(args: argparse.Namespace) -> None:
    root = args.work_dir
    if b"--op-package" not in args.runner.read_bytes():
        raise ValueError(
            "Runner predates Native support; build the current runner first"
        )
    baseline = read(args.baseline / "convert_report.json")
    config = get_profile("k4_v4_scaled")
    if (
        baseline["config_hash"] != config.config_hash()
        or baseline["context_buckets"] != [1024]
        or not baseline["quantize_current_kv"]
        or not baseline["native_decoder"]
        or baseline["attention_tile"] != 256
    ):
        raise ValueError(
            "Baseline must be unchanged Dense seed42, LM-tree Native, current KV, fixed CL1024"
        )
    assets = read(args.assets / "assets.json")
    if assets["wikitext_windows"] != [f"wikitext_w{i}.bin" for i in range(4)]:
        raise ValueError(
            "Use the legacy four WikiText windows, not rotation-selection documents"
        )
    for name, expected in assets["sha256"].items():
        if digest(args.assets / name) != expected:
            raise ValueError(f"Asset changed: {name}")
    baseline_bins = {p.name: digest(p) for p in args.baseline.glob("part*_of_4.bin")}
    if len(baseline_bins) != 4:
        raise ValueError("Incomplete baseline")
    protocol = {
        "model": "Qwen3-1.7B W4A16",
        "context_length": 1024,
        "profiles": {
            g: {
                "config": get_profile(p).to_dict(),
                "config_hash": get_profile(p).config_hash(),
            }
            for g, p in PROFILES.items()
        },
        "group_order": list(PROFILES),
        "baseline_source": str(args.baseline),
        "baseline_bins_sha256": baseline_bins,
        "assets": str(args.assets),
        "assets_manifest": assets,
        "runner": str(args.runner),
        "runner_sha256": digest(args.runner),
        "split": str(args.split),
        "split_manifest_sha256": digest(args.split / "split_manifest.json"),
        "native_package": read(root / "native/manifest.json"),
        "performance": {
            "sessions_per_group_condition": 1,
            "profiling": False,
            "conditions": {
                "short": {"prompt_tokens": 35, "generated_tokens": 128},
                "long": {"prompt_tokens": 897, "generated_tokens": 128},
            },
        },
        "quality": {
            "windows": 4,
            "tokens_per_window": 1024,
            "scored_per_window": 1023,
            "aggregation": "exp(sum(nll_sum)/sum(scored_tokens))",
            "runs_per_window": 1,
        },
        "fixed": [
            "K seed42/V seed542 Dense QR",
            "exact LM tree",
            "frozen LM codebooks from pinned reference",
            "QJL off",
            "quantize current and past KV",
            "FP16 effective scales with norm correction",
            "Native HVX LUT",
            "tile256",
            "weights/calibration",
        ],
        "storage": {
            "bits_K_plus_V": 8,
            "scales_per_KV_vector_pair": 2,
            "expected_host_KV_MiB": 28.875,
        },
        "notes": [
            "Historical PPL approximately 25.66 identifies the dataset, not a promised fresh result.",
            "Native4 keeps its legacy kernel/package; asymmetric codecs use the new tight-bit HVX kernel.",
            "Single runs have no variance estimate; diagnostic timings are excluded.",
            "Existing encoder FP16 numerical validation limitations are tracked separately.",
        ],
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
    }
    if protocol["git_branch"] != "main":
        raise ValueError("This experiment was requested on main, without commits")
    save_new(root / "protocol.json", protocol)
    # Wrapper links preserve the old bundle: push writes only wrapper-local metadata.
    target = root / "k4_v4"
    target.mkdir()
    for path in args.baseline.iterdir():
        if path.is_file() and (
            path.name.startswith(("prompt_", "token_", "part"))
            or path.suffix == ".data"
        ):
            (target / path.name).symlink_to(path.resolve())
    save_new(target / "convert_report.json", baseline)
    print(
        "Protocol frozen; baseline binaries and all input hashes retained.", flush=True
    )


def validate(root: Path) -> dict[str, Any]:
    protocol = read(root / "protocol.json")
    amendment = root / "runner_amendment.json"
    if amendment.exists():
        protocol.update(read(amendment)["effective_runner"])
    for name, expected in protocol["assets_manifest"]["sha256"].items():
        if digest(Path(protocol["assets"]) / name) != expected:
            raise ValueError(f"Asset changed after protocol freeze: {name}")
    if digest(Path(protocol["runner"])) != protocol["runner_sha256"]:
        raise ValueError("Runner changed")
    if (
        digest(Path(protocol["split"]) / "split_manifest.json")
        != protocol["split_manifest_sha256"]
    ):
        raise ValueError("Split checkpoint manifest changed")
    if read(root / "native/manifest.json") != protocol["native_package"]:
        raise ValueError("Native package changed after protocol freeze")
    for group, profile in PROFILES.items():
        if (
            get_profile(profile).config_hash()
            != protocol["profiles"][group]["config_hash"]
        ):
            raise ValueError("Profile changed after protocol freeze")
    return protocol


def validate_bundle(root: Path, group: str, protocol: dict[str, Any]) -> dict[str, Any]:
    bundle = root / group
    metadata = read(bundle / "convert_report.json")
    if (
        metadata["config_hash"] != protocol["profiles"][group]["config_hash"]
        or metadata["context_buckets"] != [1024]
        or not metadata["quantize_current_kv"]
        or not metadata["rotated_attention"]
        or metadata["attention_tile"] != 256
        or set(metadata["parts"]) != {f"part{i}_of_4" for i in range(1, 5)}
    ):
        raise ValueError(f"Wrong/incomplete bundle: {group}")
    for name, info in metadata["parts"].items():
        if "context_s" not in info or not (bundle / (name + ".bin")).is_file():
            raise ValueError(f"Unfinalized {group}/{name}")
    native = metadata["native_decoder"]
    if group != "k4_v4" and native != protocol["native_package"]:
        raise ValueError(
            "Bundle uses a different Native package than the frozen protocol"
        )
    for library in native["libraries"].values():
        if digest(Path(library["path"])) != library["sha256"]:
            raise ValueError("Native library differs from the compiled dependency")
    return metadata


def measure(
    root: Path, group: str, label: str, options: list[str], protocol: dict[str, Any]
) -> None:
    metadata = validate_bundle(root, group, protocol)
    if not read(root / "reports" / f"audit_{group}.json")["passed"]:
        raise ValueError("Compiled HTP graph audit must pass first")
    path = root / "reports" / f"{group}_{label}.json"
    if path.exists() or path.with_suffix(".log").exists():
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
    data = read(path)
    if (
        data["config_hash"] != metadata["config_hash"]
        or not data["quantize_current_kv"]
        or data.get("op_profiles")
    ):
        raise ValueError("Wrong measurement configuration")
    if (
        data["assets"]["tokens_sha256"]
        != protocol["assets_manifest"]["sha256"][data["assets"]["tokens_file"]]
    ):
        raise ValueError("Device input hash mismatch")


def audit_source(root: Path, group: str) -> None:
    """Only codec internals, their constants and packed widths may change."""
    base, candidate = root / "k4_v4", root / group
    graphs = {}
    external_files: dict[str, str] = {}
    for path in sorted(candidate.glob("*.kv_edits.json")):
        name = path.name.removesuffix(".kv_edits.json")
        left = onnx.load(base / f"{name}.onnx", load_external_data=False).graph
        right = onnx.load(candidate / f"{name}.onnx", load_external_data=False).graph

        def retained_nodes(g: onnx.GraphProto) -> list[bytes]:
            return [
                n.SerializeToString() for n in g.node if not n.name.startswith("tq_")
            ]

        def retained_constants(g: onnx.GraphProto) -> dict[str, bytes]:
            return {
                t.name: t.SerializeToString()
                for t in g.initializer
                if not t.name.startswith("tq_")
            }

        enc_left, enc_right = (read(p / f"{name}.encodings") for p in (base, candidate))
        checks = {
            "non_codec_nodes_unchanged": retained_nodes(left) == retained_nodes(right),
            "non_codec_constants_unchanged": retained_constants(left)
            == retained_constants(right),
            "weight_encodings_unchanged": enc_left["param_encodings"]
            == enc_right["param_encodings"],
            "activation_encodings_unchanged": enc_left["activation_encodings"]
            == enc_right["activation_encodings"],
        }
        for tensor in right.initializer:
            entries = {e.key: e.value for e in tensor.external_data}
            if "location" not in entries:
                continue
            lpath, rpath = (
                (p / entries["location"]).resolve(strict=True)
                for p in (base, candidate)
            )
            if lpath != rpath:
                raise ValueError(
                    f"External weights do not reference the same frozen source: {name}/{tensor.name}"
                )
            if str(rpath) not in external_files:
                external_files[str(rpath)] = digest(rpath)
        graphs[name] = checks
    report = {
        "group": group,
        "graphs": graphs,
        "external_weights_sha256": external_files,
        "passed": bool(graphs) and all(all(v.values()) for v in graphs.values()),
    }
    save_new(root / "reports" / f"source_audit_{group}.json", report)
    if not report["passed"]:
        raise ValueError(f"Changes beyond codec internals: {report}")


def existing_decoder_diagnostics(root: Path, package: dict[str, Any]) -> dict[str, Any]:
    """Read already-recorded correctness probe profiles; never execute the device."""
    sdk = Path(package["qairt_sdk"])
    reports = {}
    for path in sorted(
        root.glob("probe_b*/device/out_native_t256/qnn-profiling-data_0.log")
    ):
        text = subprocess.check_output(
            [
                str(sdk / "bin/x86_64-linux-clang/qnn-profile-viewer"),
                "--reader",
                str(sdk / "lib/x86_64-linux-clang/libQnnHtpProfilingReader.so"),
                "--input_log",
                str(path),
            ],
            text=True,
        )
        reports[path.parents[2].name] = {
            "source": str(path),
            "sha256": digest(path),
            "decoder_op_cycles": [
                int(v)
                for v in re.findall(r"decode\d+:OpId_\d+ \(cycles\) : (\d+)", text)
            ],
            "hvx_threads": [
                int(v) for v in re.findall(r"Number of HVX threads used : (\d+)", text)
            ],
        }
    binary = Path(package["libraries"]["hexagon-v81"]["path"])
    objdump = Path(package["hexagon_tools"]) / "bin/hexagon-llvm-objdump"
    assembly = subprocess.check_output([str(objdump), "-d", str(binary)], text=True)
    functions = list(re.finditer(r"(?m)^[0-9a-f]+ <([^>]+)>:\n", assembly))
    instructions = {}
    for i, function in enumerate(functions):
        name = function.group(1)
        if "tq_decode_bits_hvx" not in name and "tq_decode_hvx" not in name:
            continue
        stop = functions[i + 1].start() if i + 1 < len(functions) else len(assembly)
        body = assembly[function.end() : stop]
        instructions[name] = {
            instruction: body.count(instruction)
            for instruction in ("memub(", "vinsert(", "vmemu(", "vlut16(", "vlut32(")
        }
    return {
        "scope": "Existing isolated HTP correctness probes, 8 heads x 256 tokens x 128 coordinates; new package also contains unchanged Decode4 kernel. Not a full-model stage profile or repeated throughput benchmark; cycles must not be converted to end-to-end speedup.",
        "profiles": reports,
        "disassembly": {
            "binary_sha256": digest(binary),
            "command": [str(objdump), "-d", str(binary)],
            "static_instruction_counts": instructions,
            "interpretation": "Non-nibble partial-row memcpy is lowered to scalar DSP byte loads and vector insertions. LUT and bit extraction still use HVX; this is not an ARM/CPU fallback. Static counts do not quantify full-model latency attribution.",
        },
    }


def summarize(root: Path, protocol: dict[str, Any]) -> None:
    result: dict[str, Any] = {
        "protocol": str(root / "protocol.json"),
        "runner_sha256": protocol["runner_sha256"],
        "performance_runs_per_group_condition": 1,
        "quality_runs_per_window": 1,
        "native_correctness": {
            p.parent.name: read(p)
            for p in sorted(root.glob("probe_b*/correctness.json"))
        },
        "encoder_correctness": {},
        "existing_decoder_diagnostics": existing_decoder_diagnostics(
            root, protocol["native_package"]
        ),
        "groups": {},
        "limitations": [
            "Single performance samples: no variance estimate or statistical speedup claim.",
            "Standalone encoder strict FP16 scale tolerance remains failed for some AR128 cases; see numeric reports.",
            "K4 uses the legacy two-row nibble HVX kernel, while asymmetric profiles use the new tight-bit HVX kernel; runtime differences include implementation costs, not just bit allocation.",
        ],
    }
    for path in sorted(root.glob("encoder_*/p2_report.json")):
        report = read(path)
        cases = [
            c for g in report["graphs"].values() for c in g.get("cases", {}).values()
        ]
        result["encoder_correctness"][path.parent.name] = {
            "report": str(path),
            "passed": report["passed"],
            "max_scale_relative_error": max(c["norm_max_rel_error"] for c in cases),
            "unexplained_index_mismatches": sum(
                c["unexplained_index_mismatches"] for c in cases
            ),
            "all_ops_profiled_on_htp": all(
                not g["dlc_ops_missing_from_accelerator_profile"]
                for g in report["graphs"].values()
            ),
        }
    rows = []
    device = None
    for group in PROFILES:
        metadata = validate_bundle(root, group, protocol)
        quality = [
            read(root / "reports" / f"{group}_score_w{i}.json") for i in range(4)
        ]
        for data in quality:
            if (
                data["scored_tokens"] != 1023
                or data["config_hash"] != metadata["config_hash"]
            ):
                raise ValueError("Quality report mismatch")
            if device is not None and data["device"] != device:
                raise ValueError("Device mismatch")
            device = data["device"]
        total_nll = sum(d["nll_sum"] for d in quality)
        total_tokens = sum(d["scored_tokens"] for d in quality)
        perf = {
            condition: performance(
                root / "reports" / f"{group}_perf_{condition}.json", prompt
            )
            for condition, prompt in (("short", 35), ("long", 897))
        }
        value = {
            "config_hash": metadata["config_hash"],
            "performance": perf,
            "quality": {
                "nll_sum": total_nll,
                "scored_tokens": total_tokens,
                "mean_nll": total_nll / total_tokens,
                "ppl": math.exp(total_nll / total_tokens),
                "windows": [
                    {k: d[k] for k in ("nll_sum", "scored_tokens", "ppl", "assets")}
                    for d in quality
                ],
            },
            "binary_bytes": sum(
                p.stat().st_size for p in (root / group).glob("part*_of_4.bin")
            ),
            "bins_sha256": {
                p.name: digest(p) for p in (root / group).glob("part*_of_4.bin")
            },
            "compiled_graph_audit": read(root / "reports" / f"audit_{group}.json"),
            "source_audit": read(root / "reports" / f"source_audit_{group}.json"),
            "native_library_bytes": Path(
                metadata["native_decoder"]["libraries"]["hexagon-v81"]["path"]
            )
            .stat()
            .st_size,
        }
        result["groups"][group] = value
        for condition, metrics in perf.items():
            rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "ppl": value["quality"]["ppl"],
                    "mean_nll": value["quality"]["mean_nll"],
                    **{
                        k: metrics[k]
                        for k in (
                            "ttft_ms",
                            "prefill_tok_per_s",
                            "decode_tok_per_s",
                            "host_kv_MiB",
                            "io_buffer_MiB",
                            "end_VmRSS_MiB",
                            "mean_decode_prepare_ms",
                            "mean_decode_qnn_ms",
                        )
                    },
                }
            )
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
        choices=[
            "prepare",
            "amend-runner",
            "build",
            "audit",
            "push",
            "functional",
            "performance",
            "quality",
            "summarize",
        ],
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-tag",
        default="reset_diagnostic",
        help="Fresh label for a failed functional check retry; never affects performance names",
    )
    parser.add_argument(
        "--groups", nargs="+", choices=list(PROFILES), default=list(PROFILES)
    )
    parser.add_argument(
        "--baseline", type=Path, default=ARTIFACTS / "k_rotation_quality_20261006/A"
    )
    parser.add_argument(
        "--assets", type=Path, default=ARTIFACTS / "device_assets_cl1024"
    )
    parser.add_argument(
        "--split", type=Path, default=ARTIFACTS / "qwen3_1_7b_w4a16_split"
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=ARTIFACTS / "qnn_runner/android-arm64/qnn-llm-runner",
    )
    args = parser.parse_args()
    root = args.work_dir.resolve()
    args.work_dir = root
    if args.stage == "prepare":
        prepare(args)
        return
    if args.stage == "amend-runner":
        if list((root / "reports").glob("*_perf_*.json")) or list(
            (root / "reports").glob("*_score_w*.json")
        ):
            raise ValueError("Cannot change the runner after measurements start")
        if b"--op-package" not in args.runner.read_bytes():
            raise ValueError("Runner lacks Native package support")
        save_new(
            root / "runner_amendment.json",
            {
                "reason": "Initial legacy runner ignored --op-package and failed context loading; only failed diagnostics exist, no benchmark or quality runs.",
                "effective_runner": {
                    "runner": str(args.runner.resolve()),
                    "runner_sha256": digest(args.runner),
                },
                "launcher_fix": "ADSP_LIBRARY_PATH uses FastRPC semicolon-separated directories",
                "unchanged": "Profiles, binaries, data, performance/quality protocol and group order",
            },
        )
        return
    protocol = validate(root)
    if args.stage == "summarize":
        summarize(root, protocol)
        return
    for group in args.groups:
        bundle = root / group
        if args.stage == "build":
            if group == "k4_v4":
                continue
            execute(
                "convert_parts.py",
                [
                    "--split-dir",
                    protocol["split"],
                    "--out",
                    str(bundle),
                    "--profile",
                    PROFILES[group],
                    "--context-length",
                    "1024",
                    "--context-buckets",
                    "1024",
                    "--native-decoder-package",
                    str(root / "native"),
                ],
                root / "logs" / f"build_{group}.log",
            )
        elif args.stage == "audit":
            validate_bundle(root, group, protocol)
            execute(
                "verify_rotated_attention.py",
                [
                    "--bundle",
                    str(bundle),
                    "--report",
                    str(root / "reports" / f"audit_{group}.json"),
                ],
                root / "logs" / f"audit_{group}.log",
            )
            audit_source(root, group)
        elif args.stage == "push":
            validate_bundle(root, group, protocol)
            if not read(root / "reports" / f"audit_{group}.json")["passed"]:
                raise ValueError("Audit failed")
            execute(
                "run_device_llm.py",
                [
                    "push",
                    "--bundle-dir",
                    str(bundle),
                    "--runner",
                    protocol["runner"],
                    "--name",
                    root.name + "_" + group,
                ],
                root / "logs" / f"push_{group}.log",
            )
        elif args.stage == "functional":
            measure(
                root,
                group,
                args.diagnostic_tag,
                ["--mode", "generate", "--n-gen", "8", "--sessions", "2"],
                protocol,
            )
            data = read(root / "reports" / f"{group}_{args.diagnostic_tag}.json")
            if (
                len(data["sessions"]) != 2
                or data["sessions"][0]["generated"] != data["sessions"][1]["generated"]
            ):
                raise ValueError("Cache reset is not deterministic")
        elif args.stage == "performance":
            for condition, tokens in (
                ("short", "prompt_ids.bin"),
                ("long", "boundary_prompt_cl1024.bin"),
            ):
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
