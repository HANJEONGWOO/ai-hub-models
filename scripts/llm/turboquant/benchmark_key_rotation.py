# SPDX-License-Identifier: BSD-3-Clause
"""Frozen A/B/C rotation-only experiment, crossed-order repeated performance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import onnx
from rotation_data import SCRIPTS, digest, experiment_name, run_logged, write_json
from summarize_native_results import performance
from verify_kv_boundary import parse_dlcinfo
from verify_rotated_attention import verify_graph

SPLIT = Path("/mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_w4a16_split")
PACKAGE = Path("/mnt/d/ai-hub-models/binaries/turboquant/native_decoder_hvx_20260918")


def read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def source_signature(graph: onnx.GraphProto) -> str:
    clone = onnx.GraphProto()
    clone.CopyFrom(graph)
    keep = [t for t in clone.initializer if t.name != "tq_rotation_t_dense_qr_s42_d128"]
    clone.ClearField("initializer")
    clone.initializer.extend(keep)
    return hashlib.sha256(clone.SerializeToString()).hexdigest()


def compiled_signature(path: Path) -> dict:
    info = parse_dlcinfo(path)
    operations = []
    for op in info.ops:
        operations.append(
            {
                # Converter gives fused RMSNorm display names random numeric IDs.
                # Keep all tensor names/edges, order, dtype, shape and params exact.
                "name": re.sub(
                    r"^rms_norm_node___\d+$", "rms_norm_node___GENERATED_ID", op.name
                ),
                "type": op.op_type,
                "inputs": [vars(t) for t in op.inputs],
                "outputs": [vars(t) for t in op.outputs],
                "params": op.params,
            }
        )
    return {"ops": operations, "io": info.io_tables}


def build(root: Path, group: str) -> None:
    # CPU quality -> actual HTP probe -> full model. Fail closed on missing stages.
    read(root / "cpu_quality.json")
    probe = read(root / f"probes/{group}/validation_isolated.json")
    if not all(v["attention_conditioned_passed"] for v in probe.values()):
        raise ValueError("Required HTP functional probe did not pass")
    command = [
        sys.executable,
        str(SCRIPTS / "convert_parts.py"),
        "--split-dir",
        str(SPLIT),
        "--out",
        str(root / group),
        "--profile",
        "k4_v4_scaled",
        "--context-length",
        "1024",
        "--native-decoder-package",
        str(PACKAGE),
    ]
    if group != "A":
        command.extend(["--key-rotation-file", str(root / f"rotations/{group}.json")])
    run_logged(command, root / f"reports/{group}_build.log")


def audit(root: Path) -> None:
    report = {
        "groups": {},
        "passed": True,
        "compiled_name_normalization": "Only random numeric RMSNorm operation display IDs; all tensor names, edges, ordering, params, dtypes and shapes compared exactly.",
        "verification_scope": "ONNX and quantized DLC operations plus final HTP context I/O; internal post-finalization scheduling/layout is not exposed by context-binary-utility.",
    }
    capture = read(root / "capture_identity.json")
    for path, expected in capture["source_files_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Reference weights/calibration/source changed: {path}")
    report["capture_identity_sha256"] = digest(root / "capture_identity.json")
    report["reference_source_hashes_unchanged"] = True
    expected_sources, expected_compiled, expected_encodings, expected_io = (
        {},
        {},
        {},
        {},
    )
    for group in "ABC":
        bundle = root / group
        graphs = {}
        for path in sorted(bundle.glob("*.kv_edits.json")):
            name = path.name.removesuffix(".kv_edits.json")
            result = verify_graph(bundle, name)
            model = onnx.load(bundle / (name + ".onnx"), load_external_data=False)
            source = source_signature(model.graph)
            compiled = compiled_signature(bundle / (name + ".dlcinfo.txt"))
            enc = digest(bundle / (name + ".encodings"))
            result["source_matmul_count"] = sum(
                n.op_type == "MatMul" for n in model.graph.node
            )
            result["compiled_matmul_count"] = sum(
                o["type"] == "MatMul" for o in compiled["ops"]
            )
            if group == "A":
                (
                    expected_sources[name],
                    expected_compiled[name],
                    expected_encodings[name],
                ) = source, compiled, enc
            elif (source, compiled, enc) != (
                expected_sources[name],
                expected_compiled[name],
                expected_encodings[name],
            ):
                result["violations"].append(
                    "Source other than K constant, compiled topology/dtype/layout, or calibration changed"
                )
            graphs[name] = result
        for part in range(1, 5):
            context = read(bundle / f"part{part}_of_4.json")
            layouts = []
            for graph in context["info"]["graphs"]:
                entry = graph["info"]
                for side in ("graphInputs", "graphOutputs"):
                    for wrapped in entry[side]:
                        tensor = {k: v for k, v in wrapped["info"].items() if k != "id"}
                        layouts.append((entry["graphName"], side, tensor))
            if group == "A":
                expected_io[part] = layouts
            elif layouts != expected_io[part]:
                raise ValueError(
                    f"Final context I/O dtype/layout/cache format changed: {group} part{part}"
                )
        if len(graphs) != 6:
            raise ValueError("Expected six attention graphs")
        report["groups"][group] = graphs
        report["passed"] &= all(not g["violations"] for g in graphs.values())
    write_json(root / "reports/graph_audit.json", report)
    if not report["passed"]:
        raise ValueError("Compiled/source rotation-only audit failed")


def identity(root: Path) -> dict:
    return {
        "device": {
            name: subprocess.run(
                ["/mnt/c/adb/adb.exe", "shell", "getprop", prop],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            for name, prop in (
                ("soc_model", "ro.soc.model"),
                ("fingerprint", "ro.build.fingerprint"),
            )
        },
        "protocol_sha256": digest(root / "protocol.json"),
        "data_manifest_sha256": digest(root / "data_manifest.json"),
        "capture_identity_sha256": digest(root / "capture_identity.json"),
        "runner_sha256": digest(root / "runner/qnn-llm-runner"),
        "split_manifest_sha256": digest(SPLIT / "split_manifest.json"),
        "assets_sha256": read(root / "assets/assets.json")["sha256"],
        "graph_audit_sha256": digest(root / "reports/graph_audit.json"),
        "groups": {
            g: {
                "config_hash": read(root / g / "convert_report.json")["config_hash"],
                "conversion_sha256": digest(root / g / "convert_report.json"),
                "rotation_artifact_sha256": digest(root / f"rotations/{g}.json"),
                "probe_sha256": digest(root / f"probes/{g}/validation_isolated.json"),
                "bins_sha256": {
                    f"part{i}_of_4.bin": digest(root / g / f"part{i}_of_4.bin")
                    for i in range(1, 5)
                },
            }
            for g in "ABC"
        },
    }


def push(root: Path) -> None:
    if not read(root / "reports/graph_audit.json")["passed"]:
        raise ValueError("Graph audit required")
    write_json(root / "reports/experiment.json", identity(root))
    for group in "ABC":
        run_logged(
            [
                sys.executable,
                str(SCRIPTS / "run_device_llm.py"),
                "push",
                "--name",
                experiment_name(root) + "_" + group,
                "--bundle-dir",
                str(root / group),
                "--runner",
                str(root / "runner/qnn-llm-runner"),
            ],
            root / f"reports/{group}_push.log",
        )


def snapshot(path: Path) -> None:
    values = {"utc": datetime.now(timezone.utc).isoformat()}
    for command in ("dumpsys thermalservice", "dumpsys battery"):
        result = subprocess.run(
            ["/mnt/c/adb/adb.exe", "shell", command],
            check=False,
            capture_output=True,
            text=True,
        )
        values[command] = result.stdout
    write_json(path, values)


def run(root: Path, group: str, tag: str, options: list[str]) -> dict:
    path = root / "reports" / (tag + ".json")
    with path.with_suffix(".attempt").open("x") as stream:
        stream.write(
            "Authorized predeclared run; failures retained, no automatic repeat.\n"
        )
    snapshot(path.with_suffix(".before.json"))
    run_logged(
        [
            sys.executable,
            str(SCRIPTS / "run_device_llm.py"),
            "run",
            "--name",
            experiment_name(root) + "_" + group,
            "--assets",
            str(root / "assets"),
            "--context-buckets",
            "1024",
            "--report",
            str(path),
            "--remote-report-tag",
            experiment_name(root) + "_" + tag,
            *options,
        ],
        path.with_suffix(".stdout.log"),
    )
    snapshot(path.with_suffix(".after.json"))
    data = read(path)
    if data["device"] != read(root / "reports/experiment.json")["device"]:
        raise ValueError("Device identity changed")
    if (
        data["config_hash"] != read(root / group / "convert_report.json")["config_hash"]
        or data["kv_store_bytes"] != 30277632
        or data.get("diagnostic_only")
    ):
        raise ValueError("Unexpected configuration, storage, or instrumented run")
    if (
        data["assets"]["tokens_sha256"]
        != read(root / "assets/assets.json")["sha256"][data["assets"]["tokens_file"]]
    ):
        raise ValueError("Input changed")
    return data


def measure(root: Path, stage: str) -> None:
    if identity(root) != read(root / "reports/experiment.json"):
        raise ValueError("Frozen experiment identity changed")
    if stage == "functional":
        for group in "ABC":
            report = run(
                root,
                group,
                f"{group}_reset",
                ["--mode", "generate", "--n-gen", "8", "--sessions", "2"],
            )
            if report["sessions"][0]["generated"] != report["sessions"][1]["generated"]:
                raise ValueError("Reset isolation failed")
    elif stage == "performance":
        for repeat, order in enumerate(
            read(root / "protocol.json")["performance_order"]
        ):
            for condition in ("short", "long"):
                for group in order:
                    options = ["--mode", "generate", "--n-gen", "128"]
                    if condition == "long":
                        options += ["--tokens", "boundary_prompt_cl1024.bin"]
                    run(root, group, f"{group}_{condition}_r{repeat}", options)
    else:
        for window in range(4):
            for group in "ABC":
                run(
                    root,
                    group,
                    f"{group}_test_w{window}",
                    ["--mode", "score", "--tokens", f"test_{window}.bin"],
                )


def summarize(root: Path) -> None:
    result = {
        "protocol": read(root / "protocol.json"),
        "cpu_attention": read(root / "cpu_quality.json"),
        "selection": read(root / "selection.json"),
        "training": read(root / "training.json"),
        "rotation_validation": read(root / "rotation_validation.json"),
        "c_selected_step": read(root / "rotations/C.json")["provenance"][
            "selected_step"
        ],
        "c_matrix_equals_b": read(root / "rotations/C.json")["matrix_f32_sha256"]
        == read(root / "rotations/B.json")["matrix_f32_sha256"],
        "interpretation_limit": "Identical B/C matrices are not independent algorithms. Three runtime samples are descriptive, not evidence of statistical equivalence. Encoder and unconditioned probe failures remain separate limitations.",
        "groups": {},
    }
    for group in "ABC":
        item = {"performance": {}}
        for condition, prompt in (("short", 35), ("long", 897)):
            runs = [
                performance(root / f"reports/{group}_{condition}_r{i}.json", prompt)
                for i in range(3)
            ]
            metrics = {}
            for field in (
                "ttft_ms",
                "prefill_tok_per_s",
                "decode_tok_per_s",
                "host_kv_MiB",
                "io_buffer_MiB",
                "end_VmRSS_MiB",
                "process_VmHWM_MiB",
            ):
                values = [r[field] for r in runs]
                metrics[field] = {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "samples": values,
                }
            item["performance"][condition] = {"metrics": metrics, "runs": runs}
        quality = [read(root / f"reports/{group}_test_w{i}.json") for i in range(4)]
        item["quality"] = {
            "scored_tokens": sum(q["scored_tokens"] for q in quality),
            "ppl": math.exp(
                sum(q["nll_sum"] for q in quality)
                / sum(q["scored_tokens"] for q in quality)
            ),
            "window_ppl": [q["ppl"] for q in quality],
        }
        result["groups"][group] = item
    write_json(root / "reports/comparison.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "build",
            "audit",
            "push",
            "functional",
            "performance",
            "quality",
            "summarize",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--group", choices=list("ABC"))
    args = parser.parse_args()
    if args.stage == "build":
        if not args.group:
            parser.error("Build requires --group")
        build(args.root, args.group)
    elif args.stage in ("functional", "performance", "quality"):
        measure(args.root, args.stage)
    else:
        globals()[args.stage](args.root)


if __name__ == "__main__":
    main()
