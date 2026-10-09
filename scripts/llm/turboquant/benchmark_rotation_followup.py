# SPDX-License-Identifier: BSD-3-Clause
"""Gate the frozen followup's HTP probes, optional builds and crossed evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

import onnx
from benchmark_key_rotation import PACKAGE, SPLIT, compiled_signature, run
from onnx import numpy_helper
from rotation_data import SCRIPTS, digest, experiment_name, run_logged, write_json
from rotation_followup import read, source
from summarize_native_results import performance
from verify_rotated_attention import verify_graph

K_CONSTANT = re.compile(r"tq_rotation(_t)?_dense_qr_s4[2-9]_d128")


def groups(root: Path) -> list[str]:
    report = read(root / "cpu_quality.json")
    if not report["reference_pass"]:
        raise ValueError("CPU reference fidelity failed; no HTP advancement")
    expected = (
        ["A", "B"]
        + (["P"] if report["P_pass"] else [])
        + (["D"] if report["D_pass"] else [])
    )
    if report["accepted_groups"] != expected:
        raise ValueError("Accepted candidates disagree with CPU gates")
    return report["accepted_groups"]


def probes(root: Path) -> None:
    source(root)
    read(root / "rotation_validation.json")
    selected = read(root / "layer_selection.json")["layer_seeds"]
    for group in groups(root):
        layers = (
            sorted({selected.index(seed) for seed in selected}) if group == "P" else [0]
        )
        report = {}
        for layer in layers:
            run_logged(
                [
                    sys.executable,
                    str(SCRIPTS / "rotation_probe.py"),
                    "all",
                    "--root",
                    str(root),
                    "--group",
                    group,
                    "--matrix-layer",
                    str(layer),
                    "--sample-file",
                    str(root / f"samples/validation_0_layer{layer:02d}.npz"),
                ],
                root / f"reports/{group}_probe_layer{layer:02d}.log",
            )
            for name, entry in read(
                root / f"probes/{group}/layer{layer:02d}/validation_isolated.json"
            ).items():
                report[f"layer{layer:02d}_{name}"] = entry
        write_json(root / f"probes/{group}/validation_isolated.json", report)


def build(root: Path, group: str) -> None:
    source(root)
    read(root / "rotation_validation.json")
    if group not in groups(root) or group not in ("P", "D"):
        raise ValueError(
            "Only CPU-approved new P/D candidates may be built; A/B reuse validated originals"
        )
    report = read(root / f"probes/{group}/validation_isolated.json")
    if not report or not all(
        x["attention_conditioned_passed"] for x in report.values()
    ):
        raise ValueError("HTP functional gate failed")
    if (root / group).exists():
        raise FileExistsError("Refusing to overwrite a full build attempt")
    run_logged(
        [
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
            "--key-rotation-file",
            str(root / f"rotations/{group}.json"),
        ],
        root / f"reports/{group}_build.log",
    )


def normalized_name(value: str) -> str:
    return K_CONSTANT.sub(
        lambda m: f"tq_rotation{m.group(1) or ''}_dense_qr_s42_d128", value
    )


def graph_signature(graph: onnx.GraphProto) -> str:
    clone = onnx.GraphProto()
    clone.CopyFrom(graph)
    keep = [t for t in clone.initializer if not K_CONSTANT.fullmatch(t.name)]
    clone.ClearField("initializer")
    clone.initializer.extend(keep)
    for node in clone.node:
        for index, name in enumerate(node.input):
            node.input[index] = normalized_name(name)
    return hashlib.sha256(clone.SerializeToString()).hexdigest()


def audit(root: Path) -> None:
    source(root)
    capture = read(root / "capture_identity.json")
    for path, checksum in capture["source_files_sha256"].items():
        if digest(path) != checksum:
            raise ValueError("Weights, source graph or calibration changed")
    reference = {}
    context_reference = {}
    report = {
        "groups": {},
        "passed": True,
        "normalization": "Only K rotation constant name/data (seed42..49) and converter random RMSNorm display IDs; V constants, node count/order, edges, tensor dtype/shape and calibration exact",
        "scope": "ONNX + quantized DLC + final context I/O. Final internal HTP schedule/physical layout not exposed.",
        "original_source_files_unchanged": True,
    }
    for group in groups(root):
        bundle = root / group
        entries = {}
        for path in sorted(bundle.glob("*.kv_edits.json")):
            name = path.name.removesuffix(".kv_edits.json")
            result = verify_graph(bundle, name)
            model = onnx.load(bundle / f"{name}.onnx", load_external_data=False)
            compiled = compiled_signature(bundle / f"{name}.dlcinfo.txt")
            signature = (
                graph_signature(model.graph),
                normalized_name(json.dumps(compiled, sort_keys=True)),
                digest(bundle / f"{name}.encodings"),
            )
            if group == "A":
                reference[name] = signature
            elif signature != reference[name]:
                result["violations"].append(
                    "Changed non-K source/compiled topology/dtypes/layout/calibration"
                )
            constants = [
                t for t in model.graph.initializer if K_CONSTANT.fullmatch(t.name)
            ]
            result.update(
                {
                    "source_matmul_count": sum(
                        n.op_type == "MatMul" for n in model.graph.node
                    ),
                    "compiled_matmul_count": sum(
                        op["type"] == "MatMul" for op in compiled["ops"]
                    ),
                    "K_constant_count": len(constants),
                    "K_source_constant_bytes": sum(
                        numpy_helper.to_array(t).nbytes for t in constants
                    ),
                    "K_fp16_logical_constant_bytes": sum(
                        numpy_helper.to_array(t).size * 2 for t in constants
                    ),
                    "constant_note": "Logical matrix payload per graph, not final allocator/weight-sharing claim",
                    "K_constants": {
                        t.name: hashlib.sha256(
                            numpy_helper.to_array(t).T.astype("<f4").tobytes()
                        ).hexdigest()
                        for t in constants
                    },
                }
            )
            entries[name] = result
        if len(entries) != 6:
            raise ValueError("Expected six attention graphs")
        for part in range(1, 5):
            context = read(bundle / f"part{part}_of_4.json")
            layouts = [
                (
                    g["info"]["graphName"],
                    side,
                    {k: v for k, v in t["info"].items() if k != "id"},
                )
                for g in context["info"]["graphs"]
                for side in ("graphInputs", "graphOutputs")
                for t in g["info"][side]
            ]
            if group == "A":
                context_reference[part] = layouts
            elif context_reference[part] != layouts:
                raise ValueError("Final context I/O/cache ABI changed")
        sizes = {
            f"part{i}_of_4.bin": (bundle / f"part{i}_of_4.bin").stat().st_size
            for i in range(1, 5)
        }
        report["groups"][group] = {
            "graphs": entries,
            "context_binary_bytes": sizes,
            "total_context_binary_bytes": sum(sizes.values()),
        }
        report["passed"] &= all(not entry["violations"] for entry in entries.values())
    write_json(root / "reports/graph_audit.json", report)
    if not report["passed"]:
        raise ValueError("Graph audit failed; device performance not started")


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
        "source_identity_sha256": digest(root / "source_identity.json"),
        "capture_identity_sha256": digest(root / "capture_identity.json"),
        "data_manifest_sha256": digest(root / "data_manifest.json"),
        "runner_sha256": digest(root / "runner/qnn-llm-runner"),
        "cpu_quality_sha256": digest(root / "cpu_quality.json"),
        "rotation_validation_sha256": digest(root / "rotation_validation.json"),
        "graph_audit_sha256": digest(root / "reports/graph_audit.json"),
        "assets_sha256": read(root / "assets/assets.json")["sha256"],
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
            for g in groups(root)
        },
    }


def push(root: Path) -> None:
    if not read(root / "reports/graph_audit.json")["passed"]:
        raise ValueError("Audit did not pass")
    repo = SCRIPTS.parents[2]
    changed = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-m", "-o", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    write_json(
        root / "reports/implementation_identity.json",
        {
            "head": subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "branch": subprocess.run(
                ["git", "-C", str(repo), "branch", "--show-current"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "scope": "Working-tree code before final device measurement; documentation can subsequently add results",
            "sha256": {
                name: digest(repo / name) for name in changed if (repo / name).is_file()
            },
        },
    )
    write_json(root / "reports/experiment.json", identity(root))
    for group in groups(root):
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


def measure(root: Path, stage: str) -> None:
    if identity(root) != read(root / "reports/experiment.json"):
        raise ValueError("Frozen experiment identity changed")
    accepted = groups(root)
    if stage == "functional":
        for g in accepted:
            report = run(
                root,
                g,
                f"{g}_reset",
                ["--mode", "generate", "--n-gen", "8", "--sessions", "2"],
            )
            if report["sessions"][0]["generated"] != report["sessions"][1]["generated"]:
                raise ValueError("Reset consistency failed")
    elif stage == "performance":
        protocol = read(root / "protocol.json")
        orders = protocol[
            "performance_order_with_D" if "D" in accepted else "performance_order_no_D"
        ]
        for repeat, order in enumerate(orders):
            for condition in ("short", "long"):
                for g in order:
                    if g not in accepted:
                        continue
                    options = ["--mode", "generate", "--n-gen", "128"]
                    if condition == "long":
                        options += ["--tokens", "boundary_prompt_cl1024.bin"]
                    run(root, g, f"{g}_{condition}_r{repeat}", options)
    else:
        for window in range(4):
            for g in accepted:
                run(
                    root,
                    g,
                    f"{g}_test_w{window}",
                    ["--mode", "score", "--tokens", f"test_{window}.bin"],
                )


def summarize(root: Path) -> None:
    result = {
        "protocol": read(root / "protocol.json"),
        "cpu_quality": read(root / "cpu_quality.json"),
        "selection": read(root / "layer_selection.json"),
        "diagnostics": read(root / "diagnostics.json"),
        "graph_audit": read(root / "reports/graph_audit.json"),
        "groups": {},
        "limitations": "Fixed-capture Attention MSE is not autoregressive whole-model PPL. Three timing samples are descriptive. CPU is offline oracle only. Encoder and full FP32 probe failures remain separate.",
    }
    for g in groups(root):
        item = {"performance": {}}
        for condition, prompt in (("short", 35), ("long", 897)):
            runs = [
                performance(root / f"reports/{g}_{condition}_r{i}.json", prompt)
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
        reports = [read(root / f"reports/{g}_test_w{i}.json") for i in range(4)]
        n = sum(r["scored_tokens"] for r in reports)
        item["quality"] = {
            "scored_tokens": n,
            "ppl": math.exp(sum(r["nll_sum"] for r in reports) / n),
            "window_ppl": [r["ppl"] for r in reports],
        }
        result["groups"][g] = item
    write_json(root / "reports/comparison.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "probes",
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
    parser.add_argument("--group", choices=("P", "D"))
    args = parser.parse_args()
    if args.stage == "build":
        build(args.root, args.group)
    elif args.stage in ("functional", "performance", "quality"):
        measure(args.root, args.stage)
    else:
        globals()[args.stage](args.root)


if __name__ == "__main__":
    main()
