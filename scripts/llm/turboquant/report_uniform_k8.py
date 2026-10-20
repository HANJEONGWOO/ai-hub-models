# SPDX-License-Identifier: BSD-3-Clause
"""Graph audit and measured Uniform/Structured comparison, no device execution."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
from benchmark_asymmetric_once import digest, read, save_new
from benchmark_uniform_k8 import GROUPS, validate
from summarize_lm8_convergence import probes
from summarize_native_results import performance
from verify_kv_boundary import parse_dlcinfo

from qai_hub_models.models.templates.llm.turboquant import structured_k8, uniform_k8


def audit(root: Path) -> None:
    rows = {}
    for path in sorted((root / "uniform").glob("*.kv_edits.json")):
        name = path.name.removesuffix(".kv_edits.json")
        graphs = [
            onnx.load(root / g / (name + ".onnx"), load_external_data=False).graph
            for g in GROUPS
        ]

        def inside_k8_encoder(name: str) -> bool:
            m = re.match(r"tq_key_(\d+)_enc_(scalar_|scale_)", name)
            return bool(m and int(m[1]) % 2)

        normalized = []
        for graph in graphs:
            nodes = {}
            for node in graph.node:
                if inside_k8_encoder(node.name):
                    continue
                n = copy.deepcopy(node)
                if n.domain == "turboquant" and n.op_type in (
                    "Decode8Uniform",
                    "Decode8Structured",
                ):
                    n.op_type = "K8Arithmetic"
                    n.input[2] = "K8_COEFFICIENTS"
                nodes[tuple(n.output)] = n.SerializeToString()
            normalized.append(nodes)
        constants = [{t.name: t for t in graph.initializer} for graph in graphs]

        def variable(name: str) -> bool:
            return name.startswith(
                (
                    "tq_uniform_k8_",
                    "tq_structured_k8_",
                    "tq_native_uniform_k8_",
                    "tq_native_structured_k8_",
                )
            )

        diffs = [
            k
            for k in constants[0].keys() | constants[1].keys()
            if not variable(k)
            and (
                k not in constants[0]
                or k not in constants[1]
                or constants[0][k].SerializeToString()
                != constants[1][k].SerializeToString()
            )
        ]
        uparams = onnx.numpy_helper.to_array(
            constants[0]["tq_native_uniform_k8_coefficients_fp16"]
        ).ravel()
        sparams = onnx.numpy_helper.to_array(
            constants[1]["tq_native_structured_k8_coefficients_fp16"]
        ).ravel()
        dlc = [parse_dlcinfo(root / g / (name + ".dlcinfo.txt")) for g in GROUPS]
        matmul = []
        for info in dlc:
            matmul.append(
                {
                    op.name: {k: v for k, v in asdict(op).items() if k != "ident"}
                    for op in info.ops
                    if op.op_type in ("MatMul", "FullyConnected")
                }
            )
        checks = {
            "unchanged_non_k8_encoder_and_attention": normalized[0] == normalized[1],
            "unchanged_other_initializers": not diffs,
            "io_identical": all(
                [v.SerializeToString() for v in getattr(graphs[0], f)]
                == [v.SerializeToString() for v in getattr(graphs[1], f)]
                for f in ("input", "output")
            ),
            "uniform_coefficients_exact": np.array_equal(
                uparams, uniform_k8.coefficients().astype(np.float16)
            ),
            "structured_coefficients_exact": np.array_equal(
                sparams, structured_k8.coefficients().astype(np.float16)
            ),
            "no_k8_decoder_lut": not any(
                "k8_centroids" in k or "b8_centroids" in k
                for graph in constants
                for k in graph
            ),
            "same_activation_parameter_encodings": digest(
                root / "uniform" / (name + ".encodings")
            )
            == digest(root / "structured" / (name + ".encodings")),
            "compiled_io_identical": dlc[0].io_tables == dlc[1].io_tables,
            "compiled_matmul_contracts_identical": matmul[0] == matmul[1],
        }
        rows[name] = {
            "checks": checks,
            "unexpected_constant_changes": diffs,
            "compiled_op_counts": {
                g: len(i.ops) for g, i in zip(GROUPS, dlc, strict=True)
            },
            "unexpected_node_differences": [
                list(k)
                for k in normalized[0].keys() | normalized[1].keys()
                if normalized[0].get(k) != normalized[1].get(k)
            ],
        }
    passed = len(rows) == 6 and all(
        all(row["checks"].values()) for row in rows.values()
    )
    save_new(
        root / "reports/graph_audit.json",
        {
            "passed": passed,
            "graphs": rows,
            "limits": "ONNX and quantized DLC contracts, not a claim of identical HTP scheduling. K8 encoder and arithmetic decoder changes are intentional.",
        },
    )
    assert passed, "Graph audit failed; inspect saved report before evaluating"


def summary(root: Path) -> None:
    protocol = read(root / "protocol.json")
    rows, documents, groups = [], [], {}
    for group in GROUPS:
        perf = [
            performance(root / "reports" / f"{group}_perf_r{i}.json", 897)
            for i in range(3)
        ]
        stats = {
            k: {
                "mean": float(np.mean([p[k] for p in perf])),
                "std_ddof1": float(np.std([p[k] for p in perf], ddof=1)),
                "min": min(p[k] for p in perf),
                "max": max(p[k] for p in perf),
                "samples": [p[k] for p in perf],
            }
            for k in (
                "ttft_ms",
                "prefill_tok_per_s",
                "decode_tok_per_s",
                "host_kv_MiB",
                "io_buffer_MiB",
                "end_VmRSS_MiB",
                "process_VmHWM_MiB",
            )
        }
        quality = {}
        for split in ("legacy", "heldout"):
            docs = []
            for i in range(4):
                path = root / "reports" / f"{group}_score_{split}_{i}.json"
                report = read(path)
                assert report["scored_tokens"] == 1023
                assert (
                    report["config_hash"] == protocol["profiles"][group]["config_hash"]
                )
                nll = report["nll_sum"] / report["scored_tokens"]
                entry = {
                    "group": group,
                    "split": split,
                    "window": i,
                    "nll_sum": report["nll_sum"],
                    "mean_nll": nll,
                    "ppl": math.exp(nll),
                    "scored_tokens": 1023,
                    "sha256": digest(path),
                }
                docs.append(entry)
                documents.append(entry)
            nll = sum(d["nll_sum"] for d in docs) / 4092
            quality[split] = {
                "mean_nll": nll,
                "ppl": math.exp(nll),
                "scored_tokens": 4092,
                "documents": docs,
            }
        row = {
            "group": group,
            **{f"{s}_{m}": quality[s][m] for s in quality for m in ("mean_nll", "ppl")},
            **{
                f"{k}_{stat}": v[stat]
                for k, v in stats.items()
                for stat in ("mean", "std_ddof1", "min", "max")
            },
        }
        rows.append(row)
        groups[group] = {
            "quality": quality,
            "performance": stats,
            "raw_performance": perf,
            "config_hash": protocol["profiles"][group]["config_hash"],
            "binaries": {
                p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                for p in sorted((root / group).glob("part*_of_4.bin"))
            },
        }
    contrasts = {}
    for split in ("legacy", "heldout"):
        delta = [
            u["mean_nll"] - s["mean_nll"]
            for u, s in zip(
                groups["uniform"]["quality"][split]["documents"],
                groups["structured"]["quality"][split]["documents"],
                strict=True,
            )
        ]
        contrasts[split] = {
            "uniform_minus_structured_mean_nll": float(np.mean(delta)),
            "document_deltas": delta,
            "uniform_better_documents": sum(x < 0 for x in delta),
            "uniform_worse_documents": sum(x > 0 for x in delta),
            "uniform_ppl_relative_change": groups["uniform"]["quality"][split]["ppl"]
            / groups["structured"]["quality"][split]["ppl"]
            - 1,
        }
    save_new(
        root / "comparison.json",
        {
            "rows": rows,
            "groups": groups,
            "contrasts": contrasts,
            "protocol": protocol,
            "offline_quality": read(root / "offline_quality.json"),
            "codec_costs": read(root / "codec_costs.json"),
            "graph_audit": read(root / "reports/graph_audit.json"),
            "limits": "Three crossed performance sessions, sample std; small n does not establish statistical superiority. PPL one pass per frozen window; busy cycles not TTFT fractions.",
        },
    )
    for name, data in (("comparison.csv", rows), ("document_nll.csv", documents)):
        with (root / name).open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("audit", "probes", "summary"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    if a.stage == "probes":
        probes(root, tuple(GROUPS), structured_lut_control=False)
    elif a.stage == "audit":
        validate(root)
        audit(root)
    else:
        summary(root)
