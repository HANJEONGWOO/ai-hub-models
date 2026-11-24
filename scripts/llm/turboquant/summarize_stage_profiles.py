#!/usr/bin/env python3
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Attribute measured QNN leaf-node cycles to semantic stages of frozen graphs.

Cycles are never converted into invented per-stage wall times. Percentages use
the sum of leaf NODE cycle events, not the inclusive graph counter or wall time.
Zero-valued events are retained: they may denote fusion, elimination or missing
instrumentation. Source ONNX and compiled DLC names provide the attribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx
from verify_kv_boundary import parse_dlcinfo

LAYOUT = {
    "Convert",
    "Cast",
    "Reshape",
    "Transpose",
    "StridedSlice",
    "Slice",
    "Concat",
    "Squeeze",
    "ExpandDims",
}


def semantic_stage(name: str) -> tuple[str, int | None, str]:
    """Classify explicit exported names without treating all tq_ ops as codecs."""
    match = re.match(r"tq_(key|value)_(\d+)_(.*)", name)
    if match:
        kind, layer, tail = match.group(1), int(match.group(2)), match.group(3)
        if tail.startswith("enc_"):
            sub = tail[4:]
            if sub.startswith("rotated"):
                stage = "encode.rotate"
            # Retain old event names for historical reports, not an export path.
            elif sub in {"above", "above_i32", "index"} or sub.startswith("scalar_"):
                stage = "encode.scalar_index"
            elif sub.startswith(("index_hi", "index_lo", "byte_", "pack_")):
                stage = "encode.pack"
            elif sub.startswith("scale_") or sub == "effective_scale":
                stage = "encode.scale_correction"
            else:
                stage = "encode.normalize"
            return stage, layer, kind
        if tail == "packed_out":
            return "encode.pack", layer, kind
        if tail == "scale_out":
            return "encode.scale_correction", layer, kind
        if re.fullmatch(r"(current|tile\d+)_native_fp16", tail):
            which = "current" if tail.startswith("current") else "past"
            return "decode.native_" + which, layer, kind
        return "layout_precision", layer, kind
    match = re.match(r"tq_attn_(\d+)_(.*)", name)
    if match:
        layer, tail = int(match.group(1)), match.group(2)
        if re.fullmatch(r"head\d+_q\d+_(rotated|scaled)", tail):
            return "attention.query_rotate", layer, "query"
        if re.fullmatch(r"tile\d+_head\d+_q\d+_score", tail):
            return "attention.qk", layer, "key"
        if re.fullmatch(r"tile\d+_head\d+_q\d+_(partial|sum)", tail):
            return "attention.av", layer, "value"
        return "layout_precision", layer, ""
    match = re.match(r"fp16_attn_(\d+)_head\d+_q\d+_(qk|av)_half$", name)
    if match:
        return (
            "attention." + match.group(2),
            int(match.group(1)),
            "key" if match.group(2) == "qk" else "value",
        )
    return "", None, ""


def op_name(identifier: str) -> str:
    return re.sub(r" \((cycles|us)\)$", "", identifier.split(":OpId_", 1)[0])


def leaf_nodes(events: list[dict]) -> list[dict]:
    """Only actual NODE counters; exclude inclusive parents and other units."""
    return [
        e for e in events if e["type"] == 404 and e["unit"] == 3 and e["children"] == 0
    ]


class GraphStages:
    def __init__(self, bundle: Path, ar: int, part: int) -> None:
        stem = f"{'prompt' if ar == 128 else 'token'}_ar{ar}_cl1024_{part}_of_4"
        info = parse_dlcinfo(bundle / f"{stem}.dlcinfo.txt")
        self.compiled = {op.name: op for op in info.ops}
        self.source: dict[str, tuple[str, int | None, str]] = {}
        path = bundle / f"{stem}.onnx"
        if not path.exists():  # embedding-only part has no surgery ONNX
            return
        graph = onnx.load(path, load_external_data=False).graph
        producers = {out: node for node in graph.node for out in node.output}
        score_nodes: set[str] = set()
        allowed = {
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Where",
            "Cast",
            "Concat",
            "Slice",
            "Reshape",
            "Expand",
            "Transpose",
            "Clip",
        }
        for node in graph.node:
            if node.op_type == "Softmax":
                score_nodes.add(node.name)
                pending = list(node.input)
                seen = set()
                while pending:
                    tensor = pending.pop()
                    if tensor in seen:
                        continue
                    seen.add(tensor)
                    parent = producers.get(tensor)
                    if parent is not None and parent.op_type in allowed:
                        score_nodes.add(parent.name)
                        pending.extend(parent.input)
        for node in graph.node:
            tag = ("", None, "")
            for name in [node.name, *node.output]:
                candidate = semantic_stage(name)
                if candidate[0]:
                    tag = candidate
                    break
            if node.op_type == "MatMul" and any(
                inp.startswith("tq_rotation_") and not inp.startswith("tq_rotation_t_")
                for inp in node.input
            ):
                layer_match = re.search(r"tq_attn_(\d+)_", node.input[0])
                tag = (
                    "attention.output_rotate",
                    int(layer_match.group(1)) if layer_match else None,
                    "value",
                )
            elif node.name in score_nodes and (
                not tag[0] or tag[0] == "layout_precision"
            ):
                tag = ("attention.score_softmax", tag[1], tag[2])
            if tag[0]:
                for name in [node.name, *node.output]:
                    self.source[name] = tag

    def classify(self, identifier: str) -> dict:
        name = op_name(identifier)
        op = self.compiled.get(name)
        optype = op.op_type if op else "unknown"
        tag = self.source.get(name, semantic_stage(name))
        if op and not tag[0]:
            for tensor in op.outputs:
                if tensor.name in self.source:
                    tag = self.source[tensor.name]
                    break
        # Converter-inserted representation changes stay separate from the
        # arithmetic operation whose output name happens to prefix them.
        if "_converted_" in name or "_pre_reshape" in name or "_post_reshape" in name:
            tag = ("layout_precision", tag[1], tag[2])
        elif tag[0] == "encode.rotate" and optype in LAYOUT:
            # rotated_ht1d is a materialized transpose, not the Dense GEMM.
            tag = ("layout_precision", tag[1], tag[2])
        elif not tag[0]:
            if optype == "Softmax":
                tag = ("attention.score_softmax", None, "")
            elif optype in LAYOUT:
                tag = ("layout_precision", None, "")
            elif optype in {
                "FullyConnected",
                "Conv2d",
                "Conv1d",
                "Convolution",
            } or name.startswith(("node_Conv_", "node_linear")):
                tag = ("model.linear", None, "")
            elif name.startswith(("Input ", "Output ")):
                tag = ("backend.io", None, "")
            elif name.startswith("OpId_"):
                tag = ("unattributed.compiler", None, "")
            elif op:
                tag = ("model.other", None, "")
            else:
                tag = ("unattributed.named", None, "")
        return {
            "name": name,
            "op_type": optype,
            "stage": tag[0],
            "layer": tag[1],
            "kv": tag[2],
            "compiled_name_match": op is not None,
            "output_shapes": ";".join(t.dims for t in op.outputs) if op else "",
        }


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    total = sum(r["cycles"] for r in rows)
    for row in rows:
        stage = row["stage"]
        entry = groups.setdefault(
            stage,
            {
                "stage": stage,
                "cycles": 0,
                "events": 0,
                "zero_events": 0,
                "by_kv_cycles": Counter(),
            },
        )
        entry["cycles"] += row["cycles"]
        entry["events"] += 1
        entry["zero_events"] += row["cycles"] == 0
        entry["by_kv_cycles"][row["kv"] or "other"] += row["cycles"]
    for entry in groups.values():
        entry["percent_leaf_cycles"] = 100 * entry["cycles"] / total if total else None
        entry["mean_cycles_per_nonzero_event"] = (
            entry["cycles"] / (entry["events"] - entry["zero_events"])
            if entry["events"] > entry["zero_events"]
            else None
        )
    return sorted(groups.values(), key=lambda x: x["cycles"], reverse=True)


def layout_breakdown(rows: list[dict]) -> dict:
    by_type = Counter()
    by_owner = Counter()
    for row in rows:
        if row["stage"] != "layout_precision":
            continue
        by_type[row["op_type"]] += row["cycles"]
        name = row["name"]
        if re.match(r"tq_key_\d+_tile\d+_restored_hub", name):
            owner = "past_key_restored_transpose"
        elif name.startswith("tq_attn_"):
            owner = "tiled_attention_layout"
        elif name.startswith("tq_"):
            owner = "codec_layout"
        elif name.startswith("fp16_attn_"):
            owner = "fp16_attention_layout"
        else:
            owner = "other_model_layout"
        by_owner[owner] += row["cycles"]
    return {
        "by_compiled_op_type_cycles": dict(by_type),
        "by_owner_cycles": dict(by_owner),
    }


def analyze(report: dict, bundle: Path) -> tuple[dict, list[dict]]:
    graphs = {}
    rows = []
    profiles = []
    for profile in report["op_profiles"]:
        local = []
        inclusive_cycles = 0
        timing = defaultdict(int)
        for part in profile["parts"]:
            key = profile["ar"], part["part"]
            if key not in graphs:
                graphs[key] = GraphStages(bundle, *key)
            graph = graphs[key]
            for event in part["raw_events"]:
                if event["depth"] == 0 and event["unit"] == 1:
                    timing[f"type_{event['type']}_us"] += event["value"]
                if event["type"] == 3003 and event["unit"] == 3:
                    inclusive_cycles += event["value"]
            for event in leaf_nodes(part["raw_events"]):
                local.append(
                    {
                        "step": profile["step"],
                        "part": part["part"],
                        "ar": profile["ar"],
                        "cached_before": profile["cached_before"],
                        "cycles": event["value"],
                        **graph.classify(event["identifier"]),
                    }
                )
        rows.extend(local)
        profiles.append(
            {
                **{
                    k: profile[k]
                    for k in (
                        "step",
                        "ar",
                        "new_tokens",
                        "cached_before",
                        "past_slots",
                        "prepare_s",
                        "commit_s",
                        "part_s",
                        "profile_read_s",
                    )
                },
                "graph_inclusive_cycles": inclusive_cycles,
                "leaf_node_cycles_sum": sum(r["cycles"] for r in local),
                "node_events": len(local),
                "zero_node_events": sum(r["cycles"] == 0 for r in local),
                "timing_events": dict(timing),
                "stages": aggregate(local),
            }
        )
    prefill = [r for r in rows if r["step"].startswith("prefill_")]
    decode_last = [r for r in rows if r["step"] == "decode_126"]
    return {
        "profiles": profiles,
        "prefill_all_stages": aggregate(prefill),
        "decode_last_stages": aggregate(decode_last),
        "prefill_layout": layout_breakdown(prefill),
        "decode_last_layout": layout_breakdown(decode_last),
        "decode_last_top_nodes": sorted(
            decode_last, key=lambda r: r["cycles"], reverse=True
        )[:40],
        "observability": {
            stage: {
                "compiled_op_type": optype,
                "events": sum(
                    r["stage"] == stage and r["op_type"] == optype for r in decode_last
                ),
                "zero_events": sum(
                    r["stage"] == stage and r["op_type"] == optype and r["cycles"] == 0
                    for r in decode_last
                ),
            }
            for stage, optype in (
                ("encode.rotate", "FullyConnected"),
                ("attention.av", "MatMul"),
                ("attention.qk", "MatMul"),
            )
        },
        "temperature_before_tenths_c": report["temperature_before_tenths_c"],
        "temperature_after_tenths_c": report["temperature_after_tenths_c"],
        "generated_tokens": len(report["generated"]),
        "attribution_note": "Semantic attribution to named compiled nodes; fused work may be charged to one surviving name. Not independent stage wall times.",
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    identity = json.loads((args.reports / "experiment.json").read_text())
    summary: dict[str, Any] = {
        "diagnostic_only": True,
        "units": "raw measured leaf NODE cycles; not microseconds",
        "denominator": "sum of leaf NODE cycle events (excludes graph parents)",
        "limitations": [
            "One diagnostic generation session per group; no variance estimate or thermal control.",
            "Detailed instrumentation changes scheduling/timing; historical unprofiled tok/s remains authoritative.",
            "Zero node cycles do not prove zero cost; fusion/elimination/uninstrumented work is possible.",
            "Compiler-generated unnamed events remain unattributed rather than guessed.",
            "Output shapes are static compiled shapes, not measured DDR traffic.",
        ],
        "groups": {},
    }
    for group in ("fp16", "turboquant"):
        report = json.loads((args.reports / f"{group}_profile_once.json").read_text())
        result, rows = analyze(report, Path(identity["groups"][group]["bundle"]))
        reference = (
            Path(identity["reference_experiment"]).parent / f"{group}_long_once.json"
        )
        previous = json.loads(reference.read_text())
        result["generated_ids_match_unprofiled_reference"] = (
            report["generated"] == previous["generated"]
        )
        summary["groups"][group] = result
        with (args.reports / f"{group}_node_stages.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(group)
        for phase, entries in (
            ("prefill", result["prefill_all_stages"]),
            ("decode_126", result["decode_last_stages"]),
        ):
            print(phase)
            for entry in entries:
                print(
                    f"  {entry['stage']:30} {entry['cycles'] / 1e6:10.3f} Mcycles {entry['percent_leaf_cycles']:6.2f}% zero={entry['zero_events']}/{entry['events']}"
                )
    (args.reports / "stage_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
