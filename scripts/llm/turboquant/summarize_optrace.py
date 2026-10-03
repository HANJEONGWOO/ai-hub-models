#!/usr/bin/env python3
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Attribute physical HTP execution to the existing TurboQuant stage taxonomy.

Use actual HVX/HMX intervals from Core Overview tracks, not duplicate QNN views
or Non Executed Tensors. Durations in this SDK's optrace are CYCLES, not us.
Busy-cycle shares sum work across parallel lanes and are not latency shares.
DMA, synchronization/waits, interval unions and graph latency stay separate.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from profile_stages_once import read, save_new
from summarize_stage_profiles import GraphStages, analyze


def union_length(intervals: list[tuple[int, int]]) -> int:
    total = 0
    end = -1
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def classify_kernel(graph: GraphStages, event: dict) -> dict:
    args = event["args"]
    name = args.get("QNN Op Name", "")
    tag = graph.classify(name)
    tag["attribution"] = "qnn_owner"
    # HTP can fuse a GEMM and its shape-only output reshape. In that case the
    # physical HMX Conv is attributed to *_post_reshape, while the original
    # FullyConnected/MatMul gets a zero named-node counter. Only recover a
    # verified arithmetic parent for an actual HMX arithmetic kernel, NEVER
    # treat Reshape/ForceFormat/Transpose or DMA as the matrix multiplication.
    if (
        "uses_hmx" in args.get("Flags", [])
        and re.search(r"Conv|MatMul|FullyConnected", event["name"])
        and name.endswith("_post_reshape")
    ):
        parent = graph.classify(name.removesuffix("_post_reshape"))
        if parent["compiled_name_match"] and parent["op_type"] in {
            "FullyConnected",
            "MatMul",
        }:
            tag.update(
                stage=parent["stage"],
                layer=parent["layer"],
                kv=parent["kv"],
                attribution="hmx_arithmetic_fused_with_post_reshape",
            )
    if not name or event["name"] == "SystemService":
        tag["stage"] = "backend.system_service"
    return tag


def physical_events(trace: dict) -> list[dict]:
    events = trace["traceEvents"]
    overview = {
        e["pid"]
        for e in events
        if e.get("ph") == "M"
        and e.get("name") == "process_name"
        and re.fullmatch(r"Core \d+ Overview", e.get("args", {}).get("name", ""))
    }
    if not overview:
        raise ValueError("Missing Core Overview track; do not guess/double-count views")
    lanes = {
        (e["pid"], e["tid"]): e.get("args", {}).get("name", "")
        for e in events
        if e.get("ph") == "M" and e.get("name") == "thread_name"
    }
    result = []
    seen = set()
    for event in events:
        if event.get("ph") != "X" or event.get("pid") not in overview:
            continue
        lane = lanes.get((event["pid"], event["tid"]), "")
        if lane == "Non Executed Tensors":
            continue
        if lane not in {"Type: HVX", "Type: HMX", "Type: DMA"}:
            raise ValueError("Unknown physical execution lane " + lane)
        args = event["args"]
        # The args duration may be a per-op statistic shared by several HVX
        # workers. Use each lane's actual interval, not that shared statistic.
        if "Duration (cycles)" not in args or "Start Cycle" not in args:
            raise ValueError("Trace has no explicit cycle metadata")
        if event["dur"] < 0:
            raise ValueError("Negative duration")
        key = (event["pid"], event["tid"], args["ID"], event["ts"], event["dur"])
        if key in seen:
            raise ValueError("Duplicate physical event")
        seen.add(key)
        result.append({**event, "resource": lane.removeprefix("Type: ")})
    return result


def summarize_trace(trace: dict, graph: GraphStages) -> tuple[dict, list[dict]]:
    physical = physical_events(trace)
    compute = []
    resources = Counter()
    stages = {}
    intervals = defaultdict(list)
    all_intervals = []
    rows = {}
    for event in physical:
        flags = event["args"].get("Flags", [])
        cycles = event["dur"]
        resource = event["resource"]
        category = resource + (
            ".wait"
            if "dma_wait" in flags
            else ".transfer"
            if "dma" in flags
            else ".control"
            if resource == "DMA"
            else ".execute"
        )
        resources[category] += cycles
        if "uses_hmx" not in flags and "uses_hvx" not in flags:
            continue
        tag = classify_kernel(graph, event)
        stage = tag["stage"]
        entry = stages.setdefault(
            stage,
            {"busy_cycles": 0, "events": 0, "resources": Counter(), "by_kv": Counter()},
        )
        entry["busy_cycles"] += cycles
        entry["events"] += 1
        entry["resources"][resource] += cycles
        entry["by_kv"][tag["kv"] or "other"] += cycles
        interval = (event["ts"], event["ts"] + cycles)
        intervals[stage].append(interval)
        all_intervals.append(interval)
        compute.append(event)
        key = (tag["name"], stage, event["name"], resource)
        row = rows.setdefault(
            key,
            {
                **tag,
                "kernel": event["name"],
                "resource": resource,
                "busy_cycles": 0,
                "events": 0,
            },
        )
        row["busy_cycles"] += cycles
        row["events"] += 1
    for stage, entry in stages.items():
        entry["interval_union_cycles"] = union_length(intervals[stage])
    span = max((e["ts"] + e["dur"] for e in physical), default=0) - min(
        (e["ts"] for e in physical), default=0
    )
    return {
        "stages": stages,
        "resources": dict(resources),
        "physical_events": len(physical),
        "compute_events": len(compute),
        "physical_span_cycles": span,
        "compute_interval_union_cycles": union_length(all_intervals),
    }, list(rows.values())


def combine(profiles: list[dict]) -> dict:
    stages = {}
    resources = Counter()
    for profile in profiles:
        for stage, entry in profile["stages"].items():
            out = stages.setdefault(
                stage,
                {
                    "busy_cycles": 0,
                    "events": 0,
                    "interval_union_cycles": 0,
                    "resources": Counter(),
                    "by_kv": Counter(),
                },
            )
            for key in ("busy_cycles", "events", "interval_union_cycles"):
                out[key] += entry[key]
            out["resources"].update(entry["resources"])
            out["by_kv"].update(entry["by_kv"])
        resources.update(profile["resources"])
    total = sum(x["busy_cycles"] for x in stages.values())
    for entry in stages.values():
        entry["percent_compute_busy_cycles"] = (
            100 * entry["busy_cycles"] / total if total else None
        )
    return {
        "stages": stages,
        "resources": dict(resources),
        "compute_busy_cycles": total,
        "physical_span_cycles": sum(p["physical_span_cycles"] for p in profiles),
        "compute_interval_union_cycles": sum(
            p["compute_interval_union_cycles"] for p in profiles
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=["fp16", "turboquant"])
    args = parser.parse_args()
    for group in args.groups:
        output = args.root / "reports" / f"{group}_optrace_summary.json"
        if output.exists():
            raise FileExistsError(output)
        report = read(args.root / "reports" / f"{group}_profile_once.json")
        attempt = read(args.root / "reports" / f"{group}_profile_once.attempt.json")
        complete = read(args.root / "reports" / f"{group}_profile_once.complete.json")
        bundle = Path(attempt["source_bundle"])
        graphs = {}
        profiles = []
        rows = []
        sequence = 0
        for profile in report["op_profiles"]:
            parts = []
            for part in range(1, 5):
                ar = profile["ar"]
                stem = f"execute_{sequence:05d}_{'prompt' if ar == 128 else 'token'}_ar{ar}_cl1024_{part}_of_4"
                path = args.root / "rendered" / group / stem / "trace.json.gz"
                rendered = read(path.parent / "complete.json")
                if rendered["log_sha256"] != complete["logs_sha256"][stem + ".log"]:
                    raise ValueError("Trace log provenance changed")
                if (ar, part) not in graphs:
                    graphs[ar, part] = GraphStages(bundle, ar, part)
                with gzip.open(path, "rt") as stream:
                    parsed, local = summarize_trace(json.load(stream), graphs[ar, part])
                parts.append({"part": part, **parsed})
                rows.extend(
                    {"step": profile["step"], "part": part, **row} for row in local
                )
                sequence += 1
            profiles.append(
                {
                    "step": profile["step"],
                    "ar": profile["ar"],
                    "cached_before": profile["cached_before"],
                    "parts": parts,
                    **combine(parts),
                }
            )
            print("ANALYZED", group, profile["step"], flush=True)
        detailed, _ = analyze(
            {
                **report,
                "temperature_before_tenths_c": attempt["temperature_before_tenths_c"],
                "temperature_after_tenths_c": complete["temperature_after_tenths_c"],
            },
            bundle,
        )
        result = {
            "diagnostic_only": True,
            "units": "cycles; NOT microseconds",
            "denominator": "sum of executed HVX/HMX kernel durations across parallel lanes; NOT wall latency",
            "fusion_rule": "HMX arithmetic kernel assigned to a verified FullyConnected/MatMul parent when QNN owner is its post_reshape; actual layout kernels remain layout",
            "generated_ids_match_reference": complete["generated_ids_match_reference"],
            "prefill": combine([p for p in profiles if p["ar"] == 128]),
            "decode": {p["step"]: combine([p]) for p in profiles if p["ar"] == 1},
            "profiles": profiles,
            "detailed_counters_same_session": detailed,
        }
        save_new(output, result)
        with output.with_suffix(".csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
