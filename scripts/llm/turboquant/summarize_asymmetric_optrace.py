#!/usr/bin/env python3
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Compare stage work, host timing and graph structure without inferring latency shares."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import onnx
from profile_asymmetric_optrace import GROUPS
from profile_stages_once import SCRIPTS, digest, read, save_new
from summarize_native_results import performance
from summarize_stage_profiles import GraphStages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    root = args.root
    experiment = read(root / "reference/experiment.json")
    inventory = {}
    for group in GROUPS:
        bundle = Path(experiment["groups"][group]["bundle"])
        counts = Counter()
        files = {}
        for ar in (1, 128):
            for part in (2, 3, 4):
                classifier = GraphStages(bundle, ar, part)
                path = (
                    bundle
                    / f"{'prompt' if ar == 128 else 'token'}_ar{ar}_cl1024_{part}_of_4.onnx"
                )
                files[str(path)] = digest(path)
                graph = onnx.load(path, load_external_data=False).graph
                for node in graph.node:
                    tag = classifier.classify(node.name)
                    if tag["layer"] == 0:
                        counts[ar, tag["stage"], tag["kv"], node.op_type] += 1
        inventory[group] = {
            "source_sha256": files,
            "layer0_source_node_counts": [
                {"ar": ar, "stage": stage, "kv": kv, "op_type": op, "count": n}
                for (ar, stage, kv, op), n in sorted(counts.items())
            ],
        }
    inventory_path = root / "source_inventory.json"
    if inventory_path.exists():
        if read(inventory_path) != inventory:
            raise ValueError("Frozen graph inventory changed")
    else:
        save_new(inventory_path, inventory)
    if args.inventory_only:
        return
    summaries = {
        group: read(root / "reports" / f"{group}_optrace_summary.json")
        for group in GROUPS
    }
    accelerator_times = {}
    for group in GROUPS:
        raw = read(root / "reports" / f"{group}_profile_once.json")
        accelerator_times[group] = {
            profile["step"]: {
                name: sum(
                    timing["us"]
                    for part in profile["parts"]
                    for timing in part["timings"]
                    if timing["event"] == name
                )
                / 1000
                for name in (
                    "Accelerator (execute) time",
                    "Accelerator (execute excluding wait) time",
                )
            }
            for profile in raw["op_profiles"]
        }
        del raw
    stage_rows, timing_rows = [], []
    conditions = ("prefill", "decode_0", "decode_63", "decode_126")

    def condition(group: str, name: str) -> dict:
        return (
            summaries[group]["prefill"]
            if name == "prefill"
            else summaries[group]["decode"][name]
        )

    for name in conditions:
        stages = sorted(set().union(*(condition(g, name)["stages"] for g in GROUPS)))
        for stage in stages:
            baseline = (
                condition("k4_v4", name)["stages"].get(stage, {}).get("busy_cycles", 0)
            )
            for group in GROUPS:
                entry = condition(group, name)["stages"].get(stage, {})
                cycles = entry.get("busy_cycles", 0)
                stage_rows.append(
                    {
                        "condition": name,
                        "stage": stage,
                        "group": group,
                        "busy_cycles": cycles,
                        "delta_busy_cycles_vs_k4_v4": cycles - baseline,
                        "ratio_vs_k4_v4": cycles / baseline if baseline else None,
                        "percent_compute_busy_cycles": entry.get(
                            "percent_compute_busy_cycles", 0
                        ),
                        "interval_union_cycles": entry.get("interval_union_cycles", 0),
                        "key_busy_cycles": entry.get("by_kv", {}).get("key", 0),
                        "value_busy_cycles": entry.get("by_kv", {}).get("value", 0),
                        "HVX_busy_cycles": entry.get("resources", {}).get("HVX", 0),
                        "HMX_busy_cycles": entry.get("resources", {}).get("HMX", 0),
                    }
                )
        for group in GROUPS:
            profiles = summaries[group]["detailed_counters_same_session"]["profiles"]
            selected = [
                p
                for p in profiles
                if (p["ar"] == 128 if name == "prefill" else p["step"] == name)
            ]
            timing_rows.append(
                {
                    "condition": name,
                    "group": group,
                    "diagnostic_only": True,
                    "prepare_ms": 1000 * sum(p["prepare_s"] for p in selected),
                    "commit_ms": 1000 * sum(p["commit_s"] for p in selected),
                    "qnn_calls_ms": 1000 * sum(sum(p["part_s"]) for p in selected),
                    "profile_read_ms": 1000
                    * sum(sum(p["profile_read_s"]) for p in selected),
                    "accelerator_execute_ms": sum(
                        accelerator_times[group][p["step"]][
                            "Accelerator (execute) time"
                        ]
                        for p in selected
                    ),
                    "accelerator_execute_excluding_wait_ms": sum(
                        accelerator_times[group][p["step"]][
                            "Accelerator (execute excluding wait) time"
                        ]
                        for p in selected
                    ),
                    "physical_span_cycles": condition(group, name)[
                        "physical_span_cycles"
                    ],
                    "compute_interval_union_cycles": condition(group, name)[
                        "compute_interval_union_cycles"
                    ],
                }
            )
    result = {
        "experiment_sha256": digest(root / "reference/experiment.json"),
        "analysis_scripts_sha256": {
            name: digest(SCRIPTS / name)
            for name in (
                "summarize_asymmetric_optrace.py",
                "summarize_optrace.py",
                "summarize_stage_profiles.py",
            )
        },
        "protocol_clarification": read(root / "protocol_clarification.json")
        if (root / "protocol_clarification.json").exists()
        else None,
        "units": "busy/union/span are cycles, not microseconds; host timings are instrumented wall ms",
        "limits": "One session per configuration, no variance estimate; greedy tokens differ across configurations; fused names attribute work, not causal stage wall latency.",
        "historical_uninstrumented": {
            g: performance(Path(experiment["groups"][g]["reference_report"]), 897)
            for g in GROUPS
        },
        "generated_ids_match_reference": {
            g: summaries[g]["generated_ids_match_reference"] for g in GROUPS
        },
        "stage_comparison": stage_rows,
        "instrumented_host_timings": timing_rows,
        "source_inventory": str(inventory_path),
    }
    save_new(root / "comparison.json", result)
    for filename, rows in (
        ("stage_comparison.csv", stage_rows),
        ("instrumented_host_timings.csv", timing_rows),
    ):
        with (root / filename).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
