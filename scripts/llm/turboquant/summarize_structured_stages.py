# SPDX-License-Identifier: BSD-3-Clause
"""Compare seven codec stages with archived controls, without new inference."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import onnx
from profile_stages_once import SCRIPTS, digest, read, save_new
from profile_structured_stages_once import CONTROLS, GROUP, validate

SEVEN = (
    ("prefill", "encode.pack"),
    ("prefill", "encode.scalar_index"),
    ("prefill", "encode.scale_correction"),
    ("decode_126", "decode.native_past"),
    ("decode_126", "encode.pack"),
    ("decode_126", "encode.scalar_index"),
    ("decode_126", "encode.scale_correction"),
)


def stage_rows(group: str, summary: dict) -> list[dict]:
    rows = []
    conditions = [("prefill", summary["prefill"]), *summary["decode"].items()]
    for condition, data in conditions:
        for stage, entry in data["stages"].items():
            cycles, events = entry["busy_cycles"], entry["events"]
            rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "stage": stage,
                    "busy_cycles": cycles,
                    "million_busy_cycles": cycles / 1e6,
                    "percent_total_compute_work": entry["percent_compute_busy_cycles"],
                    "denominator_busy_cycles": data["compute_busy_cycles"],
                    "physical_lane_events": events,
                    "mean_cycles_per_physical_lane_event": cycles / events
                    if events
                    else None,
                    "interval_union_cycles": entry["interval_union_cycles"],
                    "stage_wall_ms": None,
                    "stage_wall_s": None,
                    "time_status": "unavailable: cycle trace, no per-stage calibrated clock",
                    "key_busy_cycles": entry["by_kv"].get("key", 0),
                    "value_busy_cycles": entry["by_kv"].get("value", 0),
                    "HVX_busy_cycles": entry["resources"].get("HVX", 0),
                    "HMX_busy_cycles": entry["resources"].get("HMX", 0),
                }
            )
    return rows


def route(row: dict, k8_layers: set[int]) -> str:
    if row["kv"] == "value":
        return "V2"
    if row["kv"] == "key" and row["layer"] not in (None, ""):
        return "Structured K8" if int(row["layer"]) in k8_layers else "K4"
    return "unassigned"


def csv_new(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Empty report: " + str(path))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def instrumented_times(group: str, report: dict) -> list[dict]:
    rows = []
    for p in report["op_profiles"]:
        accelerator = sum(
            t["us"]
            for part in p["parts"]
            for t in part["timings"]
            if t["event"] == "Accelerator (execute) time"
        )
        rows.append(
            {
                "group": group,
                "sample": p["step"],
                "ar": p["ar"],
                "cached_before": p["cached_before"],
                "accelerator_execute_ms": accelerator / 1000,
                "accelerator_execute_s": accelerator / 1e6,
                "host_prepare_ms": 1000 * p["prepare_s"] if "prepare_s" in p else None,
                "host_commit_ms": 1000 * p["commit_s"] if "commit_s" in p else None,
                "qnn_calls_ms": 1000 * sum(p["part_s"]) if "part_s" in p else None,
                "profile_read_ms": 1000 * sum(p["profile_read_s"])
                if "profile_read_s" in p
                else None,
                "diagnostic_only": True,
            }
        )
    return rows


def decoder_inventory(bundle: Path) -> dict:
    inventory = {}
    for ar, label in ((1, "token"), (128, "prompt")):
        counts = Counter()
        sources = {}
        for part in (2, 3, 4):
            path = bundle / f"{label}_ar{ar}_cl1024_{part}_of_4.onnx"
            graph = onnx.load(path, load_external_data=False).graph
            counts.update(
                node.op_type for node in graph.node if node.op_type.startswith("Decode")
            )
            sources[str(path)] = digest(path)
        expected = {"Decode8Structured": 70, "Decode4": 70, "Decode2": 140}
        if dict(counts) != expected:
            raise ValueError("Unexpected decoder graph: " + str(counts))
        inventory[str(ar)] = {
            "source_sha256": sources,
            "source_op_counts": dict(counts),
            "meaning": "14 K8 / 14 K4 / 28 V2 x (four past tiles + current); source nodes, NOT executed kernel invocations",
        }
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    protocol = validate(root)
    historical = Path(protocol["historical_controls"])
    groups = (*CONTROLS, GROUP)
    all_stages, times, conditions = [], [], {}
    files = {}
    for group in groups:
        base = root if group == GROUP else historical
        summary_path = base / "reports" / f"{group}_optrace_summary.json"
        raw_path = base / "reports" / f"{group}_profile_once.json"
        attempt_path = base / "reports" / f"{group}_profile_once.attempt.json"
        summary, raw, attempt = read(summary_path), read(raw_path), read(attempt_path)
        if not summary["generated_ids_match_reference"]:
            raise ValueError("Continuation mismatch: " + group)
        if (
            attempt["prompt_tokens"],
            attempt["generated_tokens"],
            attempt["decode_steps"],
            attempt["sessions"],
        ) != (897, 128, [0, 63, 126], 1):
            raise ValueError("Historical capture protocol differs: " + group)
        if [p["cached_before"] for p in raw["op_profiles"] if p["ar"] == 1] != [
            897,
            960,
            1023,
        ]:
            raise ValueError("Cache position differs: " + group)
        source = read(Path(attempt["source_bundle"]) / "convert_report.json")
        conditions[group] = {
            "source_bundle": attempt["source_bundle"],
            "source_config_hash": source["config_hash"],
            "runner_sha256": attempt["runner_sha256"],
            "assets_sha256": attempt["assets_sha256"],
            "qairt_sdk": source["sdk"],
            "native_build_sdk_if_recorded": source["native_decoder"].get("qairt_sdk"),
            "compiled_native_sha256": source["native_decoder"]["libraries"][
                "hexagon-v81"
            ]["sha256"],
            "runtime_native_sha256": raw.get("native_decoder", {}).get("sha256"),
            "backend_api": raw["backend_api"],
            "burst_power": raw["burst_power"],
            "device": raw["device"],
            "kv_store_bytes": raw["kv_store_bytes"],
            "io_buffer_bytes": raw["io_buffer_bytes"],
            "quantize_current_kv": raw["quantize_current_kv"],
            "attention_tile": source["attention_tile"],
            "model": source.get("model"),
            "split_manifest_sha256": source.get("split_manifest_sha256"),
            "new_measurement": group == GROUP,
            "prompt_tokens": 897,
            "generated_tokens": 128,
            "context_length": 1024,
        }
        for path in (summary_path, raw_path, attempt_path):
            files[str(path)] = digest(path)
        all_stages.extend(stage_rows(group, summary))
        times.extend(instrumented_times(group, raw))
    baseline = {
        (r["condition"], r["stage"]): r["busy_cycles"]
        for r in all_stages
        if r["group"] == "k4_v4"
    }
    for row in all_stages:
        ref = baseline.get((row["condition"], row["stage"]), 0)
        row["ratio_vs_historical_k4_v4"] = row["busy_cycles"] / ref if ref else None
    seven = [
        r
        for condition, stage in SEVEN
        for r in all_stages
        if (r["condition"], r["stage"]) == (condition, stage)
    ]
    # Preserve every physical-kernel-to-source mapping, adding only route/bit
    # metadata. Do not reclassify historical controls to force a favorable split.
    mapped, totals, layer_sets = [], defaultdict(lambda: [0, 0]), defaultdict(set)
    k8 = set(protocol["selected_layers"])
    with (root / "reports" / f"{GROUP}_optrace_summary.csv").open() as stream:
        for row in csv.DictReader(stream):
            condition = "prefill" if row["step"].startswith("prefill") else row["step"]
            path = route(row, k8)
            mapped.append({**row, "condition": condition, "codec_route": path})
            if row["stage"].startswith(("encode.", "decode.native_")):
                key = condition, row["stage"], path
                totals[key][0] += int(row["busy_cycles"])
                totals[key][1] += int(row["events"])
                if row["layer"]:
                    layer_sets[key].add(int(row["layer"]))
    by_route = []
    for (condition, stage, path), (cycles, events) in sorted(totals.items()):
        count = len(layer_sets[condition, stage, path])
        by_route.append(
            {
                "condition": condition,
                "stage": stage,
                "codec_route": path,
                "busy_cycles": cycles,
                "million_busy_cycles": cycles / 1e6,
                "physical_lane_events": events,
                "observed_layers": count,
                "mean_busy_cycles_per_observed_layer": cycles / count
                if count
                else None,
                "interpretation": "K8 direct UINT8: attributed cast/store, NOT bit packing"
                if stage == "encode.pack" and path == "Structured K8"
                else "physical fused-owner attribution",
            }
        )
    for row in all_stages:
        if row["group"] == GROUP and row["stage"].startswith(
            ("encode.", "decode.native_")
        ):
            components = [
                r
                for r in by_route
                if r["condition"] == row["condition"] and r["stage"] == row["stage"]
            ]
            if sum(r["busy_cycles"] for r in components) != row["busy_cycles"]:
                raise ValueError("Route attribution does not reconcile")
            if any(
                r["codec_route"] == "unassigned" and r["busy_cycles"]
                for r in components
            ):
                raise ValueError("Unassigned codec work; inspect before reporting")
    comparison = {
        "new_sessions": {GROUP: 1},
        "control_sessions": 0,
        "source_sha256": files,
        "conditions": conditions,
        "same_assets": all(
            c["assets_sha256"] == conditions[GROUP]["assets_sha256"]
            for c in conditions.values()
        ),
        "same_sdk": len({c["qairt_sdk"] for c in conditions.values()}) == 1,
        "same_backend_api": len({c["backend_api"] for c in conditions.values()}) == 1,
        "same_device_fingerprint": all(
            c["device"] == conditions[GROUP]["device"] for c in conditions.values()
        ),
        "same_runner": len({c["runner_sha256"] for c in conditions.values()}) == 1,
        "historical_analysis_source_unchanged": {
            name: digest(SCRIPTS / name) == expected
            for name, expected in read(historical / "comparison.json")[
                "analysis_scripts_sha256"
            ].items()
        },
        "units": "physical busy cycles, not exclusive latency or TTFT shares",
        "time_limitation": "Per-stage seconds/ms unavailable; report SDK graph time separately, never convert cycles with assumed clock",
        "limits": [
            "Archived controls, not contemporaneous paired measurements; no variance or statistical superiority claim",
            "Native package versions and allocations differ; comparison is of configurations, not isolated kernel causal effects",
            "Greedy continuation tokens can differ across configurations",
            "Event count is physical lane intervals, not source-node or independent kernel invocation count",
            "Prefill sums eight chunks; decode_126 is ONE sampled step, not the entire decode session",
            "Fusion assigned to surviving QNN owner using unchanged historical classifier",
        ],
        "seven_stages": seven,
        "all_stage_samples": all_stages,
        "structured_routes": by_route,
        "instrumented_graph_times": times,
        "decoder_graph_inventory": decoder_inventory(Path(protocol["source_bundle"])),
    }
    save_new(root / "comparison.json", comparison)
    csv_new(root / "seven_stage_comparison.csv", seven)
    csv_new(root / "all_stage_samples.csv", all_stages)
    csv_new(root / "structured_route_costs.csv", by_route)
    csv_new(root / "node_stage_mapping.csv", mapped)
    csv_new(root / "instrumented_graph_times.csv", times)
    print("SAVED seven-stage comparison; historical controls not executed", flush=True)


if __name__ == "__main__":
    main()
