# SPDX-License-Identifier: BSD-3-Clause
"""Compare physical HTP work by stage and bit width; never equate cycles to TTFT."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from benchmark_asymmetric_once import digest, read, save_new
from benchmark_mixed_once import GROUPS, config_for


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--groups", nargs="+", default=list(GROUPS))
    args = p.parse_args()
    stage_rows, bit_rows, latencies = [], [], []
    sources = {}
    for group in args.groups:
        summary = args.root / "reports" / f"{group}_optrace_summary.json"
        sources[str(summary)] = digest(summary)
        data = read(summary)
        config = config_for(args.experiment, group)
        for condition, values in (
            ("prefill", data["prefill"]),
            *data["decode"].items(),
        ):
            for stage, entry in values["stages"].items():
                stage_rows.append(
                    {
                        "group": group,
                        "condition": condition,
                        "stage": stage,
                        "busy_cycles": entry["busy_cycles"],
                        "percent_compute_work": entry["percent_compute_busy_cycles"],
                        "key_busy_cycles": entry["by_kv"].get("key", 0),
                        "value_busy_cycles": entry["by_kv"].get("value", 0),
                        "HVX_busy_cycles": entry["resources"].get("HVX", 0),
                        "HMX_busy_cycles": entry["resources"].get("HMX", 0),
                    }
                )
        # Same model/shape but distinct real layer values. Report per-layer work
        # and number of layers, not an isolated same-tensor kernel benchmark.
        totals = defaultdict(int)
        seen = defaultdict(set)
        with summary.with_suffix(".csv").open() as stream:
            for row in csv.DictReader(stream):
                if not row["layer"] or row["kv"] not in ("key", "value"):
                    continue
                if not row["stage"].startswith(("encode.", "decode.native_")):
                    continue
                layer = int(row["layer"])
                condition = (
                    "prefill" if row["step"].startswith("prefill") else row["step"]
                )
                bit = config.codec_for(row["kv"], layer).bits
                key = condition, row["stage"], row["kv"], bit
                totals[key] += int(row["busy_cycles"])
                seen[key].add(layer)
        for (condition, stage, kind, bit), cycles in sorted(totals.items()):
            count = len(seen[condition, stage, kind, bit])
            bit_rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "stage": stage,
                    "kv": kind,
                    "bits": bit,
                    "layer_count": count,
                    "busy_cycles": cycles,
                    "busy_cycles_per_layer": cycles / count,
                }
            )
        raw = read(args.root / "reports" / f"{group}_profile_once.json")
        per_condition = defaultdict(lambda: defaultdict(float))
        for profile in raw["op_profiles"]:
            condition = "prefill" if profile["ar"] == 128 else profile["step"]
            for part in profile["parts"]:
                for time in part["timings"]:
                    if time["event"] in (
                        "Accelerator (execute) time",
                        "Accelerator (execute excluding wait) time",
                    ):
                        per_condition[condition][time["event"]] += time["us"] / 1000
        latencies.extend(
            {"group": group, "condition": condition, **values}
            for condition, values in per_condition.items()
        )
    result = {
        "units": "physical HVX/HMX busy cycles; not microseconds, not wall-latency shares",
        "sessions_per_group": 1,
        "limits": "Bit-width comparison averages different layers/inputs of identical shape; profiler affects runtime; no variance estimate or TTFT reduction inferred from work percentages",
        "source_sha256": sources,
        "stage_comparison": stage_rows,
        "bit_width_comparison": bit_rows,
        "instrumented_accelerator_ms": latencies,
    }
    save_new(args.root / "comparison.json", result)
    for name, rows in (
        ("stage_comparison", stage_rows),
        ("bit_width_comparison", bit_rows),
    ):
        with (args.root / f"{name}.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
