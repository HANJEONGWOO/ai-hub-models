# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Summarize the one-run native decoder experiment, excluding diagnostic timings."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def performance(path: Path, prompt: int) -> dict:
    data = read(path)
    if (
        data["mode"] != "generate"
        or data["prompt_tokens"] != prompt
        or data["context_length"] != 1024
        or len(data["sessions"]) != 1
        or len(data["generated"]) != 128
        or data.get("op_profiles")
        or not data["burst_power"]
    ):
        raise ValueError(f"Not the specified single-run performance condition: {path}")
    steps = [s for s in data["steps"] if s["kind"] == "decode"]
    counts = Counter(s["graph_context"] for s in steps)
    if len(steps) != 127 or (prompt == 897 and counts != {1024: 127}):
        raise ValueError(f"Unexpected decode steps or context selection: {path}")
    memory = next(m["mem"] for m in data["memory"] if m["at"] == "end")
    return {
        "report": str(path.resolve()),
        "bundle": data["bundle_name"],
        "native_decoder": data.get("native_decoder"),
        "ttft_ms": 1000 * data["ttft_s"],
        "prefill_tok_per_s": data["prefill_tok_per_s"],
        "decode_tok_per_s": data["decode_tok_per_s"],
        "decode_ms_per_token": 1000 / data["decode_tok_per_s"],
        "host_kv_MiB": data["kv_store_bytes"] / 2**20,
        "io_buffer_MiB": data["io_buffer_bytes"] / 2**20,
        "end_VmRSS_MiB": memory["VmRSS_kb"] / 1024,
        "process_VmHWM_MiB": memory["VmHWM_kb"] / 1024,
        "decode_context_counts": dict(counts),
        "cached_tokens_at_end": data["sessions"][0]["cached_tokens_at_end"],
        "mean_decode_prepare_ms": sum(s["prepare_s"] for s in steps)
        * 1000
        / len(steps),
        "mean_decode_qnn_ms": sum(sum(s["part_s"]) for s in steps) * 1000 / len(steps),
        "mean_decode_commit_ms": sum(s["commit_s"] for s in steps) * 1000 / len(steps),
    }


def profile(path: Path) -> list[dict]:
    data = read(path)
    results = []
    for sample in data["op_profiles"]:
        for part in sample["parts"]:
            if part["part"] == 1:
                continue
            restore = [
                op
                for op in part["codec_ops"]
                if re.match(r"tq_(key|value)_\d+_tile\d+_", op["op"])
            ]
            native = [op for op in restore if "_native_fp16" in op["op"]]
            cycles = sum(op["cycles"] for op in restore)
            results.append(
                {
                    "part": part["part"],
                    "restore_op_cycles": cycles,
                    "all_op_cycles": part["op_cycles_sum"],
                    "restore_op_cycle_fraction": cycles / part["op_cycles_sum"],
                    "native_decoder_events": len(native),
                    "native_decoder_op_cycles": sum(op["cycles"] for op in native),
                    "accelerator_execute_us": next(
                        t["us"]
                        for t in part["timings"]
                        if t["event"] == "Accelerator (execute) time"
                    ),
                }
            )
    return results


def invariants(root: Path) -> dict:
    paths = {
        "reset": root / "generation_native_lut_reset.json",
        "eos": root / "generation_native_lut_eos.json",
        "switches": root / "generation_native_lut_switches.json",
        "boundary": root / "perf_native_lut_long_once.json",
    }
    data = {name: read(path) for name, path in paths.items()}
    sessions = data["reset"]["sessions"]
    eos = data["eos"]
    switches = data["switches"]
    boundary = data["boundary"]
    checks = {
        "reset": len(sessions) == 2
        and len(sessions[0]["generated"]) == 8
        and sessions[0]["generated"] == sessions[1]["generated"]
        and all(s["cached_tokens_at_end"] == 42 for s in sessions),
        "eos": eos["stop_reason"] == "eos"
        and eos["generated"][-1] == 151645
        and len(eos["generated"]) <= 32,
        "switches": len(switches["generated"]) == 600
        and switches["sessions"][0]["cached_tokens_at_end"] == 634
        and {s["graph_context"] for s in switches["steps"] if s["kind"] == "decode"}
        == {128, 256, 512, 1024},
        "boundary": boundary["prompt_tokens"] == 897
        and len(boundary["generated"]) == 128
        and boundary["sessions"][0]["cached_tokens_at_end"] == 1024,
    }
    for name in ("switches", "boundary"):
        cached = 0
        for step in data[name]["steps"]:
            expected = min(
                c
                for c in (128, 256, 512, 1024)
                if step["ar"] < c and cached <= c - step["ar"]
            )
            checks[name] &= (
                step["cached_before"] == cached and step["graph_context"] == expected
            )
            cached += step["new_tokens"]
    return {
        name: {"passed": passed, "report": str(paths[name].resolve())}
        for name, passed in checks.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.reports.expanduser()
    stems = {
        "baseline_int16": "baseline_int16_native_control",
        "tq_graph": "graph_native_control",
        "tq_native": "native_lut",
    }
    report = {
        "performance_sessions_per_configuration_condition": 1,
        "conditions": {
            "short": {"prompt_tokens": 35, "generated_tokens": 128},
            "long": {"prompt_tokens": 897, "generated_tokens": 128},
        },
        "measurement_notes": "Profiling disabled, loading excluded from TTFT; no warmup exclusion, temperature control or variance estimate. PPL and diagnostic runs are separate.",
    }
    for condition, prompt in (("short", 35), ("long", 897)):
        metrics = {
            name: performance(root / f"perf_{stem}_{condition}_once.json", prompt)
            for name, stem in stems.items()
        }
        report[condition] = metrics
        report[condition + "_ratios"] = {
            "native_vs_graph_decode": metrics["tq_native"]["decode_tok_per_s"]
            / metrics["tq_graph"]["decode_tok_per_s"],
            "native_vs_int16_decode": metrics["tq_native"]["decode_tok_per_s"]
            / metrics["baseline_int16"]["decode_tok_per_s"],
        }
    quality = {}
    for name, filename in (
        ("baseline_int16", "quality_baseline_int16_affine_control.json"),
        ("tq_graph", "quality_rotated_scaled_buckets.json"),
    ):
        data = read(root / filename)
        quality[name] = {
            "report": str((root / filename).resolve()),
            "ppl": data["ppl"],
            "scored_tokens": data["scored_tokens"],
            "reused_unchanged_binary": True,
        }
    windows = []
    for i in range(4):
        path = root / f"score_native_lut_w{i}.json"
        data = read(path)
        if data["scored_tokens"] != 1023:
            raise ValueError(f"Incorrect PPL scoring window: {path}")
        windows.append(
            {
                "window": i,
                "ppl": data["ppl"],
                "nll_sum": data["nll_sum"],
                "scored_tokens": data["scored_tokens"],
                "report": str(path.resolve()),
            }
        )
    nll = sum(w["nll_sum"] for w in windows)
    tokens = sum(w["scored_tokens"] for w in windows)
    native_quality = {
        "name": "native_lut_buckets",
        "windows": windows,
        "scored_tokens": tokens,
        "ppl": math.exp(nll / tokens),
    }
    quality_path = root / "quality_native_lut_buckets.json"
    quality_path.write_text(json.dumps(native_quality, indent=2) + "\n")
    quality["tq_native"] = {
        "report": str(quality_path.resolve()),
        "ppl": native_quality["ppl"],
        "scored_tokens": tokens,
        "reused_unchanged_binary": False,
    }
    report["quality"] = quality
    report["invariants"] = invariants(root)
    audit_path = root / "boundary_native_lut_buckets.json"
    audit = read(audit_path)
    report["native_graph_audit"] = {
        "report": str(audit_path.resolve()),
        "passed": audit["passed"],
        "graphs": len(audit["graphs"]),
        "native_decoder_ops": sum(
            g["native_decoder_ops"] for g in audit["graphs"].values()
        ),
        "max_fp16_tile_bytes": max(
            g["max_rotated_intermediate_bytes"] for g in audit["graphs"].values()
        ),
    }
    validation_path = root.parent / "native_hvx_validation_20260918/correctness.json"
    report["standalone_native_decoder"] = {
        "report": str(validation_path.resolve()),
        "results": read(validation_path),
    }
    report["profiles"] = {
        name: profile(root / f"profile_{name}_long.json")
        for name in ("graph_native_control", "native_lut")
    }
    report["profile_note"] = (
        "Op cycle sums/fractions are not wall-clock time fractions. Diagnostic runs excluded from performance."
    )
    report["inherited_encoder_gate"] = {
        "passed": False,
        "max_scale_relative_error": 0.00256411977,
        "tolerance": 0.002,
        "source_report": str(root.parent / "p2_scaled_20260917/p2_report.json"),
        "remeasured_in_this_experiment": False,
        "note": "Prior failure of the unchanged format-2 encoder, not a new encoder measurement; native decoder correctness is tested separately.",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("short_ratios", "long_ratios", "quality")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
