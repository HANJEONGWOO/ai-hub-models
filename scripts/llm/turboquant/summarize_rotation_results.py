# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Summarize int16, FWHT+Native and dense+Native one-run device comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from summarize_native_results import invariants, performance, read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.reports.expanduser()
    stems = {
        "baseline_int16": "baseline_int16_dense_control",
        "fwht_native": "fwht_native_dense_control",
        "dense_native": "dense_native",
    }
    report = {
        "performance_sessions_per_configuration_condition": 1,
        "conditions": {
            "short": {"prompt_tokens": 35, "generated_tokens": 128},
            "long": {"prompt_tokens": 897, "generated_tokens": 128},
        },
        "measurement_notes": (
            "Profiling disabled; loading excluded from TTFT. No warmup exclusion, "
            "temperature control or variance estimate. Baseline uses fixed C1024; "
            "both TurboQuant variants use C128/256/512/1024. "
            "All long-prompt decode steps use C1024. Quality/functional runs are separate."
        ),
        "qjl": False,
    }
    hashes = {}
    inputs = {}
    device = None
    native_package = None
    for condition, prompt in (("short", 35), ("long", 897)):
        metrics = {}
        for name, stem in stems.items():
            path = root / f"perf_{stem}_{condition}_once.json"
            data = read(path)
            selected_input = data["assets"]["tokens_sha256"]
            if condition in inputs and inputs[condition] != selected_input:
                raise ValueError(f"Input changed between configurations: {path}")
            inputs[condition] = selected_input
            if device is not None and data["device"] != device:
                raise ValueError(f"Device changed: {path}")
            device = data["device"]
            if name != "baseline_int16":
                package = data["native_decoder"]
                if native_package is not None and package != native_package:
                    raise ValueError(f"Native package changed: {path}")
                native_package = package
            metrics[name] = performance(path, prompt)
            digest = data["config_hash"]
            if name in hashes and hashes[name] != digest:
                raise ValueError(f"Config changed between conditions: {name}")
            hashes[name] = digest
        report[condition] = metrics
        report[condition + "_ratios"] = {
            "dense_vs_fwht_decode": metrics["dense_native"]["decode_tok_per_s"]
            / metrics["fwht_native"]["decode_tok_per_s"],
            "dense_vs_int16_decode": metrics["dense_native"]["decode_tok_per_s"]
            / metrics["baseline_int16"]["decode_tok_per_s"],
        }
    report["config_hashes"] = hashes
    report["input_sha256"] = inputs
    report["device"] = device
    report["native_decoder"] = native_package
    report["quality"] = {}
    quality_inputs = {}
    for name, stem in stems.items():
        windows = []
        for i in range(4):
            path = root / f"score_{stem}_w{i}.json"
            data = read(path)
            selected_input = data["assets"]["tokens_sha256"]
            if i in quality_inputs and quality_inputs[i] != selected_input:
                raise ValueError(f"PPL input changed between configurations: {path}")
            quality_inputs[i] = selected_input
            if (
                data["scored_tokens"] != 1023
                or data["config_hash"] != hashes[name]
                or data["mode"] != "score"
            ):
                raise ValueError(f"Incorrect quality window/config: {path}")
            windows.append(
                {"window": i, "nll_sum": data["nll_sum"], "report": str(path)}
            )
        total = sum(w["nll_sum"] for w in windows)
        report["quality"][name] = {
            "ppl": math.exp(total / 4092),
            "nll_sum": total,
            "scored_tokens": 4092,
            "windows": windows,
            "reused_prior_measurement": False,
        }
    report["quality_input_sha256"] = quality_inputs
    report["invariants"] = invariants(root, "dense_native")
    audit_path = root / "boundary_dense_native.json"
    audit = read(audit_path)
    report["dense_graph_audit"] = {
        "report": str(audit_path),
        "passed": audit["passed"],
        "graphs": len(audit["graphs"]),
        "all_source_graphs_changed_only_rotation": all(
            g["rotation_only_source_change"] for g in audit["graphs"].values()
        ),
        "native_decoder_ops": sum(
            g["native_decoder_ops"] for g in audit["graphs"].values()
        ),
        "max_fp16_tile_bytes": max(
            g["max_rotated_intermediate_bytes"] for g in audit["graphs"].values()
        ),
    }
    encoder_path = root.parent / "p2_dense_scaled_20260918/p2_report.json"
    encoder = read(encoder_path)
    report["encoder_gate"] = {
        "report": str(encoder_path),
        "passed": all(g["passed"] for g in encoder["graphs"].values()),
        "max_scale_relative_error": max(
            c["norm_max_rel_error"]
            for g in encoder["graphs"].values()
            for c in g["cases"].values()
        ),
        "tolerance": 0.002,
        "note": "Encoder tolerance remains unchanged; this is independent of native decoder correctness.",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("short_ratios", "long_ratios", "quality")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
