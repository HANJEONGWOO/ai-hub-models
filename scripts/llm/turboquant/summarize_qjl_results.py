# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Compare int16 KV, Dense+Native K4/V4, and orthogonal K3+QJL/V4."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from summarize_native_results import invariants, performance, read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--encoder-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.reports
    groups = ("baseline_int16", "dense_native", "qjl_native")
    report = {
        "performance_sessions_per_configuration_condition": 1,
        "conditions": {
            "short": {"prompt_tokens": 35, "generated_tokens": 128},
            "long": {"prompt_tokens": 897, "generated_tokens": 128},
        },
        "measurement_notes": "Profiling disabled; loading excluded from TTFT. No warmup exclusion, temperature control, or variance estimate. Baseline fixed C1024; both TurboQuant groups use C128/256/512/1024. All long decode steps use C1024. Quality and functional runs are separate.",
        "bit_budget": {
            "dense_native": "K4/V4 + one FP16 scale each",
            "qjl_native": "K3+1/V4 + two FP16 K scales and one FP16 V scale",
        },
    }
    hashes, inputs, packages = {}, {}, {}
    current_policies = {}
    device = None
    for condition, prompt in (("short", 35), ("long", 897)):
        metrics = {}
        for group in groups:
            path = root / f"perf_{group}_{condition}_once.json"
            data = read(path)
            digest = data["assets"]["tokens_sha256"]
            if condition in inputs and inputs[condition] != digest:
                raise ValueError(f"Different input: {path}")
            inputs[condition] = digest
            if device is not None and device != data["device"]:
                raise ValueError(f"Different device: {path}")
            device = data["device"]
            if group in hashes and hashes[group] != data["config_hash"]:
                raise ValueError(f"Changed configuration: {path}")
            hashes[group] = data["config_hash"]
            current_policy = data.get("quantize_current_kv", False)
            if group in current_policies and current_policies[group] != current_policy:
                raise ValueError(f"Changed current-KV policy: {path}")
            current_policies[group] = current_policy
            if group != "baseline_int16":
                packages[group] = data["native_decoder"]
            metrics[group] = performance(path, prompt)
        report[condition] = metrics
        report[condition + "_ratios"] = {
            "qjl_vs_dense_decode": metrics["qjl_native"]["decode_tok_per_s"]
            / metrics["dense_native"]["decode_tok_per_s"],
            "qjl_vs_int16_decode": metrics["qjl_native"]["decode_tok_per_s"]
            / metrics["baseline_int16"]["decode_tok_per_s"],
        }
    if packages["dense_native"] != packages["qjl_native"]:
        raise ValueError("Different Native decoder packages")
    if current_policies["dense_native"] != current_policies["qjl_native"]:
        raise ValueError("Dense and QJL measurements use different current-KV policies")
    report.update(
        {
            "config_hashes": hashes,
            "quantize_current_kv": current_policies,
            "input_sha256": inputs,
            "device": device,
            "native_decoder": packages["dense_native"],
        }
    )
    quality, quality_inputs = {}, {}
    for group in groups:
        windows = []
        for window in range(4):
            path = root / f"score_{group}_w{window}.json"
            data = read(path)
            if (
                data["mode"] != "score"
                or data["scored_tokens"] != 1023
                or data["config_hash"] != hashes[group]
                or data["device"] != device
            ):
                raise ValueError(f"Unexpected quality report: {path}")
            digest = data["assets"]["tokens_sha256"]
            if window in quality_inputs and quality_inputs[window] != digest:
                raise ValueError(f"Changed PPL input: {path}")
            quality_inputs[window] = digest
            windows.append(
                {"window": window, "nll_sum": data["nll_sum"], "report": str(path)}
            )
        nll = sum(w["nll_sum"] for w in windows)
        quality[group] = {
            "ppl": math.exp(nll / 4092),
            "nll_sum": nll,
            "scored_tokens": 4092,
            "windows": windows,
            "reused_prior_measurement": False,
        }
    report["quality"] = quality
    report["quality_input_sha256"] = quality_inputs
    report["invariants"] = invariants(root, "qjl_native")
    audit_path = root / "boundary_qjl_native.json"
    audit = read(audit_path)
    report["graph_audit"] = {
        "report": str(audit_path),
        "passed": audit["passed"],
        "graphs": len(audit["graphs"]),
        "native_decoder_ops": sum(
            g["native_decoder_ops"] for g in audit["graphs"].values()
        ),
        "max_fp16_tile_bytes": max(
            g["max_rotated_intermediate_bytes"] for g in audit["graphs"].values()
        ),
    }
    encoder = read(args.encoder_report)
    report["encoder_gate"] = {
        "report": str(args.encoder_report),
        "passed": encoder["passed"],
        "tolerances": encoder["tolerances"],
    }
    with args.out.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("short", "long", "quality", "invariants", "encoder_gate")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
