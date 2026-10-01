# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Describe train-only activation range changes, without tuning on test PPL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from calibrate_fp16_attention import target_graph

from qai_hub_models.models.templates.llm.turboquant.calibration import (
    digest,
    load_calibrated,
)


def limits(enc: dict) -> tuple[float, float]:
    delta, offset = enc["scale"][0], enc["offset"][0]
    return delta * offset, delta * (offset + 2 ** enc["bw"] - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.calibration.resolve()
    manifest = json.loads((root / "calibration_manifest.json").read_text())
    entries = []
    part_reports = {}
    for name, part in manifest["parts"].items():
        model, before = target_graph(Path(part["source"]))
        after, checks = load_calibrated(root, Path(part["source"]), before)
        observed = np.load(
            root / Path(part["encodings_file"]).parent / "activation_minmax.npy"
        )
        if observed.shape != (len(before["activation_encodings"]), 2):
            raise ValueError("Range/boundary shape mismatch")
        categories = {}
        for node in model.graph.node:
            if node.op_type == "Cast" and node.input[0].startswith("fp16_attn_"):
                if node.input[0].endswith("_qk_half"):
                    categories[node.output[0]] = "qk_score"
                elif node.input[0].endswith("_av_half"):
                    categories[node.output[0]] = "av_output"
        for old, new, bounds in zip(
            before["activation_encodings"],
            after["activation_encodings"],
            observed,
            strict=True,
        ):
            low, high = limits(old)
            new_low, new_high = limits(new)
            # Half a quantization step is not clipping: nearest rounding can
            # legitimately map such a value to the endpoint code.
            outside_old = (
                bounds[0] < low - old["scale"][0] / 2
                or bounds[1] > high + old["scale"][0] / 2
            )
            outside_new = (
                bounds[0] < new_low - new["scale"][0] / 2
                or bounds[1] > new_high + new["scale"][0] / 2
            )
            entries.append(
                {
                    "part": name,
                    "name": old["name"],
                    "category": categories.get(old["name"], "other"),
                    "old_range": [low, high],
                    "new_range": [new_low, new_high],
                    "observed_range": bounds.tolist(),
                    "scale_ratio": new["scale"][0] / old["scale"][0],
                    "observed_outside_old": bool(outside_old),
                    "observed_outside_new": bool(outside_new),
                }
            )
        part_reports[name] = {
            k: checks[k] for k in ("activation_count", "changed_count")
        }
    summaries = {}
    for category in ("all", "qk_score", "av_output", "other"):
        selected = [
            e for e in entries if category == "all" or e["category"] == category
        ]
        summaries[category] = {
            "boundaries": len(selected),
            "observed_outside_old": sum(e["observed_outside_old"] for e in selected),
            "observed_outside_new": sum(e["observed_outside_new"] for e in selected),
            "scale_ratio_p10_p50_p90": np.quantile(
                [e["scale_ratio"] for e in selected], [0.1, 0.5, 0.9]
            ).tolist()
            if selected
            else [],
        }
    report = {
        "calibration_manifest_sha256": digest(root / "calibration_manifest.json"),
        "parts": part_reports,
        "summary": summaries,
        "largest_range_changes": sorted(
            entries, key=lambda e: abs(math.log(e["scale_ratio"])), reverse=True
        )[:20],
        "notes": "Train-only min/max coverage, not a count/fraction of clipped activation values. No test-set tuning. Representative entries selected by scale-ratio magnitude, not PPL.",
        "boundaries": entries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"parts": part_reports, "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
