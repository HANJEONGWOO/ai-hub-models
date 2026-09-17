# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Verify compiled tiled KV geometry and every tile's attention int8 boundary."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from verify_kv_boundary import check_graph, parse_dlcinfo

from qai_hub_models.models.templates.llm.turboquant.config import get_profile


def verify_graph(bundle: Path, name: str) -> dict[str, Any]:
    edits = json.loads((bundle / f"{name}.kv_edits.json").read_text())
    layers = edits.get("attention_tiles", [])
    if not layers:
        raise ValueError(f"{name}: no tiled attention manifest.")
    info = parse_dlcinfo(bundle / f"{name}.dlcinfo.txt")
    if not info.ops or not info.io_tables["input"] or not info.io_tables["output"]:
        raise ValueError(f"{name}: missing/incomplete DLC op or I/O tables.")
    config = get_profile("k4_v4")
    checks, violations = [], []
    for t in range(len(layers[0]["tiles"])):
        overrides = {}
        for layer in layers:
            tile = layer["tiles"][t]
            for kind in ("key", "value"):
                original = f"tq_{kind}_{layer['layer']}_restored" + (
                    "_hub" if kind == "key" else ""
                )
                overrides[original] = tile[f"{kind}_restored"]
                if original in info.producer:
                    violations.append(f"Full KV restore still produced: {original}")
        checked = check_graph(info, config, None, overrides)
        if checked["layers"] != sorted(layer["layer"] for layer in layers):
            violations.append("Compiled KV layer set differs from the tiling manifest.")
        violations.extend(checked["violations"])
        checks.append(checked)
    max_codec_bytes = 0
    restored = []
    by_layer = {x["layer"]: x for x in layers}
    matmuls = dict.fromkeys(by_layer, 0)
    for op in info.ops:
        match = re.match(
            r"tq_attn_(\d+)_tile\d+_head\d+_q\d+_(score|partial)$", op.name
        )
        if match:
            matmuls[int(match.group(1))] += 1
            if op.op_type != "MatMul" or [t.dtype for t in op.inputs] != [
                "uFxp_16",
                "uFxp_8",
            ]:
                violations.append(
                    f"{op.name}: attention is not the expected 16x8 MatMul."
                )
        for tensor in op.outputs:
            if tensor.dtype not in ("Float_16", "Float_32"):
                continue
            match = re.match(r"tq_(key|value)_(\d+)_tile\d+_", tensor.name)
            if not match:
                continue
            layer = by_layer[int(match.group(2))]
            dims = [int(d.strip()) for d in tensor.dims.split(",")]
            elements = math.prod(dims)
            max_codec_bytes = max(
                max_codec_bytes, elements * (2 if tensor.dtype == "Float_16" else 4)
            )
            bound = layer["kv_heads"] * layer["tile_tokens"] * config.block_size
            if elements > bound:
                violations.append(
                    f"{tensor.name}: {elements} elements > tile bound {bound}"
                )
            if tensor.name.endswith(("_restored", "_restored_hub")):
                restored.append(
                    {"name": tensor.name, "dims": dims, "dtype": tensor.dtype}
                )
    if not max_codec_bytes or not restored:
        violations.append("No compiled floating-point tile restore tensors found.")
    for layer, count in matmuls.items():
        expected = 2 * by_layer[layer]["query_heads"] * len(by_layer[layer]["tiles"])
        if count != expected:
            violations.append(
                f"Layer {layer}: expected {expected} tiled attention MatMuls, got {count}."
            )
    return {
        "layers": sorted(by_layer),
        "tile_tokens": layers[0]["tile_tokens"],
        "max_codec_intermediate_bytes": max_codec_bytes,
        "restored_tensors": restored,
        "attention_matmuls_per_layer": matmuls,
        "tile_boundary_checks": checks,
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--graphs", nargs="*", default=[])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.expanduser()
    names = args.graphs or sorted(
        p.name.removesuffix(".kv_edits.json") for p in bundle.glob("*.kv_edits.json")
    )
    results = {}
    for name in names:
        try:
            results[name] = verify_graph(bundle, name)
        except (ValueError, OSError) as error:
            results[name] = {
                "violations": [str(error)],
                "max_codec_intermediate_bytes": None,
            }
    report = {
        "bundle": str(bundle),
        "graphs": results,
        "passed": bool(results) and all(not r["violations"] for r in results.values()),
    }
    for name, result in results.items():
        print(
            name,
            "PASS" if not result["violations"] else "FAIL",
            result["max_codec_intermediate_bytes"],
            flush=True,
        )
        for error in result["violations"][:10]:
            print(error)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
