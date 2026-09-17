# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Audit effective-scale I/O and rotated tiled attention in compiled DLC graphs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import onnx
from verify_kv_boundary import INT8_TYPES, WIDE_TYPES, parse_dlcinfo, walk_back


def verify_graph(bundle: Path, name: str) -> dict[str, Any]:
    manifest = json.loads((bundle / f"{name}.kv_edits.json").read_text())
    layers = manifest.get("attention_tiles", [])
    if not layers or any(x["strategy"] != "rotated_precomputed_scale" for x in layers):
        raise ValueError("Missing rotated attention manifest.")
    info = parse_dlcinfo(bundle / f"{name}.dlcinfo.txt")
    if not info.ops or not all(info.io_tables.values()):
        raise ValueError("Missing/incomplete DLC op or I/O tables.")
    errors: list[str] = []
    sizes: list[int] = []
    for io in manifest["codec_io"]:
        for side, table, tokens in (
            ("in", "input", io["past_tokens"]),
            ("out", "output", io["new_tokens"]),
        ):
            for field, dtype, width in (
                ("packed", "Uint_8", io["packed_bytes"]),
                ("norm", "Float_16", 1),
            ):
                tensor = io[f"{field}_{side}"]
                row = info.io_tables[table].get(tensor)
                if (
                    row is None
                    or row["dtype"] != dtype
                    or "No encoding" not in row["encoding"]
                ):
                    errors.append(f"Invalid {field} I/O: {tensor}: {row}")
                    continue
                dims = [int(d.strip()) for d in row["dims"].strip("[]").split(",")]
                if dims != [io["num_kv_heads"], 1, tokens, width]:
                    errors.append(f"Incorrect I/O shape: {tensor}: {dims}")
                if field == "norm" and not tensor.endswith(f"_scale_{side}"):
                    errors.append(f"Legacy norm ABI in a scale graph: {tensor}")
        present = f"past_{io['kind']}_{io['layer']}_out"
        chain, boundary = walk_back(info, present)
        if not boundary or present not in info.producer:
            errors.append(f"Missing KV write path: {present}")
        for entry in chain + boundary:
            if entry.get("dtype") not in WIDE_TYPES:
                errors.append(f"Narrow KV write path: {present}: {entry}")
        for entry in boundary:
            if any(
                t["dtype"] in INT8_TYPES and not t["static"]
                for t in entry.get("inputs", [])
            ):
                errors.append(f"Int8 input to KV writer: {present}")
        if f"tq_{io['kind']}_{io['layer']}_restored" in info.producer:
            errors.append(f"Full original-domain KV still restored: {present}")
    for layer in layers:
        number = layer["layer"]
        queries = [
            op
            for op in info.ops
            if re.fullmatch(rf"tq_attn_{number}_head\d+_q\d+_rotated", op.name)
        ]
        products = [
            op
            for op in info.ops
            if re.fullmatch(
                rf"tq_attn_{number}_tile\d+_head\d+_q\d+_(score|partial)", op.name
            )
        ]
        if len(queries) != layer["query_heads"] or len(products) != 2 * layer[
            "query_heads"
        ] * len(layer["tiles"]):
            errors.append(
                f"Layer {number}: missing query rotations or attention products"
            )
        for op in products:
            if op.op_type != "MatMul" or [t.dtype for t in op.inputs] != [
                "Float_16",
                "Float_16",
            ]:
                errors.append(f"Rotated attention is not FP16: {op.name}")
        for tile in layer["tiles"]:
            for kind in ("key", "value"):
                if tile[f"{kind}_restored"] not in info.producer:
                    errors.append(f"Missing {kind} tile {number}/{tile['start']}")
        for op in info.ops:
            if re.match(
                rf"tq_(key|value)_{number}_tile\d+_dec_", op.name
            ) and op.op_type in {
                "MatMul",
                "Reduce",
                "ReduceSum",
                "Sqrt",
                "ElementWiseSquareRoot",
                "ElementWiseDivide",
            }:
                errors.append(
                    f"Repeated inverse/norm operation: {op.name}/{op.op_type}"
                )
            for t in op.outputs:
                if t.dtype == "Float_16" and re.match(
                    rf"tq_(key|value)_{number}_tile\d+_", t.name
                ):
                    elements = math.prod(int(d.strip()) for d in t.dims.split(","))
                    sizes.append(elements * 2)
                    if elements > layer["kv_heads"] * layer["tile_tokens"] * 128:
                        errors.append(f"Oversized rotated KV tile: {t.name}")
    graph = onnx.load(bundle / f"{name}.onnx", load_external_data=False).graph
    inverses = [
        n
        for n in graph.node
        if n.op_type == "MatMul" and n.input[1].startswith("tq_rotation_fwht_s542_")
    ]
    if len(inverses) != sum(x["query_heads"] for x in layers):
        errors.append("Inverse V rotation is not once per query head.")
    if not sizes:
        errors.append("No FP16 rotated tile tensors found.")
    return {
        "layers": [x["layer"] for x in layers],
        "past_tokens": layers[0]["past_tokens"],
        "new_tokens": layers[0]["new_tokens"],
        "max_rotated_intermediate_bytes": max(sizes, default=0),
        "violations": errors,
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
        except (ValueError, OSError, KeyError) as error:
            results[name] = {"violations": [str(error)]}
        print(
            name,
            "FAIL" if results[name]["violations"] else "PASS",
            results[name]["violations"][:3],
            flush=True,
        )
    passed = bool(results) and all(not r["violations"] for r in results.values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {"bundle": str(bundle), "passed": passed, "graphs": results}, indent=2
        )
        + "\n"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
