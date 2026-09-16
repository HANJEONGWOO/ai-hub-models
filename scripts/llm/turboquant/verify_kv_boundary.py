# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Check the KV quantization boundary of converted graphs from ``qairt-dlc-info`` output.

For every KV tensor a profile changes, this walks the final QNN op table
(``{graph}.dlcinfo.txt`` written by ``convert_parts.py``) and records:

- the write path from the first non-pass-through op (the tap) to
  ``past_{kind}_{L}_out``: every tensor must be 16-bit (uFxp_16 on the tap's
  grid or float16) and no 8-bit activation may feed it, so the codec (or the
  uFxp_16 cache output, whose input and output grids must match) reads the
  value before the KV-specific int8 grid;
- that the codec reads ``past_{kind}_{L}_out`` directly or through one
  16-bit -> float16 Convert, and that packed / norm I/O carry no encodings;
- the read path from the cache input (or the codec's restored tensor) to the
  first int8 conversion, which is expected at the attention-side Concat, whose
  other (new-token) input must still be the exported int8 tensor.

Usage:

    python scripts/llm/turboquant/verify_kv_boundary.py \
        --bundle ~/.qaihm/tmp/turboquant/qwen3_1_7b_k4_v4_cl1024 \
        --report ~/.qaihm/tmp/turboquant/reports/boundary_k4_v4.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qai_hub_models.models.templates.llm.turboquant.config import (
    TurboQuantConfig,
    get_profile,
)

TENSOR = re.compile(
    r"^(?P<name>\S+) \(data type: (?P<dtype>[^;]+); tensor dimension: "
    r"\[(?P<dims>[^\]]*)\]; tensor type: (?P<ttype>[^)]+)\)"
)
KV_IO = re.compile(
    r"(past_(key|value)_(\d+)_(in|out)|tq_(key|value)_(\d+)_(packed|norm)_(in|out))"
)
PASS_THROUGH = {
    "Concat",
    "Transpose",
    "Reshape",
    "StridedSlice",
    "Squeeze",
    "ExpandDims",
}
INT8_TYPES = {"uFxp_8", "sFxp_8"}
WIDE_TYPES = {"uFxp_16", "sFxp_16", "Float_16", "Float_32"}


@dataclass
class Tensor:
    name: str
    dtype: str
    dims: str
    ttype: str


@dataclass
class Op:
    ident: str
    name: str
    op_type: str
    inputs: list[Tensor] = field(default_factory=list)
    outputs: list[Tensor] = field(default_factory=list)
    params: list[str] = field(default_factory=list)


@dataclass
class DlcInfo:
    ops: list[Op]
    producer: dict[str, Op]
    consumers: dict[str, list[Op]]
    io_tables: dict[str, dict[str, dict[str, str]]]  # "input"/"output" -> name -> row

    def encoding_text(self, tensor: str) -> str | None:
        for op in self.ops:
            for p in op.params:
                if p.startswith(f"{tensor} encoding :"):
                    return p.split(":", 1)[1].strip()
        return None


def _tensor(cell: str) -> Tensor | None:
    m = TENSOR.match(cell.strip())
    if m is None:
        return None
    return Tensor(m["name"], m["dtype"], m["dims"], m["ttype"])


def parse_dlcinfo(path: Path) -> DlcInfo:
    ops: list[Op] = []
    io_tables: dict[str, dict[str, dict[str, str]]] = {"input": {}, "output": {}}
    table: str | None = None
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        head = cells[0]
        if head == "Id":
            table = "ops"
            continue
        if head == "Input Name":
            table = "input"
            continue
        if head == "Output Name":
            table = "output"
            continue
        if table in ("input", "output") and len(cells) >= 4:
            io_tables[table][cells[0]] = {
                "dims": cells[1],
                "dtype": cells[2],
                "encoding": cells[3],
            }
            continue
        if table != "ops" or len(cells) < 8:
            continue
        if head:
            ops.append(Op(head, cells[1], cells[2]))
        op = ops[-1]
        for cell, target in ((cells[3], op.inputs), (cells[4], op.outputs)):
            t = _tensor(cell)
            if t is not None:
                target.append(t)
        if cells[7]:
            op.params.append(cells[7])
    producer: dict[str, Op] = {}
    consumers: dict[str, list[Op]] = defaultdict(list)
    for op in ops:
        for t in op.outputs:
            producer[t.name] = op
        for t in op.inputs:
            consumers[t.name].append(op)
    return DlcInfo(ops, producer, consumers, io_tables)


def walk_back(
    info: DlcInfo, start: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pass-through tensors behind ``start`` and the boundary ops that feed them."""
    chain: list[dict[str, Any]] = []
    boundary: list[dict[str, Any]] = []
    seen: set[str] = set()
    stack = [start]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        op = info.producer.get(name)
        if op is None:
            boundary.append({"tensor": name, "op": None, "op_type": "graph_input"})
            continue
        out = next(t for t in op.outputs if t.name == name)
        # The surgery's cache-branch guard (an exact Max(x, x)) is walked through.
        if op.op_type in PASS_THROUGH or op.name.endswith("_tq_cache_guard"):
            chain.append(
                {
                    "tensor": name,
                    "dtype": out.dtype,
                    "op": op.name,
                    "op_type": op.op_type,
                }
            )
            # Only the first input of a StridedSlice/Reshape/Transpose carries data;
            # Concat carries data on every input (the guard's inputs are one tensor).
            data_inputs = op.inputs if op.op_type == "Concat" else op.inputs[:1]
            stack.extend(t.name for t in data_inputs)
        else:
            boundary.append(
                {
                    "tensor": name,
                    "dtype": out.dtype,
                    "op": op.name,
                    "op_type": op.op_type,
                    "inputs": [
                        {
                            "tensor": t.name,
                            "dtype": t.dtype,
                            "static": t.ttype == "STATIC",
                        }
                        for t in op.inputs
                    ],
                    "input_producers": [
                        info.producer[t.name].op_type
                        if t.name in info.producer
                        else "graph_input"
                        for t in op.inputs
                    ],
                }
            )
    return chain, boundary


def walk_forward(
    info: DlcInfo, start: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pass-through tensors after ``start`` and the first non-pass-through consumers."""
    chain: list[dict[str, Any]] = []
    boundary: list[dict[str, Any]] = []
    seen: set[str] = set()
    stack = [start]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        for op in info.consumers.get(name, []):
            if op.op_type in PASS_THROUGH:
                for t in op.outputs:
                    chain.append(
                        {
                            "tensor": t.name,
                            "dtype": t.dtype,
                            "op": op.name,
                            "op_type": op.op_type,
                        }
                    )
                    stack.append(t.name)
            else:
                entry: dict[str, Any] = {
                    "from": name,
                    "op": op.name,
                    "op_type": op.op_type,
                    "outputs": [
                        {"tensor": t.name, "dtype": t.dtype} for t in op.outputs
                    ],
                }
                if op.op_type == "Convert":
                    nxt = [
                        c for t in op.outputs for c in info.consumers.get(t.name, [])
                    ]
                    entry["consumers"] = [
                        {
                            "op": c.name,
                            "op_type": c.op_type,
                            "inputs": [
                                {"tensor": t.name, "dtype": t.dtype} for t in c.inputs
                            ],
                            "outputs": [
                                {"tensor": t.name, "dtype": t.dtype} for t in c.outputs
                            ],
                            "encoding": info.encoding_text(c.outputs[0].name)
                            if c.outputs
                            else None,
                        }
                        for c in nxt
                    ]
                boundary.append(entry)
    return chain, boundary


def check_graph(
    info: DlcInfo, config: TurboQuantConfig, baseline: DlcInfo | None
) -> dict[str, Any]:
    layers = sorted(
        {
            int(m.group(3) or m.group(6))
            for name in list(info.io_tables["input"])
            if (m := KV_IO.fullmatch(name))
        }
    )
    report: dict[str, Any] = {"layers": layers, "kv": [], "violations": []}
    violations = report["violations"]
    for layer in layers:
        for kind in ("key", "value"):
            spec = config.key if kind == "key" else config.value
            if not spec.modifies_graph:
                continue
            present = f"past_{kind}_{layer}_out"
            entry: dict[str, Any] = {"kind": kind, "layer": layer}
            if present not in info.producer:
                violations.append(f"{present}: not produced by any op")
                continue
            present_dtype = info.producer[present].outputs[0].dtype
            for t in info.producer[present].outputs:
                if t.name == present:
                    present_dtype = t.dtype
            entry["present_dtype"] = present_dtype
            chain, boundary = walk_back(info, present)
            entry["write_chain_ops"] = sorted({c["op_type"] for c in chain})
            entry["write_chain_dtypes"] = sorted({c["dtype"] for c in chain})
            entry["write_boundary"] = boundary
            # The cache chain carries the tap's 16-bit grid (or float16);
            # the only int8 tensors allowed near it are static weights.
            if present_dtype not in WIDE_TYPES:
                violations.append(f"{present}: dtype {present_dtype}, expected 16-bit")
            bad = [c for c in chain if c["dtype"] not in WIDE_TYPES]
            if bad:
                violations.append(
                    f"{present}: narrow tensors on the write path: {bad[:3]}"
                )
            for b in boundary:
                if b.get("dtype", "") not in WIDE_TYPES:
                    violations.append(
                        f"{present}: tap {b['tensor']} is {b.get('dtype')}"
                    )
                for i in b.get("inputs", []):
                    if i["dtype"] in INT8_TYPES and not i["static"]:
                        violations.append(
                            f"{present}: 8-bit tensor {i['tensor']} feeds {b['op']}"
                        )
                # For a Convert boundary the integer tap is its input; record
                # its grid here and in the baseline bundle (same range, 16 bits).
                tap = b["tensor"]
                if b["op_type"] == "Convert" and b.get("inputs"):
                    tap = b["inputs"][0]["tensor"]
                b["tap_tensor"] = tap
                b["tap_encoding"] = info.encoding_text(tap)
                if baseline is not None:
                    b["tap_encoding_baseline"] = baseline.encoding_text(tap)
            entry["tap_ops"] = sorted({b["op_type"] for b in boundary})

            if spec.is_polar:
                consumers = info.consumers.get(present, [])
                codec_prefix = f"tq_{kind}_{layer}_"

                def is_codec(op: Op, prefix: str = codec_prefix) -> bool:
                    return op.name.startswith(prefix)

                # Either the codec reads the present tensor itself or through one
                # Convert that must start from a 16-bit grid.
                hops = []
                for c in consumers:
                    hop = {"op": c.name, "op_type": c.op_type, "codec": is_codec(c)}
                    if c.op_type == "Convert":
                        hop["input_dtype"] = c.inputs[0].dtype
                        hop["consumers"] = [
                            {"op": n.name, "codec": is_codec(n)}
                            for t in c.outputs
                            for n in info.consumers.get(t.name, [])
                        ]
                    hops.append(hop)
                entry["present_consumers"] = hops
                ok = bool(hops) and all(
                    h["codec"]
                    or (
                        h["op_type"] == "Convert"
                        and h["input_dtype"] in WIDE_TYPES
                        and all(n["codec"] for n in h["consumers"])
                    )
                    for h in hops
                )
                if not ok:
                    violations.append(
                        f"{present}: codec does not read the 16-bit present: {hops}"
                    )
                for io_kind, table in (("in", "input"), ("out", "output")):
                    for part, dtype in (("packed", "Uint_8"), ("norm", "Float_16")):
                        name = f"tq_{kind}_{layer}_{part}_{io_kind}"
                        row = info.io_tables[table].get(name)
                        if row is None:
                            violations.append(f"{name}: missing from {table} table")
                            continue
                        if (
                            row["dtype"] != dtype
                            or "No encoding" not in row["encoding"]
                        ):
                            violations.append(f"{name}: {row}")
                read_start = f"tq_{kind}_{layer}_restored" + (
                    "_hub" if kind == "key" else ""
                )
            else:
                for io_kind, table in (("in", "input"), ("out", "output")):
                    name = f"past_{kind}_{layer}_{io_kind}"
                    row = info.io_tables[table].get(name)
                    if row is None:
                        violations.append(f"{name}: missing from {table} table")
                        continue
                    entry[f"io_{io_kind}"] = row
                    if row["dtype"] != "uFxp_16" or "No encoding" in row["encoding"]:
                        violations.append(f"{name}: {row}")
                if (
                    "io_in" in entry
                    and "io_out" in entry
                    and (entry["io_in"]["encoding"] != entry["io_out"]["encoding"])
                ):
                    violations.append(
                        f"past_{kind}_{layer}: cache input/output grids differ"
                    )
                read_start = f"past_{kind}_{layer}_in"

            if read_start not in info.consumers:
                violations.append(f"{read_start}: no consumers")
            else:
                chain, boundary = walk_forward(info, read_start)
                entry["read_chain_dtypes"] = sorted({c["dtype"] for c in chain})
                entry["read_boundary"] = boundary
                bad = [c for c in chain if c["dtype"] not in WIDE_TYPES]
                if bad:
                    violations.append(f"{read_start}: narrow read path: {bad[:3]}")
                for b in boundary:
                    if b["op_type"] != "Convert":
                        violations.append(
                            f"{read_start}: unexpected consumer {b['op']} ({b['op_type']})"
                        )
                    for c in b.get("consumers", []):
                        # The past meets the new-token tensor at the attention
                        # Concat and reaches the 16x8 MatMul as int8. QAIRT puts
                        # the Convert either on the Concat inputs (int8 Concat)
                        # or after a 16-bit Concat, right before the MatMul;
                        # both keep the MatMul's int8 K/V input.
                        out_dtypes = {o["dtype"] for o in c["outputs"]}
                        converted = {o["tensor"] for o in b["outputs"]}
                        others = [
                            i["dtype"]
                            for i in c["inputs"]
                            if i["tensor"] not in converted
                        ]
                        conv_dtypes = {o["dtype"] for o in b["outputs"]}
                        at_concat = (
                            c["op_type"] == "Concat"
                            and not out_dtypes - INT8_TYPES
                            and bool(others)
                            and not set(others) - INT8_TYPES
                        )
                        at_matmul = (
                            c["op_type"] == "MatMul"
                            and not conv_dtypes - INT8_TYPES
                            and b["from"].startswith("cat_")
                        )
                        entry["attention_int8_at"] = (
                            "concat_inputs" if at_concat else "concat_output"
                        )
                        if not (at_concat or at_matmul):
                            violations.append(
                                f"{read_start}: int8 conversion feeds {c['op']} "
                                f"({c['op_type']}, out {sorted(out_dtypes)}, other "
                                f"inputs {others}), not the attention Concat/MatMul"
                            )
            report["kv"].append(entry)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--profile", default=None, help="Defaults to convert_report.json"
    )
    parser.add_argument("--graphs", nargs="*", default=[])
    parser.add_argument("--baseline-bundle", type=Path, default=None)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    bundle = args.bundle.expanduser()
    convert_report = json.loads((bundle / "convert_report.json").read_text())
    profile = args.profile or convert_report["profile"]
    config = get_profile(profile)
    graphs = args.graphs or sorted(
        p.stem.removesuffix(".dlcinfo") for p in bundle.glob("*.dlcinfo.txt")
    )
    out: dict[str, Any] = {"bundle": str(bundle), "profile": profile, "graphs": {}}
    failed = False
    for name in graphs:
        info = parse_dlcinfo(bundle / f"{name}.dlcinfo.txt")
        if not any(KV_IO.fullmatch(n) for n in info.io_tables["input"]):
            out["graphs"][name] = {"skipped": "no KV I/O"}
            continue
        baseline = None
        if args.baseline_bundle is not None:
            path = args.baseline_bundle.expanduser() / f"{name}.dlcinfo.txt"
            if path.exists():
                baseline = parse_dlcinfo(path)
        report = check_graph(info, config, baseline)
        out["graphs"][name] = report
        status = (
            "OK" if not report["violations"] else f"FAIL ({len(report['violations'])})"
        )
        print(
            f"{name}: layers {report['layers']} kv entries {len(report['kv'])} -> {status}"
        )
        for v in report["violations"][:10]:
            print("   ", v)
        failed |= bool(report["violations"])
    args.report.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.report.expanduser().write_text(json.dumps(out, indent=1) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
