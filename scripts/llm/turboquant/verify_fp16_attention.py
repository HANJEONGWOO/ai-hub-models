# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Fail-closed audit of FP16 cache and QK/AV inputs in the converted QNN DLC."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import onnx
from verify_kv_boundary import INT8_TYPES, PASS_THROUGH, parse_dlcinfo

from qai_hub_models.models.templates.llm.turboquant.calibration import (
    digest,
    load_calibrated,
)
from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.fp16_attention import (
    use_fp16_kv_attention,
)
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)


def verify_graph(bundle: Path, name: str) -> dict[str, Any]:
    manifest = json.loads((bundle / f"{name}.kv_edits.json").read_text())
    heads = manifest.get("fp16_attention", [])
    if not heads or manifest["codec_io"]:
        raise ValueError("Expected uncompressed FP16 attention manifest")
    info = parse_dlcinfo(bundle / f"{name}.dlcinfo.txt")
    model = onnx.load(bundle / f"{name}.onnx", load_external_data=False)
    source = {o: n for n in model.graph.node for o in n.output}
    errors = []
    kv_names = {
        v.name
        for v in (*model.graph.input, *model.graph.output)
        if v.name.startswith("past_")
    }
    for side, values in (("input", model.graph.input), ("output", model.graph.output)):
        for value in values:
            if value.name not in kv_names:
                continue
            row = info.io_tables[side].get(value.name)
            if (
                row is None
                or row["dtype"] != "Float_16"
                or "No encoding" not in row["encoding"]
            ):
                errors.append(f"Non-FP16 or encoded cache {side}: {value.name}: {row}")
    products = {
        op.name: op
        for op in info.ops
        if re.fullmatch(r"fp16_attn_\d+_head\d+_q\d+_(qk|av)_half", op.name)
    }
    if len(products) != 2 * len(heads):
        errors.append(
            f"Missing attention products: expected {2 * len(heads)}, got {len(products)}"
        )
    for head in heads:
        for kind in ("qk", "av"):
            item = head[kind]
            op = products.get(item["output"])
            if (
                op is None
                or op.op_type != "MatMul"
                or [t.dtype for t in op.inputs] != ["Float_16", "Float_16"]
            ):
                errors.append(f"Non-FP16 QK/AV: {item['output']}")
                continue
            # Both current and past must reach this exact product from the
            # exported cache tensors, not a parallel raw-current branch.
            kind_kv = "key" if kind == "qk" else "value"
            wanted = {
                f"past_{kind_kv}_{head['layer']}_{side}" for side in ("in", "out")
            }
            visited_rhs: set[str] = set()
            pending_rhs = [op.inputs[1].name]
            while pending_rhs:
                tensor = pending_rhs.pop()
                if tensor in visited_rhs:
                    continue
                visited_rhs.add(tensor)
                if tensor in wanted:
                    continue
                producer = info.producer.get(tensor)
                if producer is not None and producer.op_type in PASS_THROUGH | {
                    "Convert",
                    "Cast",
                    "ElementWiseMultiply",
                    "ElementWiseDivide",
                }:
                    pending_rhs.extend(
                        t.name for t in producer.inputs if t.ttype != "STATIC"
                    )
            if not wanted <= visited_rhs:
                errors.append(f"Missing stored past/current KV dependency: {op.name}")
        for kind in ("key", "value"):
            current = f"fp16_attn_{head['layer']}_head{head['head']}_{kind}_current"
            node = source.get(current)
            if (
                node is None
                or node.op_type != "Slice"
                or node.input[0] != f"past_{kind}_{head['layer']}_out"
            ):
                errors.append(f"Current KV does not read stored FP16: {current}")

    # Walk every cache output back through conversions/layout/guards to the
    # first computing op. Unlike a Cast-only check this catches hidden int8.
    seen: set[str] = set()
    stack = [n for n in kv_names if n.endswith("_out")]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        op = info.producer.get(name)
        if op is None:
            errors.append(f"Missing KV writer: {name}")
            continue
        out = next(t for t in op.outputs if t.name == name)
        if out.dtype not in ("Float_16", "uFxp_16", "sFxp_16"):
            errors.append(f"Narrow or FP32 KV writer: {name}: {out.dtype}")
        if any(t.dtype in INT8_TYPES and t.ttype != "STATIC" for t in op.inputs):
            errors.append(f"Int8 KV writer input: {op.name}")
        if op.op_type in PASS_THROUGH | {"Convert", "Cast"} or op.name.endswith(
            "_tq_cache_guard"
        ):
            stack.extend(
                t.name
                for t in (op.inputs if op.op_type == "Concat" else op.inputs[:1])
                if t.ttype != "STATIC"
            )

    # Follow both past and current cache forward to QK/AV. No quantized
    # intermediates may be introduced, even when the final MatMul is FP16.
    visited: set[str] = set()
    stack = list(kv_names)
    reached: set[str] = set()
    while stack:
        name = stack.pop()
        if name in visited:
            continue
        visited.add(name)
        for op in info.consumers.get(name, []):
            if op.name in products:
                reached.add(op.name)
                continue
            if op.op_type not in PASS_THROUGH | {
                "Convert",
                "Cast",
                "ElementWiseMultiply",
                "ElementWiseDivide",
            }:
                errors.append(f"Unexpected cache reader: {op.name}/{op.op_type}")
                continue
            for tensor in op.outputs:
                if tensor.dtype != "Float_16":
                    errors.append(
                        f"Non-FP16 cache read path: {tensor.name}: {tensor.dtype}"
                    )
                stack.append(tensor.name)
    if reached != set(products):
        errors.append("Not every QK/AV product is reachable from cache I/O")
    return {
        "layers": sorted({h["layer"] for h in heads}),
        "query_heads": len(heads),
        "fp16_products": len(products),
        "kv_tensors": len(kv_names),
        "violations": errors,
    }


def verify_source(
    bundle: Path, name: str, original: Path, calibration: Path | None = None
) -> dict[str, Any]:
    """Ensure graph surgery did not change weights or unrelated live encodings."""
    before = onnx.load(original, load_external_data=False)
    after = onnx.load(bundle / f"{name}.onnx", load_external_data=False)
    params = {t.name: t for t in before.graph.initializer}
    changed = []
    count = 0
    checked_files: set[str] = set()
    for tensor in after.graph.initializer:
        if tensor.name in params:
            count += 1
            if tensor.SerializeToString() != params[tensor.name].SerializeToString():
                changed.append(f"Changed original initializer: {tensor.name}")
            for field in tensor.external_data:
                if (
                    field.key == "location"
                    and field.value not in checked_files
                    and (bundle / field.value).resolve()
                    != (original.parent / field.value).resolve()
                ):
                    changed.append(f"Different external weight file: {field.value}")
                if field.key == "location":
                    checked_files.add(field.value)
    enc_before = json.loads(original.with_suffix(".encodings").read_text())
    enc_after = json.loads((bundle / f"{name}.encodings").read_text())
    if enc_before["param_encodings"] != enc_after["param_encodings"]:
        changed.append("Changed parameter encodings")
    edits = json.loads((bundle / f"{name}.kv_edits.json").read_text())
    allowed = {
        e["tensor"]: e["after"]
        for e in edits["encoding_edits"]
        if e["action"] == "regrid"
    }
    acts = {e["name"]: e for e in enc_before["activation_encodings"]}
    if calibration is not None:
        config = get_profile("baseline_fp16_kv_fp16_attn_calibrated")
        result = use_fp16_kv_attention(
            apply_kv_profile(before, enc_before, config, 1, 1024), config
        )
        calibrated, _ = load_calibrated(calibration, original, result.encodings)
        acts = {e["name"]: e for e in calibrated["activation_encodings"]}
        allowed = {}
        if enc_after != calibrated:
            changed.append("Bundle encodings differ from verified calibration artifact")
    for enc in enc_after["activation_encodings"]:
        if enc != allowed.get(enc["name"], acts.get(enc["name"])):
            changed.append(f"Changed unrelated activation encoding: {enc['name']}")
    return {
        "original": str(original),
        "unchanged_initializers": count,
        "parameter_encodings_unchanged": enc_before["param_encodings"]
        == enc_after["param_encodings"],
        "violations": changed,
    }


def verify_shared_interfaces(bundle: Path, conversion: dict) -> dict:
    """Check raw-buffer compatibility across split graph inputs and outputs."""
    seen: dict[str, dict] = {}
    repeated = set()
    errors = []
    for part in conversion["parts"].values():
        for name in part["graphs"]:
            match = re.fullmatch(r"(.+)_\d+_of_\d+", name)
            if match is None:
                raise ValueError(f"Unrecognized split graph name: {name}")
            group = match.group(1)
            tensors = seen.setdefault(group, {})
            info = parse_dlcinfo(bundle / f"{name}.dlcinfo.txt")
            for side in ("input", "output"):
                for tensor, row in info.io_tables[side].items():
                    signature = {k: row[k] for k in ("dims", "dtype", "encoding")}
                    if tensor in tensors:
                        repeated.add((group, tensor))
                        if tensors[tensor] != signature:
                            errors.append(
                                f"Shared interface mismatch: {group}/{tensor}"
                            )
                    else:
                        tensors[tensor] = signature
    return {
        "graph_sets": len(seen),
        "shared_tensors_checked": len(repeated),
        "violations": sorted(set(errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--split-dir", type=Path, help="Also verify original weights/encodings"
    )
    args = parser.parse_args()
    conversion = json.loads((args.bundle / "convert_report.json").read_text())
    if conversion["profile"] not in (
        "baseline_fp16_kv_fp16_attn",
        "baseline_fp16_kv_fp16_attn_calibrated",
    ):
        raise ValueError("Not an FP16 attention baseline bundle")
    calibration = (
        Path(conversion["activation_calibration_dir"])
        if conversion.get("activation_calibration_dir")
        else None
    )
    if get_profile(conversion["profile"]).activation_calibrated != bool(calibration):
        raise ValueError("Calibration metadata/profile mismatch")
    if calibration and not args.split_dir:
        raise ValueError(
            "Calibrated audit requires --split-dir to verify fixed weights"
        )
    graphs = {}
    if calibration and digest(
        calibration / "calibration_manifest.json"
    ) != conversion.get("activation_calibration_sha256"):
        raise ValueError("Conversion/calibration artifact identity mismatch")
    passthrough = {}
    split = (
        json.loads((args.split_dir / "split_manifest.json").read_text())
        if args.split_dir
        else None
    )
    for part_name, part in conversion["parts"].items():
        for name, entry in part["graphs"].items():
            if entry["surgery"].get("fp16_attention_heads"):
                graphs[name] = verify_graph(args.bundle, name)
                if split:
                    original = split["parts"][part_name]
                    check = verify_source(
                        args.bundle,
                        name,
                        Path(original["bundle_dir"]) / (original["class"] + ".onnx"),
                        calibration,
                    )
                    graphs[name]["source_preservation"] = check
                    graphs[name]["violations"].extend(check["violations"])
            elif calibration and split:
                original = split["parts"][part_name]
                source = Path(original["bundle_dir"]) / (original["class"] + ".onnx")
                before = json.loads(source.with_suffix(".encodings").read_text())
                calibrated, check = load_calibrated(calibration, source, before)
                actual = json.loads((args.bundle / f"{name}.encodings").read_text())
                if actual != calibrated:
                    raise ValueError(
                        "Non-attention part differs from calibration artifact"
                    )
                passthrough[name] = check
    shared = verify_shared_interfaces(args.bundle, conversion)
    report = {
        "bundle": str(args.bundle.resolve()),
        "config_hash": conversion["config_hash"],
        "graphs": graphs,
        "calibration_passthrough": passthrough,
        "shared_interfaces": shared,
        "passed": bool(graphs)
        and not shared["violations"]
        and not any(g["violations"] for g in graphs.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
