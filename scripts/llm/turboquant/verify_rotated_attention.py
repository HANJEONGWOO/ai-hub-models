# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Audit effective-scale I/O and rotated tiled attention in compiled DLC graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper
from verify_kv_boundary import INT8_TYPES, WIDE_TYPES, parse_dlcinfo, walk_back

from qai_hub_models.models.templates.llm.turboquant.config import Rotation, get_profile
from qai_hub_models.models.templates.llm.turboquant.reference import make_rotation


def rotation_only_change(left: onnx.GraphProto, right: onnx.GraphProto) -> bool:
    """Compare all source graph structure/weights except audited rotation constants."""
    normalized = []
    for original in (left, right):
        graph = onnx.GraphProto()
        graph.CopyFrom(original)
        keep = [t for t in graph.initializer if not t.name.startswith("tq_rotation")]
        graph.ClearField("initializer")
        graph.initializer.extend(keep)
        for node in graph.node:
            for i, name in enumerate(node.input):
                if name.startswith("tq_rotation"):
                    node.input[i] = name.replace("_fwht_", "_rotation_").replace(
                        "_dense_qr_", "_rotation_"
                    )
        normalized.append(graph.SerializeToString())
    return normalized[0] == normalized[1]


def verify_rotations(graph: onnx.GraphProto, config: dict[str, Any]) -> list[str]:
    """Check source constants against the recorded rotation, seed and matrix hash."""
    errors = []
    actual = {
        t.name: numpy_helper.to_array(t)
        for t in graph.initializer
        if t.name.startswith("tq_rotation")
    }
    expected = set()
    rotation = Rotation(config["rotation"])
    d = config["block_size"]
    for kind in ("key", "value"):
        spec = config[kind]
        matrix = make_rotation(rotation, spec["seed"], d).matrix().astype("<f4")
        if rotation == Rotation.DENSE_QR and hashlib.sha256(
            matrix.tobytes()
        ).hexdigest() != spec.get("rotation_f32_sha256"):
            errors.append(f"Dense matrix digest mismatch: {kind}")
        # K only needs the forward R.T; V additionally needs inverse R.
        for transpose in (
            (True,) if kind == "key" and not config.get("qjl") else (False, True)
        ):
            suffix = "_t" if transpose else ""
            name = f"tq_rotation{suffix}_{rotation.value}_s{spec['seed']}_d{d}"
            expected.add(name)
            value = matrix.T if transpose else matrix
            if name not in actual or not np.array_equal(actual[name], value):
                errors.append(f"Incorrect/missing rotation constant: {name}")
    if set(actual) != expected:
        errors.append("Unexpected rotation constants in source graph.")
    return errors


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
                rf"tq_attn_{number}_tile\d+_head\d+_q\d+_({'score_mse' if layer.get('qjl') else 'score'}|partial)",
                op.name,
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
        if layer.get("qjl"):
            qjl_queries = [
                op
                for op in info.ops
                if re.fullmatch(rf"tq_attn_{number}_head\d+_q\d+_rotated_qjl", op.name)
            ]
            if len(qjl_queries) != layer["query_heads"] or any(
                op.op_type not in ("MatMul", "FullyConnected")
                or [t.dtype for t in op.inputs] != ["Float_16", "Float_16"]
                for op in qjl_queries
            ):
                errors.append(
                    f"Missing/non-FP16 QJL query projections in layer {number}"
                )
            for side, table, tokens in (
                ("input", "input", layer["past_tokens"]),
                ("output", "output", layer["new_tokens"]),
            ):
                tensor = layer["qjl"][f"scale_{side}"]
                row = info.io_tables[table].get(tensor)
                if (
                    row is None
                    or row["dtype"] != "Float_16"
                    or "No encoding" not in row["encoding"]
                ):
                    errors.append(f"Invalid QJL scale I/O: {tensor}: {row}")
                elif [int(d.strip()) for d in row["dims"].strip("[]").split(",")] != [
                    layer["kv_heads"],
                    1,
                    tokens,
                    1,
                ]:
                    errors.append(f"Invalid QJL scale shape: {tensor}")
            corrections = [
                op
                for op in info.ops
                if re.fullmatch(
                    rf"tq_attn_{number}_tile\d+_head\d+_q\d+_score_qjl_correction",
                    op.name,
                )
            ]
            if len(corrections) != layer["query_heads"] * len(layer["tiles"]):
                errors.append(f"Missing QJL score products in layer {number}")
            if any(
                op.op_type != "MatMul"
                or [t.dtype for t in op.inputs] != ["Float_16", "Float_16"]
                for op in corrections
            ):
                errors.append(f"Non-FP16 QJL score products in layer {number}")
            for suffix in ["qjl_base_native_fp16"] + [
                f"tile{t['start']}_qjl_decoded16" for t in layer["tiles"]
            ]:
                op = info.producer.get(f"tq_key_{number}_{suffix}")
                if (
                    op is None
                    or op.op_type != "Decode4"
                    or [t.dtype for t in op.inputs]
                    != ["Uint_8", "Float_16", "Float_16"]
                ):
                    errors.append(
                        f"Missing QJL Native decoder: layer {number}/{suffix}"
                    )
        for tile in layer["tiles"]:
            for kind in ("key", "value"):
                restored = tile[f"{kind}_restored"]
                if layer.get("decoder") == "native_unpack_lut_v1":
                    prefix = f"tq_{kind}_{number}_tile{tile['start']}_"
                    native = info.producer.get(prefix + "native_fp16")
                    if (
                        native is None
                        or native.op_type != "Decode4"
                        or [t.dtype for t in native.inputs]
                        != ["Uint_8", "Float_16", "Float_16"]
                        or [t.dtype for t in native.outputs] != ["Float_16"]
                        or not any(
                            "packageName: TurboQuantNative" in p for p in native.params
                        )
                    ):
                        errors.append(f"Missing/incorrect native decoder: {prefix}")
                    # Converter removes the no-op FP16/FP32 source-model cast.
                    if kind == "value":
                        restored = prefix + "native_fp16"
                if restored not in info.producer:
                    errors.append(f"Missing {kind} tile {number}/{tile['start']}")
        for op in info.ops:
            if layer.get("decoder") == "native_unpack_lut_v1" and re.match(
                rf"tq_(key|value)_{number}_tile\d+_dec_", op.name
            ):
                errors.append(f"Graph decoder remains beside native decoder: {op.name}")
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
    conversion = json.loads((bundle / "convert_report.json").read_text())
    config = conversion["config"]
    current_entries = manifest.get("current_kv_attention", [])
    if conversion.get("quantize_current_kv", False):
        if len(current_entries) != len(manifest["codec_io"]):
            errors.append("Missing current-token quantization manifest.")
        source_ops = {o: n for n in graph.node for o in n.output}
        for entry in current_entries:
            prefix = f"tq_{entry['kind']}_{entry['layer']}_current_"
            # Audit source connectivity as well as the compiled Native boundary.
            for cat_name in entry["attention_concats"]:
                cat = source_ops.get(cat_name)
                if cat is None or not cat.input[1].startswith(prefix + "head"):
                    errors.append(f"Unquantized current KV attention: {cat_name}")
            if entry["decoder"] == "native":
                decoder = info.producer.get(prefix + "native_fp16")
                if (
                    decoder is None
                    or decoder.op_type != "Decode4"
                    or [t.dtype for t in decoder.inputs]
                    != ["Uint_8", "Float_16", "Float_16"]
                    or decoder.inputs[0].name != entry["packed"]
                    or [t.dtype for t in decoder.outputs] != ["Float_16"]
                ):
                    errors.append(f"Missing/incorrect current Native decoder: {prefix}")
                source_decoder = source_ops.get(prefix + "native_fp16")
                scale_cast = source_ops.get(prefix + "scale16")
                if (
                    source_decoder is None
                    or source_decoder.input[0] != entry["packed"]
                    or scale_cast is None
                    or scale_cast.input[0] != entry["scale"]
                ):
                    errors.append(
                        f"Current decoder does not reuse cache outputs: {prefix}"
                    )
            if entry.get("qjl"):
                correction_heads = [
                    node
                    for node in graph.node
                    if re.fullmatch(
                        rf"tq_attn_{entry['layer']}_tile\d+_head\d+_q\d+_score_qjl_cat",
                        node.output[0],
                    )
                ]
                if not correction_heads or any(
                    not n.input[1].startswith(prefix + "qjl_head")
                    for n in correction_heads
                ):
                    errors.append(f"Current QJL correction is missing: {prefix}")
    elif current_entries:
        errors.append("Current-token policy disagrees with graph manifest.")
    current = get_profile(conversion["profile"], Rotation(config["rotation"]))
    if current.config_hash() != conversion["config_hash"]:
        errors.append("Bundle rotation/config hash does not match current constants.")
    errors.extend(verify_rotations(graph, config))
    if config.get("qjl"):
        from qai_hub_models.models.templates.llm.turboquant.qjl import projection
        from qai_hub_models.models.templates.llm.turboquant.reference import (
            load_codebook,
        )

        constants = {
            t.name: numpy_helper.to_array(t)
            for t in graph.initializer
            if t.name.startswith(("tq_qjl_", "tq_native_key3_"))
        }
        expected = {
            "tq_qjl_projection_t": projection(
                128, config["key"]["seed"] + 1000
            ).T.astype(np.float32),
            "tq_qjl_coefficient": np.array(
                np.sqrt(np.pi / 2) / np.sqrt(128), dtype=np.float32
            ),
            "tq_qjl_sign_lut_fp16": np.repeat(
                np.array([-1, 1], dtype=np.float16), 8
            ).reshape(1, 1, 1, 16),
            "tq_native_key3_centroids_fp16": np.tile(load_codebook(3, 128), 2)
            .astype(np.float16)
            .reshape(1, 1, 1, 16),
        }
        for key, value in expected.items():
            if key not in constants or not np.array_equal(constants[key], value):
                errors.append(f"QJL constant mismatch: {key}")
        kinverse = f"tq_rotation_{config['rotation']}_s{config['key']['seed']}_d128"
        inverses = [
            n for n in graph.node if n.op_type == "MatMul" and n.input[1] == kinverse
        ]
        if len(inverses) != len(layers) or any(
            not n.name.endswith("_qjl_mse_key") for n in inverses
        ):
            errors.append(
                "QJL inverse K rotation must occur only once per new-token encoder."
            )
    inverse_name = (
        f"tq_rotation_{config['rotation']}_s{config['value']['seed']}"
        f"_d{config['block_size']}"
    )
    inverses = [
        n for n in graph.node if n.op_type == "MatMul" and n.input[1] == inverse_name
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
        "native_decoder_ops": sum(op.op_type == "Decode4" for op in info.ops),
        "rotation": config["rotation"],
        "quantize_current_kv": conversion.get("quantize_current_kv", False),
        "current_kv_decoders": len(current_entries),
        "violations": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--graphs", nargs="*", default=[])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.expanduser()
    names = args.graphs or sorted(
        p.name.removesuffix(".kv_edits.json") for p in bundle.glob("*.kv_edits.json")
    )
    results = {}
    for name in names:
        try:
            results[name] = verify_graph(bundle, name)
            if args.reference_bundle:
                graphs = [
                    onnx.load(
                        path.expanduser() / f"{name}.onnx", load_external_data=False
                    ).graph
                    for path in (bundle, args.reference_bundle)
                ]
                same = rotation_only_change(*graphs)
                results[name]["rotation_only_source_change"] = same
                if not same:
                    results[name]["violations"].append(
                        "Source graph changed beyond rotation."
                    )
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
