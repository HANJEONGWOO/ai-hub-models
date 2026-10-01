# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Activation-only calibration of the existing FP16 KV/attention graph.

This module deliberately does not call the LLM INT8 KV quantization recipe.
Weights/parameter encodings and the set/policy of activation boundaries are
immutable. AIMET is imported lazily; codec/export users do not need AIMET.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def parameter_interfaces(sources: list[Path]) -> dict[str, dict]:
    """Gather outputs inherit a frozen table's grid, not an independent observer.

    In the split Qwen checkpoint the embedding part has no activation encoding:
    its output uses the embedding weight encoding. Its consumer must retain that
    same grid because the runner shares raw buffers without requantization.
    """
    fixed = {}
    for source in sources:
        model = onnx.load(source, load_external_data=False)
        enc = json.loads(source.with_suffix(".encodings").read_text())
        params = {e["name"]: e for e in enc["param_encodings"]}
        outputs = {v.name for v in model.graph.output}
        for node in model.graph.node:
            if (
                node.op_type == "Gather"
                and node.output[0] in outputs
                and node.input[0] in params
            ):
                grid = {**params[node.input[0]], "name": node.output[0]}
                if grid["enc_type"] != "PER_TENSOR":
                    raise ValueError("Unsupported parameter-bound interface grid")
                if grid["name"] in fixed and fixed[grid["name"]] != grid:
                    raise ValueError("Conflicting parameter-bound interface grids")
                fixed[grid["name"]] = grid
    return fixed


def preserve_parameter_interfaces(
    before: dict, after: dict, fixed: dict[str, dict]
) -> tuple[dict, list[str]]:
    """Constrain deployment grids without changing observed statistics/weights."""
    result = copy.deepcopy(after)
    old = {e["name"]: e for e in before["activation_encodings"]}
    kept = []
    for i, enc in enumerate(result["activation_encodings"]):
        name = enc["name"]
        if name in fixed:
            if old[name] != fixed[name]:
                raise ValueError(f"Original parameter/interface grid mismatch: {name}")
            result["activation_encodings"][i] = copy.deepcopy(fixed[name])
            kept.append(name)
    validate_encodings(before, result)
    return result, kept


def validate_encodings(before: dict, after: dict) -> dict:
    """Reject weight, boundary, dtype, bitwidth or symmetry policy changes."""
    if before["param_encodings"] != after["param_encodings"]:
        raise ValueError("Calibration changed parameter encodings")
    old = {e["name"]: e for e in before["activation_encodings"]}
    new = {e["name"]: e for e in after["activation_encodings"]}
    if len(new) != len(after["activation_encodings"]) or old.keys() != new.keys():
        raise ValueError("Calibration changed activation boundary names")
    changed = []

    def policy(e: dict) -> dict:
        return {k: v for k, v in e.items() if k not in ("scale", "offset")}

    for name, enc in new.items():
        if policy(enc) != policy(old[name]):
            raise ValueError(f"Calibration changed activation policy: {name}")
        if enc["dtype"] != "INT" or enc["enc_type"] != "PER_TENSOR":
            raise ValueError(f"Unsupported activation encoding: {name}")
        if len(enc["scale"]) != 1 or len(enc["offset"]) != 1:
            raise ValueError(f"Expected per-tensor encoding: {name}")
        if not np.isfinite(enc["scale"][0]) or enc["scale"][0] <= 0:
            raise ValueError(f"Invalid activation scale: {name}")
        offset = enc["offset"][0]
        if not np.isfinite(offset) or offset != int(offset):
            raise ValueError(f"Invalid activation offset: {name}")
        if not -(2 ** enc["bw"] - 1) <= offset <= 0:
            raise ValueError(f"Out-of-range activation offset: {name}")
        if enc != old[name]:
            changed.append(name)
    return {
        "activation_count": len(old),
        "changed_count": len(changed),
        "changed": changed,
    }


def assert_fp16_boundaries(model: onnx.ModelProto, encodings: dict) -> None:
    types = {
        v.name: v.type.tensor_type.elem_type
        for v in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
        )
    }
    required = {e["name"] for e in encodings["activation_encodings"]}
    required.update(
        n
        for node in model.graph.node
        if node.op_type == "MatMul" and node.output[0].startswith("fp16_attn_")
        for n in node.input
    )
    if not required <= types.keys():
        # Large real models supply inferred metadata before loading external
        # weights, avoiding protobuf's 2GiB serialization limit here.
        inferred = onnx.shape_inference.infer_shapes(model)
        types.update(
            {v.name: v.type.tensor_type.elem_type for v in inferred.graph.value_info}
        )
    for enc in encodings["activation_encodings"]:
        name = enc["name"]
        if (
            name.startswith(("past_", "fp16_"))
            or types.get(name) == onnx.TensorProto.FLOAT16
        ):
            raise ValueError(f"Affine quantizer on FP16 cache/attention: {name}")
    for node in model.graph.node:
        if (
            node.op_type == "MatMul"
            and node.output[0].startswith("fp16_attn_")
            and [types.get(n) for n in node.input] != [onnx.TensorProto.FLOAT16] * 2
        ):
            raise ValueError(f"Non-FP16 attention inputs: {node.name}")


def create_sim(
    model: onnx.ModelProto,
    encodings: dict,
    dummy: dict,
    path: Path,
    providers: list | None = None,
) -> Any:
    """Construct a simulation without HTP rules silently tying new boundaries.

    Explicit ONNX FP16 casts remain in the model. Only the original integer
    activation boundaries are observed; all other AIMET quantizers are disabled.
    """
    from aimet_onnx import QuantizationSimModel, quantsim
    from aimet_onnx.quantsim import load_encodings_to_sim

    assert_fp16_boundaries(model, encodings)
    validate_encodings(encodings, encodings)
    old_fusions = quantsim._fuse_supergroups
    try:
        quantsim._fuse_supergroups = False
        with quantsim._apply_constraints(False):
            sim = QuantizationSimModel(
                model,
                param_type="int4",
                activation_type="int16",
                quant_scheme="min_max",
                config_file="htp_v73",
                dummy_input=dummy,
                providers=providers
                or [
                    ("CUDAExecutionProvider", {"use_tf32": "0"}),
                    "CPUExecutionProvider",
                ],
                path=str(path),
            )
    finally:
        quantsim._fuse_supergroups = old_fusions
    load_encodings_to_sim(sim, encodings, strict=False, allow_overwrite=True)
    parameters = {e["name"] for e in encodings["param_encodings"]}
    activations = {e["name"] for e in encodings["activation_encodings"]}
    for name, quantizer in sim.qc_quantize_op_dict.items():
        if name in parameters:
            quantizer.freeze_encodings()
        elif name not in activations:
            quantizer.enabled = False
    enabled_acts = {
        n
        for n, q in sim.qc_quantize_op_dict.items()
        if q.enabled and n not in parameters
    }
    if enabled_acts != activations:
        raise ValueError("QuantSim activation boundary mismatch")
    return sim


def start_observers(sim: Any, encodings: dict) -> None:
    from aimet_onnx.qc_quantize_op import OpMode

    for enc in encodings["activation_encodings"]:
        q = sim.qc_quantize_op_dict[enc["name"]]
        if q.is_encoding_frozen():
            raise ValueError("Activation quantizer unexpectedly frozen")
        q.reset_encoding_stats()
        q.op_mode = OpMode.updateStats
    for enc in encodings["param_encodings"]:
        q = sim.qc_quantize_op_dict[enc["name"]]
        if not q.is_encoding_frozen():
            raise ValueError("Unfrozen parameter quantizer")
        q.op_mode = OpMode.quantizeDequantize


def finish_observers(sim: Any, encodings: dict) -> tuple[dict, dict]:
    """Only compute activation ranges; never adjust weight scales or biases."""
    from aimet_onnx.qc_quantize_op import OpMode

    result = copy.deepcopy(encodings)
    for enc in result["activation_encodings"]:
        q = sim.qc_quantize_op_dict[enc["name"]]
        q.compute_encodings()
        q.op_mode = OpMode.quantizeDequantize
        observed = q.get_encodings()
        if not observed or len(observed) != 1:
            raise ValueError(f"Missing/non-per-tensor statistics: {enc['name']}")
        enc["scale"] = [float(observed[0].delta)]
        enc["offset"] = [float(observed[0].offset)]
    return result, validate_encodings(encodings, result)


def range_session(sim: Any, encodings: dict, path: Path) -> Any:
    """GPU-native min/max collection; equivalent to pass-through minmax observers.

    Fold ONLY the simulation's frozen parameter QDQ (not source/export weights).
    The CPU receives one small vector per step instead of a synchronization for
    each of thousands of activation observers. AIMET still computes the grids.
    """
    import onnxruntime as ort
    from aimet_onnx.utils import (
        OrtInferenceSession,
        create_ort_session_options_with_aimet_custom_ops,
    )

    sim.fold_param_quantizers()
    model = copy.deepcopy(sim.model.model)
    # Keep disabled opaque FP16 product/input barriers. ORT's mandatory CPU
    # FP16 promotion otherwise elides half rounding even with optimizations off.
    parameters = {e["name"] for e in encodings["param_encodings"]}
    keep = {
        n.name
        for n in model.graph.node
        if n.op_type == "QcQuantizeOp"
        and (
            n.input[0] in parameters
            or (
                n.input[0].startswith("fp16_attn_")
                and n.input[0].endswith(("_half", "_lhs"))
            )
        )
    }
    sim._remove_quantizers(
        model,
        {
            n.name
            for n in model.graph.node
            if n.op_type == "QcQuantizeOp" and n.name not in keep
        },
    )
    for node in model.graph.node:
        if (
            node.op_type == "QcQuantizeOp"
            and sim.qc_quantize_op_dict[node.input[0]].enabled
            and node.input[0] not in parameters
        ):
            raise ValueError("Observer graph contains an enabled quantizer")
    shape = "calibration_range_vector_shape"
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name=shape)
    )
    scalars = []
    for i, enc in enumerate(encodings["activation_encodings"]):
        for op in ("ReduceMin", "ReduceMax"):
            stem = f"calibration_range_{i}_{op}"
            model.graph.node.append(
                onnx.helper.make_node(op, [enc["name"]], [stem], name=stem, keepdims=0)
            )
            model.graph.node.append(
                onnx.helper.make_node(
                    "Reshape", [stem, shape], [stem + "_flat"], name=stem + "_flat"
                )
            )
            scalars.append(stem + "_flat")
    if scalars:
        model.graph.node.append(
            onnx.helper.make_node(
                "Concat",
                scalars,
                ["calibration_ranges"],
                axis=0,
                name="calibration_ranges",
            )
        )
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                "calibration_ranges", onnx.TensorProto.FLOAT, [len(scalars)]
            )
        )
    path.mkdir(exist_ok=False)
    options = create_ort_session_options_with_aimet_custom_ops()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 4
    return OrtInferenceSession(
        model, sim.providers, session_options=options, path=str(path)
    )


def load_calibrated(root: Path, source: Path, baseline: dict) -> tuple[dict, dict]:
    manifest = json.loads((root / "calibration_manifest.json").read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("profile") != "baseline_fp16_kv_fp16_attn_calibrated"
    ):
        raise ValueError("Incomplete or wrong calibration profile")
    entry = manifest["parts"][source.stem]
    if entry["source_onnx_sha256"] != digest(source) or entry[
        "source_encodings_sha256"
    ] != digest(source.with_suffix(".encodings")):
        raise ValueError("Calibration source checkpoint mismatch")
    for name, sha in entry["source_data_sha256"].items():
        if digest(source.parent / name) != sha:
            raise ValueError(f"Calibration source weights changed: {name}")
    path = root / entry["encodings_file"]
    if digest(path) != entry["encodings_sha256"]:
        raise ValueError("Calibrated encoding digest mismatch")
    calibrated = json.loads(path.read_text())
    report = validate_encodings(baseline, calibrated)
    report["manifest_sha256"] = digest(root / "calibration_manifest.json")
    return calibrated, report
