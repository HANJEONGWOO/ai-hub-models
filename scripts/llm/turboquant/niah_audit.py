# SPDX-License-Identifier: BSD-3-Clause
"""Read-only provenance checks and explicit-buffer estimates for NIAH."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def memory_estimate(capacity: int, compressed: bool) -> dict[str, Any]:
    """Exact extrapolation of the historical single-capacity AR1+AR128 I/O.

    This is not peak device memory: QNN/model/HTP buffers are excluded. The
    current runner allocates separate graph buffers for AR1 and AR128, in
    addition to its canonical host KV store. Adding buckets increases I/O.
    """
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 128:
        raise ValueError("Expected an integer context capacity greater than AR128")
    kv = capacity * (29_568 if compressed else 114_688)
    io = 2 * kv + 40_818_180 + 258 * capacity
    return {
        "context_capacity": capacity,
        "compressed": compressed,
        "host_kv_bytes": kv,
        "runtime_io_bytes": io,
        "explicit_host_bytes": kv + io,
        "host_kv_mib": kv / 2**20,
        "runtime_io_mib": io / 2**20,
        "explicit_host_mib": (kv + io) / 2**20,
        "past_tiles_per_layer_ar1": (capacity - 1 + 255) // 256,
        "past_tiles_per_layer_ar128": (capacity - 128 + 255) // 256,
        "scope": "Single capacity, AR1 and AR128; excludes model/QNN/HTP/internal memory; compiled metadata must confirm.",
    }


def audit_source_graph(original: Path, converted: Path) -> dict[str, Any]:
    """Verify unchanged learned tensors and unchanged/regridded shared ranges.

    Removed dead shape constants and newly introduced codec encodings are
    counted, not treated as learned-weight modifications. The separate graph
    audit checks codec/dtype/rotation contracts.
    """
    import onnx

    source = onnx.load(original, load_external_data=False).graph
    target = onnx.load(converted, load_external_data=False).graph
    old = {t.name: t for t in source.initializer}
    new = {t.name: t for t in target.initializer}
    problems: list[str] = []
    shared = old.keys() & new.keys()
    for name in sorted(shared):
        if old[name].SerializeToString() != new[name].SerializeToString():
            problems.append(f"Changed original initializer: {name}")
    external: dict[str, str] = {}
    for tensor in source.initializer:
        if not tensor.external_data:
            continue
        if tensor.name not in new:
            problems.append(f"Missing learned external tensor: {tensor.name}")
        for item in tensor.external_data:
            if item.key == "location":
                if item.value in external:
                    continue
                expected = (original.parent / item.value).resolve(strict=True)
                actual = (converted.parent / item.value).resolve(strict=True)
                if actual != expected:
                    problems.append(f"Different external file: {item.value}")
                external[item.value] = str(actual)
    before = json.loads(original.with_suffix(".encodings").read_text())
    after = json.loads(converted.with_suffix(".encodings").read_text())
    if before["param_encodings"] != after["param_encodings"]:
        problems.append("Changed learned-parameter encodings")
    edits = json.loads(converted.with_suffix(".kv_edits.json").read_text())
    allowed = {
        e["tensor"]: e["after"]
        for e in edits["encoding_edits"]
        if e["action"] == "regrid"
    }
    a = {v["name"]: v for v in before["activation_encodings"]}
    b = {v["name"]: v for v in after["activation_encodings"]}
    changes = 0
    for name in a.keys() & b.keys():
        if a[name] != b[name]:
            changes += 1
            if b[name] != allowed.get(name):
                problems.append(f"Unrecorded activation range change: {name}")
    return {
        "original": str(original),
        "converted": str(converted),
        "parameter_encodings_unchanged": before["param_encodings"]
        == after["param_encodings"],
        "shared_initializer_count": len(shared),
        "pruned_source_initializer_count": len(old.keys() - new.keys()),
        "recorded_shared_activation_regrids": changes,
        "external_weight_files": external,
        "violations": problems,
        "passed": not problems,
    }


def compiled_io_contract(
    contexts: list[dict], capacity: int, config: Any
) -> dict[str, Any]:
    """Inspect compiled QNN tensor metadata and reproduce runner allocations."""
    sets: dict[int, dict[str, dict]] = {1: {}, 128: {}}
    violations: list[str] = []
    for context in contexts:
        for item in context["info"]["graphs"]:
            graph = item["info"]
            match = re.fullmatch(
                r"(?:token|prompt)_ar(1|128)_cl(\d+)_\d+_of_4", graph["graphName"]
            )
            if not match or int(match[2]) != capacity:
                violations.append(
                    f"Wrong compiled graph context/name: {graph['graphName']}"
                )
                continue
            ar = int(match[1])
            for tensor in graph["graphInputs"] + graph["graphOutputs"]:
                value = tensor["info"]
                name = value["name"]
                # Ignore graph-local tensor ID and APP_READ/WRITE direction.
                contract = {
                    k: value[k] for k in ("dimensions", "dataType", "quantizeParams")
                }
                previous = sets[ar].get(name)
                if previous is not None and previous != contract:
                    violations.append(
                        f"Shared compiled tensor disagrees: AR{ar}/{name}"
                    )
                sets[ar][name] = contract
    compressed = bool(config.enabled)
    stream_names: set[str] = set()
    actual_kv = 0
    actual_io = 0
    for ar, tensors in sets.items():
        expected_streams = set()
        for layer in range(28):
            for kind in ("key", "value"):
                if compressed:
                    spec = config.codec_for(kind, layer)
                    fields = (
                        ("packed", "QNN_DATATYPE_UINT_8", 16 * spec.bits),
                        ("scale", "QNN_DATATYPE_FLOAT_16", 1),
                    )
                    for field, dtype, width in fields:
                        base = f"tq_{kind}_{layer}_{field}"
                        expected_streams.add(base)
                        for side, count in (("in", capacity - ar), ("out", ar)):
                            name = f"{base}_{side}"
                            value = tensors.get(name)
                            if (
                                not value
                                or value["dimensions"] != [8, 1, count, width]
                                or value["dataType"] != dtype
                                or value["quantizeParams"].get("definition")
                                != "QNN_DEFINITION_UNDEFINED"
                            ):
                                violations.append(
                                    f"Incorrect compiled KV contract: AR{ar}/{name}"
                                )
                else:
                    base = f"past_{kind}_{layer}"
                    expected_streams.add(base)
                    for side, count in (("in", capacity - ar), ("out", ar)):
                        name = f"{base}_{side}"
                        shape = (
                            [8, 1, 128, count] if kind == "key" else [8, 1, count, 128]
                        )
                        value = tensors.get(name)
                        if (
                            not value
                            or value["dimensions"] != shape
                            or value["dataType"] != "QNN_DATATYPE_FLOAT_16"
                            or value["quantizeParams"].get("definition")
                            != "QNN_DEFINITION_UNDEFINED"
                        ):
                            violations.append(
                                f"Incorrect compiled KV contract: AR{ar}/{name}"
                            )
        observed = set()
        for name, tensor in tensors.items():
            dtype = re.search(r"_(\d+)$", tensor["dataType"])
            if not dtype:
                violations.append(f"Unknown tensor byte width: {tensor['dataType']}")
                continue
            size = math.prod(tensor["dimensions"]) * int(dtype[1]) // 8
            actual_io += size
            if name.endswith("_in") and name.startswith(("past_", "tq_")):
                base = name.removesuffix("_in")
                observed.add(base)
                if base not in stream_names:
                    actual_kv += size // (capacity - ar) * capacity
                    stream_names.add(base)
        if observed != expected_streams:
            violations.append(f"Compiled KV stream set differs: AR{ar}")
        for name, shape in {
            "input_ids": [1, ar],
            "attention_mask": [1, 1, ar, capacity],
            "position_ids_cos": [1, 1, ar, 64],
            "position_ids_sin": [1, 1, ar, 64],
            "logits": [1, ar, 151936],
        }.items():
            if tensors.get(name, {}).get("dimensions") != shape:
                violations.append(
                    f"Incorrect compiled input/output shape: AR{ar}/{name}"
                )
    estimate = memory_estimate(capacity, compressed)
    if actual_kv != estimate["host_kv_bytes"]:
        violations.append("Compiled host KV bytes differ from frozen budget")
    if actual_io != estimate["runtime_io_bytes"]:
        violations.append(
            "Compiled runtime I/O differs from the historical-layout estimate"
        )
    return {
        "host_kv_bytes": actual_kv,
        "runtime_io_bytes": actual_io,
        "expected": estimate,
        "stream_count": len(stream_names),
        "violations": violations,
        "passed": not violations,
    }


def frozen_codec_constants(reference: Path, converted: Path) -> dict[str, Any]:
    """Compare fixed codebook/index/norm/rotation constants with CL1024 source."""
    import hashlib

    import onnx

    prefixes = (
        "tq_centroid",
        "tq_scalar_",
        "tq_native_",
        "tq_structured_k8",
        "tq_rotation",
        "tq_pair_threshold",
        "tq_prescale",
    )
    records = []
    for path in (reference, converted):
        graph = onnx.load(path, load_external_data=False).graph
        records.append(
            {
                t.name: hashlib.sha256(t.SerializeToString()).hexdigest()
                for t in graph.initializer
                if t.name.startswith(prefixes)
            }
        )
    return {
        "constant_count": len(records[1]),
        "reference": str(reference),
        "passed": records[0] == records[1],
        "constant_sha256": records[1],
    }
