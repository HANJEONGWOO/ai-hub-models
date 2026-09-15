# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Split an LLM's AIMET checkpoint into per-part ONNX bundles locally.

Reuses the exact Part/split code the AI Hub export pipeline runs before upload
(``DynamicSplitPartBase.serialize_graph``), but stops there: no Workbench jobs.
Each part lands in ``<out>/<PartClass>_<precision>.aimet/`` with ``.onnx``,
``.data`` and ``.encodings``, plus ``split_manifest.json`` describing I/O.

Usage:

    HF_HUB_OFFLINE=1 PYTHONPATH=src python scripts/llm/turboquant/split_checkpoint.py \
        --model-id qwen3_1_7b --out ~/.qaihm/tmp/turboquant/qwen3_1_7b_w4a16_split
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import onnx


def tensor_info(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    dims = [d.dim_param if d.dim_param else d.dim_value for d in tensor_type.shape.dim]
    return {
        "name": value.name,
        "dtype": onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type).name,
        "shape": dims,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="qwen3_1_7b")
    parser.add_argument("--checkpoint", default="DEFAULT")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out: Path = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    module = importlib.import_module(f"qai_hub_models.models.{args.model_id}")
    collection_cls = module.Model
    manifest: dict[str, Any] = {
        "model_id": args.model_id,
        "checkpoint": args.checkpoint,
        "parts": {},
    }
    for part_name, part_cls in collection_cls.parts.items():
        part = part_cls.from_pretrained(checkpoint=args.checkpoint)
        bundle_dir = part.serialize_graph(part.graph_names[0], out)
        onnx_path = bundle_dir / f"{part_cls.__name__}.onnx"
        graph = onnx.load(str(onnx_path), load_external_data=False).graph
        encodings = bundle_dir / f"{part_cls.__name__}.encodings"
        manifest["parts"][part_name] = {
            "class": part_cls.__name__,
            "bundle_dir": str(bundle_dir),
            "graph_names": part.graph_names,
            "inputs": [tensor_info(v) for v in graph.input],
            "outputs": [tensor_info(v) for v in graph.output],
            "num_nodes": len(graph.node),
            "encodings_sha256": hashlib.sha256(encodings.read_bytes()).hexdigest()
            if encodings.exists()
            else None,
        }
        print(f"{part_name}: {bundle_dir}", flush=True)

    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
