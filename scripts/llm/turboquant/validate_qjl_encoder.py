# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Isolated HTP K3+1 encoder validation; not an LLM performance measurement."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
from convert_parts import build_context
from htp_codec_validation import (
    DEFAULT_ADB,
    DEFAULT_NDK,
    DEFAULT_QNN_PYTHON,
    DEFAULT_SDK,
    encode_cases,
    qairt_env,
    run_logged,
)
from validate_native_decoder import run as run_device

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.numerics import (
    HTP_FP16,
    compare_encode,
)
from qai_hub_models.models.templates.llm.turboquant.packing import (
    pack_indices,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.qjl import (
    QJLKeyReference,
    build_qjl_encode_model,
    dequantize_residual,
    projection,
    quantize_residual,
)


def reference_check(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("external_qjl_reference", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load reference {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.QJL(128, seed=1042)
    residual = np.random.default_rng(711).normal(size=(1024, 128))
    residual[0] = 0
    signs, norms = quantize_residual(residual)
    expected_signs, expected_norms = original.quantize(residual)
    expected = original.dequantize(expected_signs, expected_norms)
    actual = dequantize_residual(signs, norms)
    np.testing.assert_array_equal(projection(128, 1042), original.S)
    np.testing.assert_array_equal(signs, expected_signs)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "projection_equal": True,
        "signs_equal": True,
        "max_reconstruction_error": float(np.max(np.abs(actual - expected))),
    }


def build(args: argparse.Namespace) -> None:
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    config = get_profile("k3qjl_v4_scaled")
    env = qairt_env(args.sdk, args.qnn_python)
    manifest = {
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "reference": reference_check(args.reference),
        "graphs": {},
    }
    args.native_decoder_package = args.package
    with np.load(args.snapshot) as snapshot:
        for tokens in args.tokens:
            name = f"native_t{tokens}"
            model = build_qjl_encode_model(config, tokens)
            onnx.save(model, work / f"{name}.onnx")
            cases = encode_cases(snapshot, "key", tokens, range_scale=0.5)
            lines = []
            for case, x in cases.items():
                x = x.reshape(8, 1, tokens, 128).astype(np.float16)
                raw = work / f"{name}_{case}.raw"
                x.tofile(raw)
                np.save(work / f"{name}_{case}.npy", x.astype(np.float64))
                lines.append(
                    f"tq_key_0_present_tokens_last:={args.device_dir}/{raw.name}"
                )
            (work / f"{name}_inputs.txt").write_text("\n".join(lines) + "\n")
            sdkbin = args.sdk / "bin/x86_64-linux-clang"
            run_logged(
                [
                    str(args.qnn_python),
                    str(sdkbin / "qairt-converter"),
                    "--input_network",
                    str(work / f"{name}.onnx"),
                    "--op_package_config",
                    str(Path(__file__).with_name("native_decoder") / "Decode4.xml"),
                    "--float_bitwidth",
                    "16",
                    "--output_path",
                    str(work / f"{name}.dlc"),
                ],
                work / f"{name}.convert.log",
                env,
            )
            run_logged(
                [
                    str(args.qnn_python),
                    str(sdkbin / "qairt-dlc-info"),
                    "-i",
                    str(work / f"{name}.dlc"),
                ],
                work / f"{name}.dlcinfo.txt",
                env,
            )
            build_context(args, [name], name, work, env)
            manifest["graphs"][name] = {"tokens": tokens, "cases": list(cases)}
            print(f"Built QJL encoder T={tokens}", flush=True)
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def compare(args: argparse.Namespace) -> None:
    work = args.work_dir
    manifest = json.loads((work / "manifest.json").read_text())
    codec = QJLKeyReference(get_profile("k3qjl_v4_scaled"))
    reports = {}
    for name, graph in manifest["graphs"].items():
        cases = {}
        for index, case in enumerate(graph["cases"]):
            x = np.load(work / f"{name}_{case}.npy")
            result = work / "device" / f"out_{name}" / f"Result_{index}"

            def output(
                field: str,
                dtype: type,
                width: int,
                result: Path = result,
                tokens: int = graph["tokens"],
            ) -> np.ndarray:
                paths = list(result.glob(f"tq_key_0_{field}_out*.raw"))
                if len(paths) != 1:
                    raise ValueError(f"Missing/ambiguous {field} output: {paths}")
                return np.fromfile(paths[0], dtype=dtype).reshape(8, 1, tokens, width)

            packed = output("packed", np.uint8, 64)
            scales = output("scale", np.float16, 1)
            qscales = output("qjlscale", np.float16, 1)
            codes = unpack_indices(packed, 4, 128)
            indices = codes & 7
            base_check = compare_encode(
                codec.base, x, pack_indices(indices, 3), scales, HTP_FP16
            )
            # Condition on actually stored MSE codes/scale to isolate QJL errors.
            base = (codec.base.centroids[indices].astype(np.float16) * scales).astype(
                np.float16
            )
            residual = x - codec.base.rotation.inverse(base)
            signs, norms = quantize_residual(residual)
            projected = residual @ projection(128, 1042).T
            mismatches = np.where(codes >= 8, 1, -1) != signs
            margin = np.abs(projected) / np.maximum(norms, 1e-30)
            unexplained = mismatches & (margin > HTP_FP16.boundary_distance)
            expected_scale = norms * codec.coefficient
            relative = np.abs(qscales - expected_scale) / np.maximum(
                expected_scale, 1e-30
            )
            qjl_check = {
                "sign_mismatches": int(mismatches.sum()),
                "unexplained_sign_mismatches": int(unexplained.sum()),
                "max_sign_mismatch_normalized_margin": float(margin[mismatches].max())
                if mismatches.any()
                else 0,
                "scale_max_relative_error": float(relative.max()),
                "finite": bool(
                    np.isfinite(scales).all() and np.isfinite(qscales).all()
                ),
                "zero_residual_scale_exact": bool(np.all(qscales[norms == 0] == 0)),
            }
            qjl_check["passed"] = (
                qjl_check["finite"]
                and not unexplained.any()
                and relative.max() <= HTP_FP16.norm_rel
                and qjl_check["zero_residual_scale_exact"]
            )
            qjl_check["passed"] = bool(qjl_check["passed"])
            cases[case] = {
                "mse": base_check,
                "qjl": qjl_check,
                "passed": base_check["passed"] and qjl_check["passed"],
            }
        reports[name] = cases
    report = {
        "reference": manifest["reference"],
        "config_hash": manifest["config_hash"],
        "tolerances": {
            "scale_relative": HTP_FP16.norm_rel,
            "normalized_sign_boundary": HTP_FP16.boundary_distance,
        },
        "graphs": reports,
        "passed": all(c["passed"] for g in reports.values() for c in g.values()),
    }
    (work / "correctness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError(
            "QJL/MSE encoder tolerance gate failed; see correctness.json"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "run", "compare", "all"])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 128])
    parser.add_argument(
        "--device-dir", default="/data/local/tmp/qaihm_turboquant/qjl_encoder_20260919"
    )
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--qnn-python", type=Path, default=DEFAULT_QNN_PYTHON)
    parser.add_argument("--ndk", type=Path, default=DEFAULT_NDK)
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    args = parser.parse_args()
    for stage, fn in (("build", build), ("run", run_device), ("compare", compare)):
        if args.stage in (stage, "all"):
            fn(args)


if __name__ == "__main__":
    main()
