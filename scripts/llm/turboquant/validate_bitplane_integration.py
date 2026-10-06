# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""HTP prepare/run a tiny quantized GQA graph including current/past packed K.

Functional validation, not an additional throughput measurement. Includes the
real score-grid boundary absent from standalone float QK microbenchmarks.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import numpy as np
import onnx
from htp_codec_validation import (
    DEFAULT_ADB,
    DEFAULT_NDK,
    DEFAULT_QNN_PYTHON,
    DEFAULT_SDK,
    DEVICE_LIBS,
    NDK_LIBCXX,
    adb,
    adb_shell_rc,
    qairt_env,
    run_logged,
    windows_path,
)
from onnx.reference import ReferenceEvaluator

from qai_hub_models.models.templates.llm.turboquant.bitplane_attention import (
    use_bitplane_qk,
)
from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.current_attention import (
    quantize_current_attention,
)
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)
from qai_hub_models.models.templates.llm.turboquant.native_decoder import (
    use_native_decoder,
    with_reference_decoder,
)
from qai_hub_models.models.templates.llm.turboquant.packing import pack_indices
from qai_hub_models.models.templates.llm.turboquant.reference import PolarQuantReference
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)
from qai_hub_models.test.test_models import test_turboquant_tiled_attention as fixture
from qai_hub_models.test.test_models.test_turboquant_structured import BitplaneQK4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--device-dir", required=True)
    parser.add_argument(
        "--compare-with",
        type=Path,
        help="Compare two completed functional runs without executing again",
    )
    parser.add_argument(
        "--profile",
        choices=["k4s_v4_scaled", "k4s_v4_bitplane"],
        default="k4s_v4_bitplane",
    )
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--qnn-python", type=Path, default=DEFAULT_QNN_PYTHON)
    parser.add_argument("--ndk", type=Path, default=DEFAULT_NDK)
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    args = parser.parse_args()
    work, remote = args.work_dir, args.device_dir
    if args.compare_with:
        report = {}
        for path in sorted((work / "outputs/Result_0").glob("*.raw")):
            a = np.fromfile(path, dtype=np.float32).astype(np.float64)
            b = np.fromfile(
                args.compare_with / "outputs/Result_0" / path.name, dtype=np.float32
            ).astype(np.float64)
            delta = a - b
            report[path.stem] = {
                "bit_exact": bool(np.array_equal(a, b)),
                "max_abs": float(np.abs(delta).max()),
                "rmse": float(np.sqrt(np.mean(delta**2))),
                "relative_l2": float(
                    np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-30)
                ),
            }
        with (work / "pair_comparison.json").open("x") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2))
        if not report or any(
            not item["bit_exact"]
            for name, item in report.items()
            if name.startswith("tq_")
        ):
            raise ValueError("LUT and Bit-plane paths wrote different cache values")
        return
    work.mkdir(parents=True, exist_ok=True)
    with (work / "attempt").open("x") as stream:
        stream.write("functional integration, no performance result\n")
    cfg = get_profile(args.profile)
    model, enc = fixture.attention_part(3)
    result = quantize_current_attention(
        use_native_decoder(
            tile_kv_attention(
                apply_kv_profile(model, enc, cfg, 3, 35), cfg, 7, rotated=True
            ),
            cfg,
        ),
        cfg,
    )
    if cfg.bitplane_qk:
        result = use_bitplane_qk(result, cfg)
    name = "integration"
    onnx.save(result.model, work / f"{name}.onnx")
    (work / "encodings.json").write_text(json.dumps(result.encodings))
    feeds = fixture.feeds(3, 13)
    rng = np.random.default_rng(433)
    for kind in ("key", "value"):
        feeds.pop(f"tq_{kind}_0_norm_in")
        codec = PolarQuantReference(getattr(cfg, kind), precomputed_norm=True)
        vectors = rng.normal(size=(2, 1, 32, 128)) * 0.2
        vectors[..., :2, :] = 0
        indices, scale = codec.encode(vectors)
        feeds[f"tq_{kind}_0_packed_in"] = pack_indices(indices, 4)
        feeds[f"tq_{kind}_0_scale_in"] = scale.astype(np.float16).astype(np.float32)
        feeds[f"new_{kind}"] *= 0.2
    expected = ReferenceEvaluator(
        with_reference_decoder(result.model), new_ops=[BitplaneQK4]
    ).run(None, feeds)
    for key, values in feeds.items():
        # qnn-net-run performs declared input-grid conversions from float files.
        values.astype(np.float32).tofile(work / f"{key}.raw")
    (work / "inputs.txt").write_text(
        " ".join(f"{key}:={remote}/{key}.raw" for key in feeds) + "\n"
    )
    np.savez(
        work / "expected.npz",
        **{o.name: x for o, x in zip(result.model.graph.output, expected, strict=True)},
    )
    env = qairt_env(args.sdk, args.qnn_python)
    sdkbin = args.sdk / "bin/x86_64-linux-clang"
    run_logged(
        [
            str(args.qnn_python),
            str(sdkbin / "qairt-converter"),
            "--input_network",
            str(work / f"{name}.onnx"),
            "--quantization_overrides",
            str(work / "encodings.json"),
            "--op_package_config",
            str(Path(__file__).with_name("native_decoder") / "Decode4.xml"),
            "--output_path",
            str(work / "float.dlc"),
        ],
        work / "convert.log",
        env,
    )
    run_logged(
        [
            str(args.qnn_python),
            str(sdkbin / "qairt-quantizer"),
            "--input_dlc",
            str(work / "float.dlc"),
            "--output_dlc",
            str(work / f"{name}.dlc"),
            "--enable_float_fallback",
            "--float_bitwidth",
            "16",
            "--act_bitwidth",
            "16",
            "--bias_bitwidth",
            "32",
        ],
        work / "quantize.log",
        env,
    )
    run_logged(
        [
            str(args.qnn_python),
            str(sdkbin / "qairt-dlc-info"),
            "-i",
            str(work / f"{name}.dlc"),
        ],
        work / "dlcinfo.txt",
        env,
    )
    htp = work / "htp.json"
    htp.write_text(
        json.dumps(
            {
                "graphs": [{"graph_names": ["float"], "O": 3}],
                "devices": [
                    {"soc_model": 87, "dsp_arch": "v81", "pd_session": "unsigned"}
                ],
            }
        )
    )
    be = work / "backend.json"
    be.write_text(
        json.dumps(
            {
                "backend_extensions": {
                    "shared_library_path": str(
                        args.sdk / "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so"
                    ),
                    "config_file_path": str(htp),
                }
            }
        )
    )
    run_logged(
        [
            str(sdkbin / "qnn-context-binary-generator"),
            "--backend",
            str(args.sdk / "lib/x86_64-linux-clang/libQnnHtp.so"),
            "--model",
            str(args.sdk / "lib/x86_64-linux-clang/libQnnModelDlc.so"),
            "--dlc_path",
            str(work / f"{name}.dlc"),
            "--op_packages",
            f"{args.package / 'x86_64-linux-clang/libTurboQuantNative.so'}:TurboQuantInterfaceProvider",
            "--binary_file",
            name,
            "--config_file",
            str(be),
            "--output_dir",
            str(work),
        ],
        work / "prepare.log",
        env,
    )
    if adb(args, "shell", "getprop", "ro.soc.model").strip() != "SM8850":
        raise RuntimeError("Expected SM8850")
    adb(args, "shell", "mkdir", "-p", remote)
    for rel in DEVICE_LIBS:
        adb(args, "push", windows_path(args.sdk / rel), remote + "/")
    for path in (
        args.ndk / NDK_LIBCXX,
        args.package / "hexagon-v81/libTurboQuantNative.so",
        work / f"{name}.bin",
        work / "inputs.txt",
        *work.glob("*.raw"),
    ):
        adb(args, "push", windows_path(path), remote + "/")
    adb(args, "shell", "chmod", "+x", remote + "/qnn-net-run")
    prefix = f"cd {shlex.quote(remote)} && export LD_LIBRARY_PATH={shlex.quote(remote)}:/vendor/lib64 && export ADSP_LIBRARY_PATH={shlex.quote(remote)}:/vendor/dsp/cdsp:/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp"
    rc, log = adb_shell_rc(
        args,
        f"{prefix} && ./qnn-net-run --backend libQnnHtp.so --retrieve_context {name}.bin --op_packages libTurboQuantNative.so:TurboQuantInterfaceProvider --input_list inputs.txt --output_dir outputs",
    )
    (work / "device.log").write_text(log)
    if rc:
        raise RuntimeError(f"HTP integration failed: {rc}")
    adb(args, "pull", remote + "/outputs", windows_path(work))
    report = {}
    for o, ref in zip(result.model.graph.output, expected, strict=True):
        x = np.fromfile(
            work / "outputs/Result_0" / f"{o.name}.raw", dtype=np.float32
        ).reshape(ref.shape)
        diff = x.astype(np.float64) - ref.astype(np.float64)
        report[o.name] = {
            "max_abs": float(np.abs(diff).max()),
            "rmse": float(np.sqrt(np.mean(diff**2))),
        }
    (work / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
