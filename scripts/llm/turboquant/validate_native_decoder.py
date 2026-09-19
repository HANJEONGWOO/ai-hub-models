# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Isolated HTP custom-decoder correctness test; not an LLM performance run."""

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
from onnx import TensorProto, helper, numpy_helper

from qai_hub_models.models.templates.llm.turboquant.constants import CODEBOOK_HEX


def build(args: argparse.Namespace) -> None:
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    sdkbin = args.sdk / "bin/x86_64-linux-clang"
    env = qairt_env(args.sdk, args.qnn_python)
    centroids = np.array(
        [float.fromhex(x) for x in CODEBOOK_HEX[3 if args.lut == "mse3" else 4, 128]],
        dtype=np.float16,
    )
    if args.lut == "mse3":
        centroids = np.tile(centroids, 2)
    elif args.lut == "qjl_sign":
        centroids = np.repeat(np.array([-1, 1], dtype=np.float16), 8)
    (work / "lut.json").write_text(
        json.dumps({"kind": args.lut, "values": centroids.tolist()}) + "\n"
    )
    for tokens in args.tokens:
        name = f"native_t{tokens}"
        packed_shape, scale_shape = [8, 1, tokens, 64], [8, 1, tokens, 1]
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Decode4",
                    ["packed", "scale", "centroids"],
                    ["decoded"],
                    domain="turboquant",
                    name="decode4",
                )
            ],
            name,
            [
                helper.make_tensor_value_info(
                    "packed", TensorProto.UINT8, packed_shape
                ),
                helper.make_tensor_value_info(
                    "scale", TensorProto.FLOAT16, scale_shape
                ),
            ],
            [
                helper.make_tensor_value_info(
                    "decoded", TensorProto.FLOAT16, [8, 1, tokens, 128]
                )
            ],
            [numpy_helper.from_array(centroids.reshape(1, 1, 1, 16), "centroids")],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 17),
                helper.make_opsetid("turboquant", 1),
            ],
            ir_version=9,
        )
        onnx.checker.check_model(model)
        onnx.save(model, work / f"{name}.onnx")
        rng = np.random.default_rng(1809 + tokens)
        packed = rng.integers(0, 256, packed_shape, dtype=np.uint8)
        # Include every byte pattern, both nibbles, and the final partial tile.
        packed.flat[: min(256, packed.size)] = np.arange(
            min(256, packed.size), dtype=np.uint8
        )
        scale = np.exp(rng.uniform(-5, 8, scale_shape)).astype(np.float16)
        scale.flat[:6] = [0, 1, 2**-14, 2**-10, 32, 65504]
        indices = np.stack((packed >> 4, packed & 15), axis=-1).reshape(
            8, 1, tokens, 128
        )
        expected = (
            centroids[indices].astype(np.float32) * scale.astype(np.float32)
        ).astype(np.float16)
        packed.tofile(work / f"{name}_packed.raw")
        scale.tofile(work / f"{name}_scale.raw")
        np.save(work / f"{name}_expected.npy", expected)
        (work / f"{name}_inputs.txt").write_text(
            f"packed:={args.device_dir}/{name}_packed.raw scale:={args.device_dir}/{name}_scale.raw\n"
        )
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
            work / f"{name}.info.log",
            env,
        )
        htp = work / f"{name}_htp.json"
        htp.write_text(
            json.dumps(
                {
                    "graphs": [{"graph_names": [name], "O": 3}],
                    "devices": [
                        {"soc_model": 87, "dsp_arch": "v81", "pd_session": "unsigned"}
                    ],
                }
            )
        )
        backend = work / f"{name}_be.json"
        backend.write_text(
            json.dumps(
                {
                    "backend_extensions": {
                        "shared_library_path": str(
                            args.sdk
                            / "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so"
                        ),
                        "config_file_path": str(htp),
                    }
                }
            )
        )
        package = f"{args.package / 'x86_64-linux-clang/libTurboQuantNative.so'}:TurboQuantInterfaceProvider"
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
                package,
                "--binary_file",
                name,
                "--output_dir",
                str(work),
                "--config_file",
                str(backend),
                "--log_level",
                "verbose",
            ],
            work / f"{name}.ctxgen.log",
            env,
        )
        print(f"Built {name}", flush=True)


def run(args: argparse.Namespace) -> None:
    work, remote = args.work_dir, args.device_dir
    if adb(args, "shell", "getprop", "ro.soc.model").strip() != "SM8850":
        raise RuntimeError("This package targets SM8850/V81 only")
    adb(args, "shell", "mkdir", "-p", remote)
    for rel in DEVICE_LIBS:
        adb(args, "push", windows_path(args.sdk / rel), remote + "/")
    for path in (
        args.ndk / NDK_LIBCXX,
        args.package / "hexagon-v81/libTurboQuantNative.so",
    ):
        adb(args, "push", windows_path(path), remote + "/")
    adb(args, "shell", "chmod", "+x", remote + "/qnn-net-run")
    for path in work.iterdir():
        if path.suffix in (".bin", ".raw", ".txt"):
            adb(args, "push", windows_path(path), remote + "/")
    prefix = f"cd {shlex.quote(remote)} && export LD_LIBRARY_PATH={shlex.quote(remote)}:/vendor/lib64 && export ADSP_LIBRARY_PATH={shlex.quote(remote)}:/vendor/dsp/cdsp:/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp"
    (work / "device").mkdir(exist_ok=True)
    for tokens in args.tokens:
        name = f"native_t{tokens}"
        rc, log = adb_shell_rc(
            args,
            f"{prefix} && ./qnn-net-run --backend libQnnHtp.so --retrieve_context {name}.bin --op_packages libTurboQuantNative.so:TurboQuantInterfaceProvider --input_list {name}_inputs.txt --use_native_input_files --use_native_output_files --output_dir out_{name} --profiling_level detailed --log_level verbose",
        )
        (work / f"{name}.device.log").write_text(log)
        if rc:
            raise RuntimeError(f"{name} failed ({rc}), see device log")
        adb(args, "pull", f"{remote}/out_{name}", windows_path(work / "device"))
        print(f"Executed {name} on HTP", flush=True)


def compare(args: argparse.Namespace) -> None:
    reports = {}
    for tokens in args.tokens:
        name = f"native_t{tokens}"
        result = args.work_dir / "device" / f"out_{name}" / "Result_0"
        paths = list(result.glob("decoded*.raw"))
        if len(paths) != 1:
            raise RuntimeError(f"Missing/ambiguous device output: {paths}")
        expected = np.load(args.work_dir / f"{name}_expected.npy")
        actual = np.fromfile(paths[0], dtype=np.float16).reshape(expected.shape)
        diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        passed = bool(np.allclose(actual, expected, rtol=0.001, atol=2**-24))
        reports[name] = {
            "passed": passed,
            "max_abs_error": float(diff.max()),
            "bit_exact_fraction": float(
                np.mean(actual.view(np.uint16) == expected.view(np.uint16))
            ),
            "finite": bool(np.isfinite(actual).all()),
        }
    (args.work_dir / "correctness.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    print(json.dumps(reports, indent=2))
    if not all(r["passed"] for r in reports.values()):
        raise RuntimeError("Native decoder does not match the FP16 LUT oracle")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "run", "compare", "all"])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[3])
    parser.add_argument("--lut", choices=["mse4", "mse3", "qjl_sign"], default="mse4")
    parser.add_argument(
        "--device-dir", default="/data/local/tmp/qaihm_turboquant/native_smoke_20260918"
    )
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--qnn-python", type=Path, default=DEFAULT_QNN_PYTHON)
    parser.add_argument("--ndk", type=Path, default=DEFAULT_NDK)
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    args = parser.parse_args()
    for stage in ("build", "run", "compare"):
        if args.stage in (stage, "all"):
            globals()[stage](args)


if __name__ == "__main__":
    main()
