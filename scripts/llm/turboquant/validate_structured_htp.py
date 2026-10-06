# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Single-execute HTP encoder/QK diagnostics; CPU reference is validation only.

Every graph executes once, with detailed profiling. These isolated, instrumented
times are not full-model TTFT or summed layer latency. No CPU backend is loaded.
Use a fresh work directory: build/run attempts and failures are preserved.
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
    parse_profile,
    qairt_env,
    read_native,
    run_logged,
    windows_path,
)
from onnx import TensorProto, helper, numpy_helper

from qai_hub_models.models.templates.llm.turboquant.bitplane_attention import (
    bitplane_reference,
)
from qai_hub_models.models.templates.llm.turboquant.config import (
    TurboQuantConfig,
    get_profile,
)
from qai_hub_models.models.templates.llm.turboquant.export import build_encode_model
from qai_hub_models.models.templates.llm.turboquant.numerics import (
    HTP_FP16,
    compare_encode,
)
from qai_hub_models.models.templates.llm.turboquant.packing import pack_indices
from qai_hub_models.models.templates.llm.turboquant.reference import (
    PolarQuantReference,
    load_codebook,
)
from qai_hub_models.models.templates.llm.turboquant.structured import load_parameters

GROUPS = {
    "lm": "k4_v4_scaled",
    "structured_lut": "k4s_v4_scaled",
    "bitplane": "k4s_v4_bitplane",
}


def qk_model(
    name: str, cfg: TurboQuantConfig, seq: int, tokens: int
) -> onnx.ModelProto:
    inputs = [
        helper.make_tensor_value_info(n, t, s)
        for n, t, s in (
            ("packed", TensorProto.UINT8, [8, 1, tokens, 64]),
            ("scale", TensorProto.FLOAT16, [8, 1, tokens, 1]),
            ("query", TensorProto.FLOAT16, [8, 2, seq, 128]),
        )
    ]
    if cfg.bitplane_qk:
        constants = [
            numpy_helper.from_array(
                np.asarray(load_parameters()["beta"], "<f4")
                .view(np.uint8)
                .reshape(1, 1, 4, 4),
                "beta",
            )
        ]
        nodes = [
            helper.make_node(
                "BitplaneQK4",
                ["packed", "scale", "query", "beta"],
                ["score"],
                name="bitplane_qk",
                domain="turboquant",
            )
        ]
    else:
        constants = [
            numpy_helper.from_array(
                load_codebook(4, 128, cfg.key.codebook)
                .astype(np.float16)
                .reshape(1, 1, 1, 16),
                "centroids",
            )
        ]
        nodes = [
            helper.make_node(
                "Decode4",
                ["packed", "scale", "centroids"],
                ["key"],
                domain="turboquant",
                name="native_key",
            ),
            helper.make_node(
                "Transpose", ["key"], ["key_t"], perm=[0, 1, 3, 2], name="key_t"
            ),
            helper.make_node("MatMul", ["query", "key_t"], ["score"], name="qk"),
        ]
    graph = helper.make_graph(
        nodes,
        name,
        inputs,
        [
            helper.make_tensor_value_info(
                "score", TensorProto.FLOAT16, [8, 2, seq, tokens]
            )
        ],
        constants,
    )
    if not cfg.bitplane_qk:
        graph.value_info.append(
            helper.make_tensor_value_info(
                "key", TensorProto.FLOAT16, [8, 1, tokens, 128]
            )
        )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("turboquant", 1),
        ],
        ir_version=8,
    )


def build(args: argparse.Namespace) -> None:
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    with (work / "build.attempt").open("x") as stream:
        stream.write("HTP-only single-execute diagnostic\n")
    env = qairt_env(args.sdk, args.qnn_python)
    sdkbin = args.sdk / "bin/x86_64-linux-clang"
    manifest = {
        "graphs": {},
        "runs_per_graph": 1,
        "profiling": "detailed",
        "configurations": {g: get_profile(p).to_dict() for g, p in GROUPS.items()},
    }
    for group, profile in GROUPS.items():
        cfg = get_profile(profile)
        codec = PolarQuantReference(cfg.key, precomputed_norm=True)
        for seq in args.sequences:
            rng = np.random.default_rng(961 + seq)
            for operation in ("encode", "qk"):
                if group == "bitplane" and operation == "encode":
                    continue  # Same encoder as structured_lut; not a repeat measurement.
                name = f"{group}_{operation}_ar{seq}"
                if operation == "encode":
                    model = build_encode_model(
                        cfg, cfg.key, 8, seq, name, head_major=True
                    )
                    x = rng.normal(size=(8, 1, seq, 128)).astype(np.float16)
                    x[0, 0, 0] = 0
                    inputs = {model.graph.input[0].name: x}
                    np.save(work / f"{name}_x.npy", x)
                    expected = {}
                else:
                    tokens = args.tokens
                    model = qk_model(name, cfg, seq, tokens)
                    # All groups see the same samples; LUT/BP get identical bytes/scales.
                    qrng = np.random.default_rng(707 + seq)
                    vectors = qrng.normal(size=(8, 1, tokens, 128))
                    indices, scale = codec.encode(vectors)
                    scale = scale.astype(np.float16)
                    packed = pack_indices(indices, 4)
                    packed.flat[:256] = np.arange(256, dtype=np.uint8)
                    scale.flat[0] = 0
                    query = (qrng.normal(size=(8, 2, seq, 128)) / np.sqrt(128)).astype(
                        np.float16
                    )
                    inputs = {"packed": packed, "scale": scale, "query": query}
                    z = np.stack((packed >> 4, packed & 15), axis=-1).reshape(
                        8, 1, tokens, 128
                    )
                    lut = (
                        load_codebook(4, 128, cfg.key.codebook)
                        .astype(np.float16)[z]
                        .astype(np.float32)
                        * scale.astype(np.float32)
                    ).astype(np.float16)
                    lut_scores = (
                        query.astype(np.float32)
                        @ lut.astype(np.float32).swapaxes(-1, -2)
                    ).astype(np.float16)
                    expected = {"lut": lut_scores}
                    expected["score"] = (
                        bitplane_reference(
                            packed, scale, query, np.asarray(load_parameters()["beta"])
                        )
                        if cfg.bitplane_qk
                        else lut_scores
                    )
                    np.savez(work / f"{name}_expected.npz", **expected)
                onnx.checker.check_model(model)
                onnx.save(model, work / f"{name}.onnx")
                for tensor, values in inputs.items():
                    values.tofile(work / f"{name}_{tensor}.raw")
                (work / f"{name}_inputs.txt").write_text(
                    " ".join(f"{n}:={args.device_dir}/{name}_{n}.raw" for n in inputs)
                    + "\n"
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
                                {
                                    "soc_model": 87,
                                    "dsp_arch": "v81",
                                    "pd_session": "unsigned",
                                }
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
                        "--output_dir",
                        str(work),
                        "--config_file",
                        str(backend),
                    ],
                    work / f"{name}.ctxgen.log",
                    env,
                )
                manifest["graphs"][name] = {
                    "group": group,
                    "profile": profile,
                    "operation": operation,
                    "seq": seq,
                    "tokens": args.tokens,
                    "outputs": [o.name for o in model.graph.output],
                }
                (work / "manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                print("Built", name, flush=True)


def run(args: argparse.Namespace) -> None:
    work, remote = args.work_dir, args.device_dir
    if adb(args, "shell", "getprop", "ro.soc.model").strip() != "SM8850":
        raise RuntimeError("Expected SM8850/V81")
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
    manifest = json.loads((work / "manifest.json").read_text())
    for name in manifest["graphs"]:
        with (work / f"{name}.attempt").open("x") as stream:
            stream.write("one execute; no retry\n")
        rc, log = adb_shell_rc(
            args,
            f"{prefix} && ./qnn-net-run --backend libQnnHtp.so --retrieve_context {name}.bin --op_packages libTurboQuantNative.so:TurboQuantInterfaceProvider --input_list {name}_inputs.txt --use_native_input_files --use_native_output_files --output_dir out_{name} --profiling_level detailed --log_level info",
        )
        (work / f"{name}.device.log").write_text(log)
        if rc:
            raise RuntimeError(f"HTP failed: {name}, exit={rc}")
        adb(args, "pull", f"{remote}/out_{name}", windows_path(work))
        run_logged(
            [
                str(args.sdk / "bin/x86_64-linux-clang/qnn-profile-viewer"),
                "--reader",
                str(args.sdk / "lib/x86_64-linux-clang/libQnnHtpProfilingReader.so"),
                "--input_log",
                str(work / f"out_{name}/qnn-profiling-data_0.log"),
            ],
            work / f"{name}.profile.txt",
            qairt_env(args.sdk, args.qnn_python),
        )
        print("Executed once", name, flush=True)


def compare(args: argparse.Namespace) -> None:
    work = args.work_dir
    manifest = json.loads((work / "manifest.json").read_text())
    reports = {}
    for name, info in manifest["graphs"].items():
        outputs = work / f"out_{name}/Result_0"
        profile = parse_profile((work / f"{name}.profile.txt").read_text())
        if profile["executes"] != 1:
            raise ValueError(f"Expected one execute: {name}")
        if info["operation"] == "encode":
            shape = (8, 1, info["seq"])
            packed = read_native(outputs, info["outputs"][0], "uint8", (*shape, 64))
            scale = read_native(outputs, info["outputs"][1], "float16", (*shape, 1))
            cfg = get_profile(info["profile"])
            report = compare_encode(
                PolarQuantReference(cfg.key, precomputed_norm=True),
                np.load(work / f"{name}_x.npy"),
                packed,
                scale,
                HTP_FP16,
            )
        else:
            expected = np.load(work / f"{name}_expected.npz")
            actual = read_native(outputs, "score", "float16", expected["score"].shape)
            error = actual.astype(np.float64) - expected["score"]
            lut_error = actual.astype(np.float64) - expected["lut"]
            report = {
                "passed": bool(
                    np.all(np.isfinite(actual)) and np.max(np.abs(error)) < 0.03
                ),
                "qk_max_abs_vs_own_oracle": float(np.abs(error).max()),
                "qk_rmse_vs_own_oracle": float(np.sqrt(np.mean(error**2))),
                "qk_max_abs_vs_lut": float(np.abs(lut_error).max()),
                "qk_rmse_vs_lut": float(np.sqrt(np.mean(lut_error**2))),
            }

            def softmax(scores: np.ndarray) -> np.ndarray:
                z = scores.astype(np.float64)
                z -= z.max(axis=-1, keepdims=True)
                e = np.exp(z)
                return e / e.sum(axis=-1, keepdims=True)

            v = np.random.default_rng(893).normal(size=(8, 1, info["tokens"], 128))
            attention_error = (softmax(actual) - softmax(expected["lut"])) @ v
            report["attention_max_abs_vs_lut"] = float(np.abs(attention_error).max())
            report["attention_rmse_vs_lut"] = float(
                np.sqrt(np.mean(attention_error**2))
            )
        reports[name] = {"validation": report, "profile": profile}
    output = {
        "scope": "single-run isolated HTP graphs, not end-to-end speedup",
        "graphs": reports,
    }
    (work / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "run", "compare", "all"])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--sequences", type=int, nargs="+", default=[1, 128])
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument(
        "--device-dir",
        default="/data/local/tmp/qaihm_turboquant/structured_bitplane_20261006_micro",
    )
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--qnn-python", type=Path, default=DEFAULT_QNN_PYTHON)
    parser.add_argument("--ndk", type=Path, default=DEFAULT_NDK)
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    args = parser.parse_args()
    for stage, function in (("build", build), ("run", run), ("compare", compare)):
        if args.stage in (stage, "all"):
            function(args)


if __name__ == "__main__":
    main()
