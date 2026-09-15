# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Minimal on-device HTP validation of the TurboQuant 4-bit codec graphs (spec P2).

Stages:
  build    ONNX graphs -> onnxruntime check vs float64 oracle -> qairt-converter
           (fp16) -> offline HTP context binaries for SM8850 (soc_model 87, V81).
  run      Push QAIRT runtime + contexts + inputs over adb, run qnn-net-run with
           libQnnHtp.so (basic and detailed profiling), pull outputs and profiles.
  compare  Check device outputs against the oracle with the HTP fp16 tolerance
           and summarize per-op accelerator profiling evidence.

qnn-net-run + libQnnHtp.so has no CPU partitioning: an op the HTP cannot run
fails graph prepare/finalize instead of silently falling back.

Usage (host tools from QAIRT, adb from Windows under WSL):

    PYTHONPATH=src python scripts/llm/turboquant/htp_codec_validation.py all \
        --work-dir ~/.qaihm/tmp/turboquant/p2 \
        --snapshot ~/.qaihm/tmp/turboquant/qwen3_1_7b_kv_snapshot.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from qai_hub_models.models.templates.llm.turboquant.config import (
    KVCodecSpec,
    get_profile,
)
from qai_hub_models.models.templates.llm.turboquant.export import (
    build_decode_model,
    build_encode_model,
)
from qai_hub_models.models.templates.llm.turboquant.numerics import (
    FLOAT32_GRAPH,
    HTP_FP16,
    compare_decode,
    compare_encode,
    reconstruction_stats,
)
from qai_hub_models.models.templates.llm.turboquant.packing import (
    pack_indices,
    to_storage_norms,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.reference import (
    PolarQuantReference,
)

PROFILE = "k4_v4"
HEADS, D = 8, 128
TOKEN_COUNTS = (128, 1)
SOC_MODEL = 87
DSP_ARCH = "v81"
DEFAULT_SDK = Path("~/qairt/2.48.0.260626").expanduser()
DEFAULT_QNN_PYTHON = Path("~/qnn-venv/bin/python").expanduser()
DEFAULT_NDK = Path("~/android-ndk-r26c").expanduser()
DEFAULT_ADB = Path("/mnt/c/adb/adb.exe")
DEVICE_DIR = "/data/local/tmp/qaihm_turboquant/p2"
DEVICE_LIBS = (
    "lib/aarch64-android/libQnnHtp.so",
    "lib/aarch64-android/libQnnHtpV81Stub.so",
    "lib/aarch64-android/libQnnSystem.so",
    "lib/aarch64-android/libQnnHtpNetRunExtensions.so",
    "lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so",
    "bin/aarch64-android/qnn-net-run",
)
NDK_LIBCXX = "toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
PROFILING_LEVELS = ("basic", "detailed")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph_specs() -> list[dict[str, Any]]:
    config = get_profile(PROFILE)
    graphs = []
    for which in ("key", "value"):
        for tokens in TOKEN_COUNTS:
            for op in ("encode", "decode"):
                graphs.append(
                    {
                        "name": f"tq_{op}_{which}_t{tokens}",
                        "op": op,
                        "which": which,
                        "tokens": tokens,
                        "seed": getattr(config, which).seed,
                    }
                )
    return graphs


def codec_for(which: str) -> tuple[KVCodecSpec, PolarQuantReference]:
    config = get_profile(PROFILE)
    spec = getattr(config, which)
    return spec, PolarQuantReference(spec, config.block_size)


def range_case(tokens: int, seed: int) -> np.ndarray:
    """Rows spanning the format's input domain: max|x| in [1e-2, 6.5e4], norm <= 6.5e4.

    Cycles one-hot, Gaussian and constant rows; row 0 is a zero vector.
    """
    rng = np.random.default_rng(seed)
    n = HEADS * tokens
    rows = np.zeros((n, D))
    for r, mag in enumerate(np.geomspace(1e-2, 6.5e4, n)):
        kind = r % 3
        if kind == 0:
            rows[r, r % D] = mag if r % 2 else -mag
        elif kind == 1:
            g = rng.standard_normal(D)
            g *= mag / np.abs(g).max()
            rows[r] = g * min(1.0, 6.5e4 / np.linalg.norm(g))
        else:
            rows[r] = min(mag, 6.5e4 / np.sqrt(D)) * np.where(np.arange(D) % 2, 1, -1)
    rows[0] = 0.0
    return rows.reshape(1, HEADS, tokens, D)


def encode_cases(snapshot: Any, which: str, tokens: int) -> dict[str, np.ndarray]:
    """Real Qwen3 KV (first and last layer) plus the input-domain range case."""
    kind = "keys" if which == "key" else "values"
    start = 0 if tokens > 1 else 128
    cases = {
        f"layer{layer}": snapshot[f"layer{layer}_{kind}"][
            None, :, start : start + tokens
        ]
        for layer in (0, 27)
    }
    cases["range"] = range_case(tokens, seed=tokens * 10 + (which == "value"))
    return {
        name: np.ascontiguousarray(x, dtype=np.float32) for name, x in cases.items()
    }


def qairt_env(sdk: Path, qnn_python: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["QNN_SDK_ROOT"] = str(sdk)
    env["PYTHONPATH"] = str(sdk / "lib/python")
    env["LD_LIBRARY_PATH"] = str(sdk / "lib/x86_64-linux-clang")
    env["PATH"] = f"{qnn_python.parent}:{sdk / 'bin/x86_64-linux-clang'}:{env['PATH']}"
    return env


def run_logged(cmd: list[str], log: Path, env: dict[str, str] | None = None) -> None:
    with log.open("w") as f:
        result = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=False
        )
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({result.returncode}); see {log}")


def ort_run(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return session.run(None, feeds)


def cmd_build(args: argparse.Namespace) -> None:
    work: Path = args.work_dir
    for sub in ("onnx", "dlc", "ctx", "inputs", "logs"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    snapshot = np.load(args.snapshot)
    env = qairt_env(args.sdk, args.qnn_python)
    config = get_profile(PROFILE)
    manifest: dict[str, Any] = {
        "profile": PROFILE,
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "qairt_sdk": str(args.sdk),
        "soc_model": SOC_MODEL,
        "dsp_arch": DSP_ARCH,
        "snapshot": str(args.snapshot),
        "snapshot_sha256": sha256_file(args.snapshot),
        "graphs": {},
    }

    for g in graph_specs():
        name = g["name"]
        spec, codec = codec_for(g["which"])
        builder = build_encode_model if g["op"] == "encode" else build_decode_model
        model = builder(config, spec, HEADS, g["tokens"], graph_name=name)
        onnx_path = work / "onnx" / f"{name}.onnx"
        onnx.save(model, onnx_path)

        cases = encode_cases(snapshot, g["which"], g["tokens"])
        input_lines, ort_reports = [], {}
        for case, x in cases.items():
            if g["op"] == "encode":
                x16 = x.astype(np.float16)
                raw = work / "inputs" / f"{name}_{case}_x.raw"
                raw.write_bytes(x16.tobytes())
                # The oracle must see the float16-rounded values the device receives.
                np.save(
                    work / "inputs" / f"{name}_{case}_x.npy", x16.astype(np.float32)
                )
                input_lines.append(f"x:=inputs/{raw.name}")
                packed, norm = ort_run(model, {"x": x})
                ort_reports[case] = compare_encode(
                    codec, x, packed, norm, FLOAT32_GRAPH
                )
            else:
                idx, norms = codec.encode(x)
                packed = pack_indices(idx, spec.bits)
                norm16 = to_storage_norms(norms, "float16")
                packed_raw = work / "inputs" / f"{name}_{case}_packed.raw"
                norm_raw = work / "inputs" / f"{name}_{case}_norm.raw"
                packed_raw.write_bytes(packed.tobytes())
                norm_raw.write_bytes(norm16.tobytes())
                input_lines.append(
                    f"packed:=inputs/{packed_raw.name} norm:=inputs/{norm_raw.name}"
                )
                (x_hat,) = ort_run(
                    model, {"packed": packed, "norm": norm16.astype(np.float32)}
                )
                ort_reports[case] = compare_decode(
                    codec, packed, norm16.astype(np.float32), x_hat, FLOAT32_GRAPH
                )
        (work / "inputs" / f"{name}_inputs.txt").write_text(
            "\n".join(input_lines) + "\n"
        )

        dlc_path = work / "dlc" / f"{name}.dlc"
        run_logged(
            [
                str(args.qnn_python),
                str(args.sdk / "bin/x86_64-linux-clang/qairt-converter"),
                "--input_network",
                str(onnx_path),
                "--output_path",
                str(dlc_path),
                "--float_bitwidth",
                "16",
            ],
            work / "logs" / f"{name}.convert.log",
            env,
        )
        run_logged(
            [
                str(args.qnn_python),
                str(args.sdk / "bin/x86_64-linux-clang/qairt-dlc-info"),
                "-i",
                str(dlc_path),
            ],
            work / "logs" / f"{name}.dlcinfo.txt",
            env,
        )
        htp_cfg = work / "ctx" / f"{name}_htp.json"
        htp_cfg.write_text(
            json.dumps(
                {
                    "graphs": [{"graph_names": [name], "O": 3}],
                    "devices": [
                        {
                            "soc_model": SOC_MODEL,
                            "dsp_arch": DSP_ARCH,
                            "pd_session": "unsigned",
                        }
                    ],
                }
            )
        )
        be_cfg = work / "ctx" / f"{name}_be.json"
        be_cfg.write_text(
            json.dumps(
                {
                    "backend_extensions": {
                        "shared_library_path": str(
                            args.sdk
                            / "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so"
                        ),
                        "config_file_path": str(htp_cfg),
                    }
                }
            )
        )
        run_logged(
            [
                str(args.sdk / "bin/x86_64-linux-clang/qnn-context-binary-generator"),
                "--backend",
                str(args.sdk / "lib/x86_64-linux-clang/libQnnHtp.so"),
                "--model",
                str(args.sdk / "lib/x86_64-linux-clang/libQnnModelDlc.so"),
                "--dlc_path",
                str(dlc_path),
                "--binary_file",
                name,
                "--config_file",
                str(be_cfg),
                "--output_dir",
                str(work / "ctx"),
            ],
            work / "logs" / f"{name}.ctxgen.log",
            env,
        )
        ctx_bin = work / "ctx" / f"{name}.bin"
        run_logged(
            [
                str(args.sdk / "bin/x86_64-linux-clang/qnn-context-binary-utility"),
                "--context_binary",
                str(ctx_bin),
                "--json_file",
                str(work / "ctx" / f"{name}.json"),
            ],
            work / "logs" / f"{name}.ctxinfo.log",
            env,
        )
        ctx_info = json.loads((work / "ctx" / f"{name}.json").read_text())
        manifest["graphs"][name] = {
            **g,
            "cases": list(cases),
            "onnx_sha256": sha256_file(onnx_path),
            "dlc_sha256": sha256_file(dlc_path),
            "context_sha256": sha256_file(ctx_bin),
            "context_metadata": _find_keys(
                ctx_info, ("dspArch", "socModel", "vtcmSize")
            ),
            "onnx_ops": sorted({n.op_type for n in model.graph.node}),
            "ort_float32_check": ort_reports,
        }
        print(f"built {name}", flush=True)

    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    failed = [
        f"{n}/{c}"
        for n, info in manifest["graphs"].items()
        for c, r in info["ort_float32_check"].items()
        if not r["passed"]
    ]
    if failed:
        raise RuntimeError(f"onnxruntime float32 check failed: {failed}")


def _find_keys(obj: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and k not in found:
                found[k] = v
            for kk, vv in _find_keys(v, keys).items():
                found.setdefault(kk, vv)
    elif isinstance(obj, list):
        for item in obj:
            for kk, vv in _find_keys(item, keys).items():
                found.setdefault(kk, vv)
    return found


def windows_path(path: Path) -> str:
    return subprocess.run(
        ["wslpath", "-w", str(path)], check=True, capture_output=True, text=True
    ).stdout.strip()


def adb(args: argparse.Namespace, *cmd: str, check: bool = True) -> str:
    result = subprocess.run(
        [str(args.adb), *cmd], capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"adb {' '.join(cmd)} failed: {result.stderr}")
    return result.stdout


def adb_shell_rc(args: argparse.Namespace, script: str) -> tuple[int, str]:
    out = adb(args, "shell", f"{script}; echo __RC__$?", check=False)
    match = re.search(r"__RC__(\d+)\s*$", out)
    return (int(match.group(1)) if match else -1), out


def cmd_run(args: argparse.Namespace) -> None:
    work: Path = args.work_dir
    device_out = work / "device"
    device_out.mkdir(parents=True, exist_ok=True)
    soc = adb(args, "shell", "getprop", "ro.soc.model").strip()
    if soc != "SM8850":
        raise RuntimeError(f"Contexts target SM8850 (V81) but device reports '{soc}'.")
    device_info = {
        "soc_model": soc,
        "model": adb(args, "shell", "getprop", "ro.product.model").strip(),
        "fingerprint": adb(args, "shell", "getprop", "ro.build.fingerprint").strip(),
    }

    adb(args, "shell", "rm", "-rf", DEVICE_DIR)
    adb(args, "shell", "mkdir", "-p", f"{DEVICE_DIR}/inputs")
    for rel in DEVICE_LIBS:
        adb(args, "push", windows_path(args.sdk / rel), f"{DEVICE_DIR}/")
    adb(args, "push", windows_path(args.ndk / NDK_LIBCXX), f"{DEVICE_DIR}/")
    adb(args, "shell", "chmod", "+x", f"{DEVICE_DIR}/qnn-net-run")
    for path in sorted((work / "ctx").glob("*.bin")):
        adb(args, "push", windows_path(path), f"{DEVICE_DIR}/")
    for path in sorted((work / "inputs").glob("*")):
        if path.suffix in (".raw", ".txt"):
            adb(args, "push", windows_path(path), f"{DEVICE_DIR}/inputs/")

    htp_dev = work / "device_htp.json"
    htp_dev.write_text(
        json.dumps(
            {
                "devices": [
                    {"soc_model": SOC_MODEL, "cores": [{"perf_profile": "burst"}]}
                ]
            }
        )
    )
    be_dev = work / "device_be.json"
    be_dev.write_text(
        json.dumps(
            {
                "backend_extensions": {
                    "shared_library_path": "./libQnnHtpNetRunExtensions.so",
                    "config_file_path": "./device_htp.json",
                }
            }
        )
    )
    adb(args, "push", windows_path(htp_dev), f"{DEVICE_DIR}/")
    adb(args, "push", windows_path(be_dev), f"{DEVICE_DIR}/")

    env_prefix = (
        f"cd {DEVICE_DIR} && export LD_LIBRARY_PATH={DEVICE_DIR}:/vendor/dsp/cdsp:/vendor/lib64 "
        # The QAIRT 2.48 android-qnn-net-run.sh V81 branch uses ':' separators.
        f"&& export ADSP_LIBRARY_PATH='{DEVICE_DIR}:/vendor/dsp/cdsp:/vendor/lib/rfsa/adsp:"
        "/system/lib/rfsa/adsp:/dsp'"
    )
    runs: dict[str, Any] = {"device": device_info, "runs": {}}
    for g in graph_specs():
        name = g["name"]
        for level in PROFILING_LEVELS:
            out_dir = f"out_{name}_{level}"
            local = device_out / out_dir
            profile_txt = device_out / f"{name}_{level}.profile.txt"
            # Never let results from an earlier run stand in for this one.
            shutil.rmtree(local, ignore_errors=True)
            profile_txt.unlink(missing_ok=True)
            rc, log = adb_shell_rc(
                args,
                f"{env_prefix} && ./qnn-net-run --backend libQnnHtp.so "
                f"--retrieve_context {name}.bin --input_list inputs/{name}_inputs.txt "
                "--use_native_input_files --use_native_output_files "
                f"--config_file device_be.json --profiling_level {level} "
                f"--output_dir {out_dir} --log_level info",
            )
            (device_out / f"{name}_{level}.log").write_text(log)
            runs["runs"][f"{name}/{level}"] = {"exit_code": rc}
            if rc != 0:
                print(f"{name} [{level}] failed with exit code {rc}", flush=True)
                continue
            adb(args, "pull", f"{DEVICE_DIR}/{out_dir}", windows_path(device_out))
            profile_log = local / "qnn-profiling-data_0.log"
            if profile_log.exists():
                run_logged(
                    [
                        str(args.sdk / "bin/x86_64-linux-clang/qnn-profile-viewer"),
                        "--reader",
                        str(
                            args.sdk
                            / "lib/x86_64-linux-clang/libQnnHtpProfilingReader.so"
                        ),
                        "--input_log",
                        str(profile_log),
                    ],
                    profile_txt,
                    qairt_env(args.sdk, args.qnn_python),
                )
            print(f"ran {name} [{level}]", flush=True)
    (work / "device_runs.json").write_text(json.dumps(runs, indent=2) + "\n")


def read_native(
    result_dir: Path, tensor: str, dtype: str, shape: tuple[int, ...]
) -> np.ndarray:
    candidates = [result_dir / f"{tensor}_native.raw", result_dir / f"{tensor}.raw"]
    for path in candidates:
        if path.exists():
            return np.fromfile(path, dtype=dtype).reshape(shape)
    raise FileNotFoundError(f"No output '{tensor}' in {result_dir}")


def parse_profile(text: str) -> dict[str, Any]:
    """Per-execute accelerator stats from ``qnn-profile-viewer`` output."""

    def ints(pattern: str) -> list[int]:
        return [int(v) for v in re.findall(pattern, text, re.MULTILINE)]

    op_cycles = re.findall(
        r"^\s+(\S+):OpId_\d+ \(cycles\) : (\d+)\s+cycles", text, re.MULTILINE
    )
    return {
        "executes": len(ints(r"^Execute Stat (\d+)")),
        "graph_ops_profiled_on_accelerator": sorted({name for name, _ in op_cycles}),
        "accelerator_execute_us": ints(r"^Accelerator \(execute\) time : (\d+)\s+us"),
        "qnn_execute_us": ints(r"^QNN \(execute\) time : (\d+)\s+us"),
        "rpc_execute_us": ints(r"^RPC \(execute\) time : (\d+)\s+us"),
        "hvx_threads": ints(r"^Number of HVX threads used : (\d+)"),
        "accelerator_cycles": ints(r"^Accelerator \(execute\) time \(cycles\) : (\d+)"),
    }


def dlc_op_names(dlc_info: Path) -> set[str]:
    """Op names from the ``qairt-dlc-info`` layer table (``| id | name | type |``)."""
    return set(
        re.findall(r"^\|\s*\d+\s*\|\s*(\S+)\s*\|", dlc_info.read_text(), re.MULTILINE)
    )


def cmd_compare(args: argparse.Namespace) -> None:
    work: Path = args.work_dir
    manifest = json.loads((work / "manifest.json").read_text())
    runs = json.loads((work / "device_runs.json").read_text())
    report: dict[str, Any] = {
        "scope": "S26 HTP execution of standalone codec graphs (P2); not full-model integration",
        "tolerance": HTP_FP16.__dict__,
        "device": runs["device"],
        "manifest_config_hash": manifest["config_hash"],
        "graphs": {},
    }
    all_passed = True
    for name, info in manifest["graphs"].items():
        spec, codec = codec_for(info["which"])
        tokens = info["tokens"]
        entry: dict[str, Any] = {"cases": {}, "profiles": {}}
        for level in PROFILING_LEVELS:
            run = runs["runs"].get(f"{name}/{level}", {})
            entry["profiles"][level] = {"exit_code": run.get("exit_code")}
            prof = work / "device" / f"{name}_{level}.profile.txt"
            if prof.exists():
                entry["profiles"][level].update(parse_profile(prof.read_text()))
        if any(
            entry["profiles"][level].get("exit_code") != 0 for level in PROFILING_LEVELS
        ):
            entry["passed"] = False
            all_passed = False
            report["graphs"][name] = entry
            continue
        out_dir = work / "device" / f"out_{name}_basic"
        for i, case in enumerate(info["cases"]):
            result = out_dir / f"Result_{i}"
            if info["op"] == "encode":
                x = np.load(work / "inputs" / f"{name}_{case}_x.npy")
                packed = read_native(
                    result, "packed", "uint8", (1, HEADS, tokens, D // 2)
                )
                norm = read_native(result, "norm", "float16", (1, HEADS, tokens, 1))
                cmp = compare_encode(codec, x, packed, norm, HTP_FP16)
                decoded = codec.decode(
                    unpack_indices(packed, spec.bits, D), norm.astype(np.float64)
                )
                cmp["reconstruction_with_device_indices"] = reconstruction_stats(
                    x, decoded
                )
                oracle = codec.decode(*codec.encode(x))
                cmp["reconstruction_oracle"] = reconstruction_stats(x, oracle)
            else:
                packed = np.fromfile(
                    work / "inputs" / f"{name}_{case}_packed.raw", dtype=np.uint8
                ).reshape(1, HEADS, tokens, D // 2)
                norm = np.fromfile(
                    work / "inputs" / f"{name}_{case}_norm.raw", dtype=np.float16
                ).reshape(1, HEADS, tokens, 1)
                x_hat = read_native(result, "x_hat", "float16", (1, HEADS, tokens, D))
                cmp = compare_decode(codec, packed, norm, x_hat, HTP_FP16)
            entry["cases"][case] = cmp
        profiled = set(
            entry["profiles"]["detailed"].get("graph_ops_profiled_on_accelerator", [])
        )
        dlc_ops = dlc_op_names(work / "logs" / f"{name}.dlcinfo.txt")
        entry["dlc_ops_missing_from_accelerator_profile"] = sorted(dlc_ops - profiled)
        entry["passed"] = (
            all(c["passed"] for c in entry["cases"].values())
            and bool(dlc_ops)
            and not entry["dlc_ops_missing_from_accelerator_profile"]
        )
        all_passed &= entry["passed"]
        report["graphs"][name] = entry
    report["passed"] = all_passed
    (work / "p2_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({n: g["passed"] for n, g in report["graphs"].items()}, indent=2))
    print(f"overall passed: {all_passed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "run", "compare", "all"])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--qnn-python", type=Path, default=DEFAULT_QNN_PYTHON)
    parser.add_argument("--ndk", type=Path, default=DEFAULT_NDK)
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    args = parser.parse_args()
    args.work_dir = args.work_dir.expanduser()
    args.snapshot = args.snapshot.expanduser()
    if args.stage in ("build", "all"):
        cmd_build(args)
    if args.stage in ("run", "all"):
        cmd_run(args)
    if args.stage in ("compare", "all"):
        cmd_compare(args)


if __name__ == "__main__":
    main()
