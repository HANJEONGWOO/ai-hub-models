# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Convert split LLM parts to QNN DLCs and HTP context binaries with local QAIRT.

Local stand-in for the Workbench compile + link used by the export pipeline:
``qairt-converter`` (AIMET encodings as quantization overrides) ->
``qairt-quantizer --enable_float_fallback`` (tensors without encodings, such as
TurboQuant codec subgraphs, stay float16) -> one weight-shared context binary
per part containing a prompt (AR=128) and a token (AR=1) graph.

Usage:

    python scripts/llm/turboquant/convert_parts.py \
        --split-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_w4a16_split \
        --out ~/.qaihm/tmp/turboquant/qwen3_1_7b_baseline_cl1024 \
        --context-length 1024 --parts 2
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import onnx

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.graph_surgery import (
    apply_kv_profile,
)
from qai_hub_models.models.templates.llm.turboquant.native_decoder import (
    use_native_decoder,
)
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)

DEFAULT_SDK = Path("~/qairt/2.48.0.260626").expanduser()
DEFAULT_QNN_PYTHON = Path("~/qnn-venv/bin/python").expanduser()
SOC_MODEL = 87
DSP_ARCH = "v81"
SEQUENCE_LENGTHS = (128, 1)


def graph_name(seq_len: int, ctx_len: int, part_id: int, num_parts: int) -> str:
    kind = "token" if seq_len == 1 else "prompt"
    return f"{kind}_ar{seq_len}_cl{ctx_len}_{part_id}_of_{num_parts}"


def input_shape(
    name: str, onnx_dims: list[Any], seq_len: int, ctx_len: int
) -> list[int]:
    """Static shape for one graph input, following the delta-KV naming contract."""
    past = ctx_len - seq_len
    if name == "input_ids":
        return [1, seq_len]
    if name == "attention_mask":
        return [1, 1, seq_len, ctx_len]
    if name in ("position_ids_cos", "position_ids_sin"):
        return [1, 1, seq_len, int(onnx_dims[3])]
    if re.fullmatch(r"past_key_\d+_in", name):
        return [int(onnx_dims[0]), 1, int(onnx_dims[2]), past]
    if re.fullmatch(r"past_value_\d+_in", name):
        return [int(onnx_dims[0]), 1, past, int(onnx_dims[3])]
    if re.fullmatch(r"tq_(key|value)_\d+_(packed|norm|scale)_in", name):
        return [int(onnx_dims[0]), 1, past, int(onnx_dims[3])]
    if len(onnx_dims) == 3:
        return [1, seq_len, int(onnx_dims[2])]
    raise ValueError(f"No shape rule for graph input '{name}' {onnx_dims}.")


def qairt_env(sdk: Path, qnn_python: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["QNN_SDK_ROOT"] = str(sdk)
    env["PYTHONPATH"] = str(sdk / "lib/python")
    env["LD_LIBRARY_PATH"] = str(sdk / "lib/x86_64-linux-clang")
    env["PATH"] = f"{qnn_python.parent}:{sdk / 'bin/x86_64-linux-clang'}:{env['PATH']}"
    return env


def run_logged(cmd: list[str], log: Path, env: dict[str, str]) -> float:
    start = time.monotonic()
    with log.open("w") as f:
        result = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=False
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"{Path(cmd[1]).name if len(cmd) > 1 else cmd[0]} failed; see {log}"
        )
    return time.monotonic() - start


def apply_profile(
    args: argparse.Namespace,
    onnx_path: Path,
    encodings: Path,
    name: str,
    seq_len: int,
    out: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Apply the profile to this graph; baseline graphs are converted as exported."""
    config = get_profile(args.profile)
    model = onnx.load(str(onnx_path), load_external_data=False)
    if not config.modifies_graph or not any(
        i.name.startswith("past_") for i in model.graph.input
    ):
        return onnx_path, encodings, {}
    result = apply_kv_profile(
        model, json.loads(encodings.read_text()), config, seq_len, args.context_length
    )
    if args.attention_tile:
        result = tile_kv_attention(
            result, config, args.attention_tile, rotated=args.rotated_attention
        )
    if args.native_decoder_package:
        result = use_native_decoder(result, config)
    # Initializers keep their external-data location, so expose the weights file here.
    for data in onnx_path.parent.glob("*.data"):
        link = out / data.name
        if not link.exists():
            link.symlink_to(data)
    new_onnx = out / f"{name}.onnx"
    onnx.save(result.model, str(new_onnx))
    new_encodings = out / f"{name}.encodings"
    new_encodings.write_text(json.dumps(result.encodings))
    report = result.report()
    (out / f"{name}.kv_edits.json").write_text(json.dumps(report, indent=1) + "\n")
    actions = [e["action"] for e in report["encoding_edits"]]
    summary = {
        "codec_io": report["codec_io"],
        "encodings_dropped": actions.count("drop"),
        "encodings_regridded": actions.count("regrid"),
        "tensors_duplicated": actions.count("duplicate"),
        "kv_paths": len(report["kv_paths"]),
        "attention_tile": args.attention_tile,
        "rotated_attention": args.rotated_attention,
        "native_decoder": bool(args.native_decoder_package),
    }
    return new_onnx, new_encodings, summary


def convert_graph(
    args: argparse.Namespace,
    onnx_path: Path,
    encodings: Path,
    name: str,
    seq_len: int,
    out: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    onnx_path, encodings, surgery = apply_profile(
        args, onnx_path, encodings, name, seq_len, out
    )
    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    shape_args: list[str] = []
    for value in graph.input:
        dims = [
            d.dim_param if d.dim_param else d.dim_value
            for d in value.type.tensor_type.shape.dim
        ]
        shape = input_shape(value.name, dims, seq_len, args.context_length)
        shape_args += [
            "--source_model_input_shape",
            value.name,
            ",".join(map(str, shape)),
        ]

    sdk_bin = args.sdk / "bin/x86_64-linux-clang"
    # The converter names the graph after the output file stem, so keep the stem clean.
    (out / "float").mkdir(exist_ok=True)
    float_dlc = out / "float" / f"{name}.dlc"
    dlc = out / f"{name}.dlc"
    timings: dict[str, Any] = {
        "surgery": surgery,
        "convert_s": run_logged(
            [
                str(args.qnn_python),
                str(sdk_bin / "qairt-converter"),
                "--input_network",
                str(onnx_path),
                "--quantization_overrides",
                str(encodings),
                *(
                    [
                        "--op_package_config",
                        str(Path(__file__).with_name("native_decoder") / "Decode4.xml"),
                    ]
                    if args.native_decoder_package
                    else []
                ),
                *shape_args,
                "--output_path",
                str(float_dlc),
            ],
            out / f"{name}.convert.log",
            env,
        ),
    }
    timings["quantize_s"] = run_logged(
        [
            str(args.qnn_python),
            str(sdk_bin / "qairt-quantizer"),
            "--input_dlc",
            str(float_dlc),
            "--output_dlc",
            str(dlc),
            "--enable_float_fallback",
            "--float_bitwidth",
            "16",
            "--act_bitwidth",
            "16",
            "--bias_bitwidth",
            "32",
        ],
        out / f"{name}.quantize.log",
        env,
    )
    run_logged(
        [str(args.qnn_python), str(sdk_bin / "qairt-dlc-info"), "-i", str(dlc)],
        out / f"{name}.dlcinfo.txt",
        env,
    )
    float_dlc.unlink()
    return timings


def build_context(
    args: argparse.Namespace,
    names: list[str],
    context_name: str,
    out: Path,
    env: dict[str, str],
) -> float:
    htp_cfg = out / f"{context_name}_htp.json"
    htp_cfg.write_text(
        json.dumps(
            {
                "graphs": [{"graph_names": names, "O": 3, "vtcm_mb": 0}],
                "devices": [
                    {
                        "soc_model": SOC_MODEL,
                        "dsp_arch": DSP_ARCH,
                        "pd_session": "unsigned",
                    }
                ],
                "context": {"weight_sharing_enabled": len(names) > 1},
            }
        )
    )
    be_cfg = out / f"{context_name}_be.json"
    be_cfg.write_text(
        json.dumps(
            {
                "backend_extensions": {
                    "shared_library_path": str(
                        args.sdk / "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so"
                    ),
                    "config_file_path": str(htp_cfg),
                }
            }
        )
    )
    sdk_bin = args.sdk / "bin/x86_64-linux-clang"
    seconds = run_logged(
        [
            str(sdk_bin / "qnn-context-binary-generator"),
            "--backend",
            str(args.sdk / "lib/x86_64-linux-clang/libQnnHtp.so"),
            "--model",
            str(args.sdk / "lib/x86_64-linux-clang/libQnnModelDlc.so"),
            *(
                [
                    "--op_packages",
                    f"{args.native_decoder_package / 'x86_64-linux-clang/libTurboQuantNative.so'}:TurboQuantInterfaceProvider",
                ]
                if args.native_decoder_package
                else []
            ),
            "--dlc_path",
            ",".join(str(out / f"{n}.dlc") for n in names),
            "--binary_file",
            context_name,
            "--config_file",
            str(be_cfg),
            "--output_dir",
            str(out),
        ],
        out / f"{context_name}.ctxgen.log",
        env,
    )
    run_logged(
        [
            str(sdk_bin / "qnn-context-binary-utility"),
            "--context_binary",
            str(out / f"{context_name}.bin"),
            "--json_file",
            str(out / f"{context_name}.json"),
        ],
        out / f"{context_name}.ctxinfo.log",
        env,
    )
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--profile", default="baseline_int8")
    parser.add_argument("--context-buckets", type=int, nargs="+", default=[])
    parser.add_argument("--rotated-attention", action="store_true")
    parser.add_argument(
        "--native-decoder-package",
        type=Path,
        help="Opt-in built QHPI Decode4 package directory; requires rotated tiled attention",
    )
    parser.add_argument(
        "--attention-tile",
        type=int,
        default=0,
        help="Opt-in two-pass KV restore/attention tiling; 0 keeps the full restore.",
    )
    parser.add_argument("--parts", type=int, nargs="*", default=[])
    parser.add_argument(
        "--sequence-lengths", type=int, nargs="+", default=list(SEQUENCE_LENGTHS)
    )
    parser.add_argument("--skip-context", action="store_true")
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Finalize previously converted graphs without reconversion",
    )
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--qnn-python", type=Path, default=DEFAULT_QNN_PYTHON)
    args = parser.parse_args()
    if args.context_only and args.skip_context:
        parser.error("--context-only and --skip-context are mutually exclusive")
    if args.attention_tile < 0:
        parser.error("--attention-tile must be nonnegative")
    if args.attention_tile and args.profile not in ("k4_v4", "k4_v4_scaled"):
        parser.error("--attention-tile requires a k4_v4 profile")
    if args.rotated_attention and (
        args.profile != "k4_v4_scaled" or not args.attention_tile
    ):
        parser.error("--rotated-attention requires k4_v4_scaled and --attention-tile")
    if args.native_decoder_package:
        args.native_decoder_package = args.native_decoder_package.expanduser().resolve()
        if not args.rotated_attention:
            parser.error("--native-decoder-package requires --rotated-attention")
        for target in ("x86_64-linux-clang", "hexagon-v81"):
            if not (
                args.native_decoder_package / target / "libTurboQuantNative.so"
            ).is_file():
                parser.error(f"Missing native decoder library for {target}")
    buckets = sorted({*args.context_buckets, args.context_length})
    if any(c <= 1 or c > args.context_length for c in buckets):
        parser.error("context buckets must be in [2, context-length]")

    split_dir = args.split_dir.expanduser()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((split_dir / "split_manifest.json").read_text())
    env = qairt_env(args.sdk, args.qnn_python)
    num_parts = len(manifest["parts"])
    report_path = out / "convert_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    config = get_profile(args.profile)
    if args.context_only:
        native_manifest = (
            json.loads((args.native_decoder_package / "manifest.json").read_text())
            if args.native_decoder_package
            else None
        )
        expected = {
            "config_hash": config.config_hash(),
            "attention_tile": args.attention_tile,
            "rotated_attention": args.rotated_attention,
            "context_buckets": buckets,
            "native_decoder": native_manifest,
        }
        for key, value in expected.items():
            if report.get(key) != value:
                raise ValueError(f"--context-only metadata mismatch: {key}")
    report.update(
        {
            "context_length": args.context_length,
            "sdk": str(args.sdk),
            "profile": args.profile,
            "config": config.to_dict(),
            "config_hash": config.config_hash(),
            "attention_tile": args.attention_tile,
            "rotated_attention": args.rotated_attention,
            "context_buckets": buckets,
            "native_decoder": json.loads(
                (args.native_decoder_package / "manifest.json").read_text()
            )
            if args.native_decoder_package
            else None,
        }
    )
    report.setdefault("parts", {})

    for part_id, (part_name, info) in enumerate(manifest["parts"].items(), start=1):
        if args.parts and part_id not in args.parts:
            continue
        bundle = Path(info["bundle_dir"])
        onnx_path = bundle / f"{info['class']}.onnx"
        encodings = bundle / f"{info['class']}.encodings"
        entry: dict[str, Any] = {"graphs": {}}
        names = []
        for context in buckets:
            graph_args = copy.copy(args)
            graph_args.context_length = context
            for seq_len in args.sequence_lengths:
                if seq_len >= context:
                    continue
                name = graph_name(seq_len, context, part_id, num_parts)
                if args.context_only:
                    entry["graphs"][name] = report["parts"][part_name]["graphs"][name]
                    if not (out / f"{name}.dlc").is_file():
                        raise FileNotFoundError(out / f"{name}.dlc")
                else:
                    entry["graphs"][name] = convert_graph(
                        graph_args, onnx_path, encodings, name, seq_len, out, env
                    )
                names.append(name)
                print(
                    f"{part_name}: {'reused DLC' if args.context_only else 'converted'} {name}",
                    flush=True,
                )
        if not args.skip_context:
            entry["context_s"] = build_context(args, names, part_name, out, env)
            print(f"{part_name}: context {entry['context_s']:.0f}s", flush=True)
        report["parts"][part_name] = entry
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
