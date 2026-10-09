# SPDX-License-Identifier: BSD-3-Clause
"""HTP current/past/GQA Native-LUT probe on real captured values (not timing).

Uses the repository's audited small attention fixture with two KV heads / two
GQA groups. The deployment graph is all-FP16 around the codec; full-model
calibrated attention grids are checked separately in the bundle audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from convert_parts import build_context
from htp_codec_validation import (
    DEFAULT_ADB,
    DEFAULT_NDK,
    DEFAULT_QNN_PYTHON,
    DEFAULT_SDK,
    qairt_env,
    run_logged,
)
from learn_key_rotation import CENTROIDS, VR, codec, rounded
from onnx.reference import ReferenceEvaluator
from rotation_data import experiment_name, write_json
from validate_native_decoder import run as run_native

from qai_hub_models.models.templates.llm.turboquant.config import (
    TurboQuantConfig,
    get_profile,
)
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
from qai_hub_models.models.templates.llm.turboquant.numerics import (
    HTP_FP16,
    compare_encode,
)
from qai_hub_models.models.templates.llm.turboquant.packing import (
    pack_indices,
    unpack_indices,
)
from qai_hub_models.models.templates.llm.turboquant.reference import PolarQuantReference
from qai_hub_models.models.templates.llm.turboquant.rotation_artifact import (
    with_key_rotation,
)
from qai_hub_models.models.templates.llm.turboquant.tiled_attention import (
    tile_kv_attention,
)
from qai_hub_models.test.test_models import test_turboquant_tiled_attention as fixture


def build(args: argparse.Namespace, config: TurboQuantConfig) -> None:
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    fixture.CONTEXT = 1024
    data = np.load(args.root / "samples/test_0_layer00.npz")
    env = qairt_env(args.sdk, args.qnn_python)
    args.native_decoder_package = args.package
    for seq in args.tokens:
        name = f"native_t{seq}"
        model, enc = fixture.attention_part(seq)
        result = apply_kv_profile(model, enc, config, seq, 1024)
        result = tile_kv_attention(result, config, 256, rotated=True)
        result = use_native_decoder(result, config)
        result = quantize_current_attention(result, config)
        onnx.save(result.model, work / f"{name}.onnx")
        feeds = {}
        for kind, field in (("key", "k"), ("value", "v")):
            values = data[field][:2, None].astype(np.float32)
            ref = PolarQuantReference(getattr(config, kind), precomputed_norm=True)
            _, indices, scales = codec(
                torch.tensor(values[:, :, : 1024 - seq]),
                torch.tensor(ref.rotation.matrix().astype(np.float32)),
            )
            feeds[f"tq_{kind}_0_packed_in"] = pack_indices(indices.numpy(), 4)
            feeds[f"tq_{kind}_0_scale_in"] = scales.numpy()
            feeds[f"new_{kind}"] = values[:, :, 1024 - seq :]
        mask = np.zeros((1, 1, seq, 1024), np.float32)
        for i in range(seq):
            mask[0, 0, i, 1024 - seq + i + 1 :] = -100
        feeds["mask"] = mask
        for head in range(2):
            for group in range(2):
                feeds[f"h{head}g{group}_q"] = data["q"][
                    2 * head + group : 2 * head + group + 1, None, 1024 - seq :
                ].astype(np.float32)
        lines = []
        for key, value in feeds.items():
            value = value.astype(np.uint8 if value.dtype == np.uint8 else np.float16)
            filename = f"{name}_{key}.raw"
            value.tofile(work / filename)
            lines.append(f"{key}:={args.device_dir}/{filename}")
            feeds[key] = value if value.dtype == np.uint8 else value.astype(np.float32)
        (work / f"{name}_inputs.txt").write_text(" ".join(lines) + "\n")
        oracle = with_reference_decoder(result.model)
        actual = ReferenceEvaluator(oracle).run(None, feeds)
        np.savez(
            work / f"{name}_cpu.npz",
            **dict(zip([v.name for v in oracle.graph.output], actual, strict=True)),
        )
        np.savez(work / f"{name}_inputs.npz", **feeds)
        write_json(
            work / f"{name}_outputs.json",
            {
                v.name: list(x.shape)
                for v, x in zip(oracle.graph.output, actual, strict=True)
            },
        )
        run_logged(
            [
                str(args.qnn_python),
                str(args.sdk / "bin/x86_64-linux-clang/qairt-converter"),
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
        build_context(args, [name], name, work, env)
        print("BUILT PROBE", args.group, seq, flush=True)
    write_json(
        work / "manifest.json",
        {
            "config": config.to_dict(),
            "config_hash": config.config_hash(),
            "purpose": "functional HTP probe, not full-model calibrated Attention fidelity or timing",
            "attention_relative_l2_gate": 0.03,
            "encoder_gate": "existing HTP_FP16 tolerance, failures reported separately",
        },
    )


def conditioned_attention(
    inputs: dict, outputs: dict, config: TurboQuantConfig
) -> np.ndarray:
    """Isolate Attention arithmetic using HTP's actual current codes/scales.

    This deliberately does not validate encoder agreement. Its independent
    golden gate and the unconditioned FP32 graph comparison remain reported.
    CPU is an offline oracle, not an inference fallback.
    """
    kv = {}
    for kind in ("key", "value"):
        packed = np.concatenate(
            [inputs[f"tq_{kind}_0_packed_in"], outputs[f"tq_{kind}_0_packed_out"]],
            axis=2,
        )
        scale = np.concatenate(
            [inputs[f"tq_{kind}_0_scale_in"], outputs[f"tq_{kind}_0_scale_out"]],
            axis=2,
        )
        indices = torch.tensor(unpack_indices(packed, 4, 128).astype(np.int64))
        kv[kind] = rounded(CENTROIDS[indices] * torch.tensor(scale.astype(np.float32)))[
            :, 0
        ].repeat_interleave(2, 0)
    q = torch.tensor(
        np.concatenate(
            [inputs[f"h{h}g{g}_q"] for h in range(2) for g in range(2)], axis=0
        )[:, 0]
    )
    rotation = torch.tensor(
        PolarQuantReference(config.key).rotation.matrix().astype(np.float32)
    )
    score = rounded(rounded(q @ rounded(rotation).T) @ kv["key"].transpose(-1, -2))
    prob = rounded(
        torch.softmax(rounded(score + torch.tensor(inputs["mask"][0])), dim=-1)
    )
    total = None
    for start in range(0, 1024, 256):
        partial = rounded(
            prob[:, :, start : start + 256] @ kv["value"][:, start : start + 256]
        )
        total = partial if total is None else rounded(total + partial)
    return rounded(total @ rounded(VR)).numpy()


def compare(args: argparse.Namespace, config: TurboQuantConfig) -> None:
    torch.set_num_threads(4)
    report = {}
    for seq in args.tokens:
        name = f"native_t{seq}"
        shapes = json.loads((args.work_dir / f"{name}_outputs.json").read_text())
        cpu = np.load(args.work_dir / f"{name}_cpu.npz")
        inputs = np.load(args.work_dir / f"{name}_inputs.npz")
        directory = args.work_dir / "device" / f"out_{name}" / "Result_0"
        outputs = {
            key: np.fromfile(
                directory / f"{key}_native.raw",
                dtype=np.uint8 if "packed" in key else np.float16,
            ).reshape(shape)
            for key, shape in shapes.items()
        }
        errors, references = [], []
        for key, value in outputs.items():
            if key.startswith("h") and key.endswith("_out"):
                errors.append((value.astype(np.float64) - cpu[key]).ravel())
                references.append(cpu[key].astype(np.float64).ravel())
        error, reference = np.concatenate(errors), np.concatenate(references)
        rel = float(np.linalg.norm(error) / np.linalg.norm(reference))
        conditioned = conditioned_attention(inputs, outputs, config)
        device_attention = np.concatenate(
            [outputs[f"h{h}g{g}_out"] for h in range(2) for g in range(2)], axis=0
        )[:, 0].astype(np.float64)
        conditioned_error = device_attention - conditioned
        conditioned_rel = float(
            np.linalg.norm(conditioned_error) / np.linalg.norm(conditioned)
        )
        entry = {
            "attention_relative_l2_vs_fp32_graph_oracle": rel,
            "attention_max_abs": float(abs(error).max()),
            "fp32_graph_oracle_passed": rel < 0.03,
            "attention_conditioned_relative_l2": conditioned_rel,
            "attention_conditioned_max_abs": float(abs(conditioned_error).max()),
            "attention_conditioned_passed": conditioned_rel < 0.03,
            "comparison_note": "Conditioned oracle isolates Attention using actual HTP current codes/scales; it does not waive encoder or end-to-end FP32 oracle failures. Both use the unchanged 0.03 Attention gate.",
            "encoder": {},
        }
        for kind in ("key", "value"):
            ref = PolarQuantReference(getattr(config, kind), precomputed_norm=True)
            entry["encoder"][kind] = compare_encode(
                ref,
                inputs[f"new_{kind}"],
                outputs[f"tq_{kind}_0_packed_out"],
                outputs[f"tq_{kind}_0_scale_out"],
                HTP_FP16,
            )
        report[name] = entry
    write_json(args.work_dir / "validation_isolated.json", report)
    if not all(x["attention_conditioned_passed"] for x in report.values()):
        raise ValueError(
            "Isolated Attention probe failed (encoder and full graph gates reported independently)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "run", "compare", "all"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--group", choices=list("ABC"), required=True)
    parser.add_argument(
        "--package",
        type=Path,
        default=Path(
            "/mnt/d/ai-hub-models/binaries/turboquant/native_decoder_hvx_20260918"
        ),
    )
    args = parser.parse_args()
    args.work_dir = args.root / "probes" / args.group
    args.device_dir = (
        "/data/local/tmp/qaihm_turboquant/"
        + experiment_name(args.root)
        + "_probe_"
        + args.group
    )
    args.sdk, args.qnn_python, args.adb, args.ndk = (
        DEFAULT_SDK,
        DEFAULT_QNN_PYTHON,
        DEFAULT_ADB,
        DEFAULT_NDK,
    )
    args.tokens = [1, 128]
    config = with_key_rotation(
        get_profile("k4_v4_scaled"),
        None if args.group == "A" else args.root / f"rotations/{args.group}.json",
    )
    if args.stage in ("build", "all"):
        build(args, config)
    if args.stage in ("run", "all"):
        run_native(args)
    if args.stage in ("compare", "all"):
        compare(args, config)


if __name__ == "__main__":
    main()
