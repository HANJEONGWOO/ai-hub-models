# SPDX-License-Identifier: BSD-3-Clause
"""Frozen two-group, single-run structured K8 comparison; no default changes."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import onnx
from benchmark_asymmetric_once import ARTIFACTS, digest, execute, read, save_new
from benchmark_mixed_once import evaluate
from summarize_native_results import performance

from qai_hub_models.models.templates.llm.turboquant.config import (
    TurboQuantConfig,
    get_profile,
)
from qai_hub_models.models.templates.llm.turboquant.structured_k8 import (
    codebook,
    metadata,
)

GROUPS = ("mixed_k6_v2_uniform", "mixed_k6_v2_uniform_structured_k8")
PREVIOUS = ARTIFACTS / "hardware_aligned_mixed_kv_20261010"
CAPTURES = ARTIFACTS / "k_rotation_quality_20261006"


def config_for(root: Path, group: str) -> TurboQuantConfig:
    return get_profile(group)


def prepare(root: Path) -> None:
    optimized = read(root / "codebook/optimization.json")
    if (
        optimized["coefficient_hex"] != metadata()["coefficient_hex"]
        or optimized["centroid_sha256"] != metadata()["codebook_sha256"]
    ):
        raise ValueError(
            "Offline fit differs from the frozen deployed structured constants"
        )
    assets = ARTIFACTS / "device_assets_cl1024"
    runner = ARTIFACTS / "asymmetric_kv_20261010/runner/qnn-llm-runner"
    native = PREVIOUS / "native_symmetric8"
    source, target = PREVIOUS / GROUPS[0], root / GROUPS[0]
    target.mkdir(parents=True, exist_ok=False)
    for file in source.iterdir():
        if file.is_file():
            (target / file.name).symlink_to(file.resolve())
    # Shared source auditor's internal baseline alias; no extra benchmark group.
    (root / "k4_v4").symlink_to(target, target_is_directory=True)
    manifest = read(assets / "assets.json")
    assert manifest["wikitext_windows"] == [f"wikitext_w{i}.bin" for i in range(4)]
    files = [
        runner,
        native / "manifest.json",
        source / "convert_report.json",
        root / "codebook/optimization.json",
    ]
    files += list(source.glob("part*_of_4.bin"))
    files += [
        Path(v["path"]) for v in read(native / "manifest.json")["libraries"].values()
    ]
    frozen = {str(p): digest(p) for p in files}
    frozen.update({str(assets / name): sha for name, sha in manifest["sha256"].items()})
    captured = read(CAPTURES / "samples/manifest.json")
    for name, sha in captured["sha256"].items():
        if name.startswith("validation_"):
            frozen[str(CAPTURES / "samples" / name)] = sha
    for name in (
        "samples/manifest.json",
        "data_manifest.json",
        "capture_identity.json",
    ):
        frozen[str(CAPTURES / name)] = digest(CAPTURES / name)
    profiles = {
        g: {
            "profile": g,
            "config": get_profile(g).to_dict(),
            "config_hash": get_profile(g).config_hash(),
        }
        for g in GROUPS
    }
    assert (
        profiles[GROUPS[0]]["config_hash"]
        == read(source / "convert_report.json")["config_hash"]
    )
    for config in map(get_profile, GROUPS):
        assert config.layer_bits == tuple((8 if i % 2 else 4, 2) for i in range(28))
    save_new(
        root / "protocol.json",
        {
            "model": "Qwen3-1.7B W4A16",
            "context_length": 1024,
            "branch": "exp/structured-k8-codebook",
            "base_branch": "exp/hardware-aligned-mixed-kv",
            "base_commit": "c022fecee",
            "groups": GROUPS,
            "profiles": profiles,
            "assets": str(assets),
            "assets_manifest": manifest,
            "runner": str(runner),
            "split": str(ARTIFACTS / "qwen3_1_7b_w4a16_split"),
            "mixed_native_override": str(native),
            "frozen_files": frozen,
            "structured_quantizer": metadata(),
            "performance": {
                "sessions_per_group_condition": 1,
                "conditions": {"short": [35, 128], "long": [897, 128]},
                "profiling": False,
                "dispersion": None,
            },
            "quality": {
                "ppl": "4 legacy WikiText windows, once/window, 4092 targets, exp(total NLL / count)",
                "offline": "two pre-existing validation documents, all 28 layers, positions 31:1024:32, causal scores only; no fitting/selection on captures",
                "captures": str(CAPTURES),
                "fp16_emulator": "FP16 operation boundaries and FP32 reductions, not HTP bit-exact",
            },
            "fixed": [
                "K8 odd layers 1,3,...27; K4 even layers",
                "V2 all layers",
                "Dense K42/V542",
                "norm correction and one FP16 effective scale",
                "same Native HVX LUT kernel binary",
                "same W4A16 and calibration",
                "compressed current/past KV",
                "tile256 FP16 attention",
                "8 payload bits plus metadata",
            ],
            "diagnostics": {
                "native": "same existing package; 1/3/256 token probes each LUT",
                "encoder": "AR1/128, real layer0/layer27 plus range*0.5; fixed 0.2% scale and 0.001 boundary tolerances",
                "optrace": "one separate 897+128 session; 44 traces/group; physical work not wall latency",
            },
        },
    )


def validate(root: Path) -> dict:
    protocol = read(root / "protocol.json")
    for name, sha in protocol["frozen_files"].items():
        if digest(Path(name)) != sha:
            raise ValueError("Frozen input changed: " + name)
    for group in GROUPS:
        if (
            get_profile(group).config_hash()
            != protocol["profiles"][group]["config_hash"]
        ):
            raise ValueError("Config changed after freeze")
    return protocol


def audit_k8_only(root: Path) -> None:
    """Reject changes outside odd-layer K8 encoder/index/LUT constants."""
    result = {}
    left, right = (root / g for g in GROUPS)
    for path in sorted(right.glob("*.kv_edits.json")):
        name = path.name.removesuffix(".kv_edits.json")
        graphs = [
            onnx.load(p / (name + ".onnx"), load_external_data=False).graph
            for p in (left, right)
        ]

        def nodes(graph: onnx.GraphProto) -> dict[str, bytes]:
            kept = {}
            for source in graph.node:
                node = onnx.NodeProto()
                node.CopyFrom(source)
                match = re.match(r"tq_key_(\d+)_(.*)", node.name)
                if match and int(match[1]) % 2:
                    if match[2].startswith(
                        (
                            "enc_scalar_",
                            "enc_scale_",
                            "enc_effective_scale",
                            "enc_index",
                        )
                    ):
                        continue
                    for i, inp in enumerate(node.input):
                        if inp in (
                            "tq_native_b8_centroids_fp16",
                            "tq_native_structured_k8_centroids_fp16",
                        ):
                            node.input[i] = "K8_VARIANT_TABLE"
                kept[node.name] = node.SerializeToString()
            return kept

        constants = [
            {t.name: t.SerializeToString() for t in g.initializer} for g in graphs
        ]
        common = constants[0].keys() & constants[1].keys()
        table = next(
            t
            for t in graphs[1].initializer
            if t.name == "tq_native_structured_k8_centroids_fp16"
        )
        checks = {
            "all_non_k8_quantizer_nodes_identical": nodes(graphs[0])
            == nodes(graphs[1]),
            "shared_constants_identical": all(
                constants[0][k] == constants[1][k] for k in common
            ),
            "io_identical": all(
                [t.SerializeToString() for t in getattr(graphs[0], field)]
                == [t.SerializeToString() for t in getattr(graphs[1], field)]
                for field in ("input", "output")
            ),
            "structured_lut_exact": bool(
                np.array_equal(
                    onnx.numpy_helper.to_array(table).ravel(),
                    codebook().astype(np.float16),
                )
            ),
        }
        result[name] = checks
    report = {
        "graphs": result,
        "passed": bool(result) and all(all(x.values()) for x in result.values()),
    }
    save_new(root / "reports/k8_only_audit.json", report)
    if not report["passed"]:
        raise ValueError(f"K8-only audit failed: {report}")


def summarize(root: Path, protocol: dict) -> None:
    groups, rows = {}, []
    for group in GROUPS:
        quality = [
            read(root / "reports" / f"{group}_score_w{i}.json") for i in range(4)
        ]
        assert all(q["scored_tokens"] == 1023 for q in quality)
        total = sum(q["nll_sum"] for q in quality)
        perf = {
            c: performance(root / "reports" / f"{group}_perf_{c}.json", n)
            for c, n in (("short", 35), ("long", 897))
        }
        native_hash = read(Path(protocol["mixed_native_override"]) / "manifest.json")[
            "libraries"
        ]["hexagon-v81"]["sha256"]
        if any(
            p["native_decoder"]["sha256"] != native_hash or p["host_kv_MiB"] != 28.875
            for p in perf.values()
        ):
            raise ValueError(
                "Measured native binary/cache budget differs from protocol"
            )
        groups[group] = {
            "config": protocol["profiles"][group],
            "performance": perf,
            "quality": {
                "mean_nll": total / 4092,
                "ppl": math.exp(total / 4092),
                "nll_sum": total,
                "scored_tokens": 4092,
                "windows": [
                    {k: q[k] for k in ("nll_sum", "scored_tokens", "ppl", "assets")}
                    for q in quality
                ],
            },
            "bins_sha256": {
                p.name: digest(p) for p in (root / group).glob("part*_of_4.bin")
            },
            "binary_bytes": sum(
                p.stat().st_size for p in (root / group).glob("part*_of_4.bin")
            ),
        }
        for condition, p in perf.items():
            rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "ppl": groups[group]["quality"]["ppl"],
                    **{
                        k: p[k]
                        for k in (
                            "ttft_ms",
                            "prefill_tok_per_s",
                            "decode_tok_per_s",
                            "host_kv_MiB",
                            "io_buffer_MiB",
                            "end_VmRSS_MiB",
                            "process_VmHWM_MiB",
                            "mean_decode_prepare_ms",
                            "mean_decode_qnn_ms",
                        )
                    },
                }
            )
    save_new(
        root / "comparison.json",
        {
            "sessions_per_condition": 1,
            "dispersion": None,
            "groups": groups,
            "protocol_sha256": digest(root / "protocol.json"),
            "offline_quality": read(root / "offline_quality.json"),
        },
    )
    with (root / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("prepare", "probe", "build", "evaluate", "summarize")
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    args = p.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        prepare(root)
        return
    protocol = validate(root)
    if args.stage == "probe":
        if any(
            (root / name).exists()
            for name in (
                "encoder_lm",
                "encoder_structured_checked",
                "decoder_lm",
                "decoder_structured",
            )
        ):
            raise FileExistsError(
                "Codec probes already exist; use a fresh experiment root"
            )
        for label, profile, encoder, lut in (
            ("lm", "k8_v4_scaled", "encoder_lm", "mse4"),
            (
                "structured",
                "structured_k8_v4_scaled",
                "encoder_structured_checked",
                "structured_k8",
            ),
        ):
            execute(
                "htp_codec_validation.py",
                [
                    "all",
                    "--work-dir",
                    str(root / encoder),
                    "--snapshot",
                    str(ARTIFACTS / "qwen3_1_7b_kv_snapshot.npz"),
                    "--profile",
                    profile,
                    "--range-scale",
                    "0.5",
                    "--tokens",
                    "1",
                    "128",
                    "--head-major",
                    "--operations",
                    "encode",
                    "--device-dir",
                    f"/data/local/tmp/qaihm_turboquant/{root.name}_{label}_encoder",
                ],
                root / "logs" / f"probe_encoder_{label}.log",
            )
            execute(
                "validate_native_decoder.py",
                [
                    "all",
                    "--work-dir",
                    str(root / f"decoder_{label}"),
                    "--package",
                    protocol["mixed_native_override"],
                    "--bits",
                    "8",
                    "--lut",
                    lut,
                    "--tokens",
                    "1",
                    "3",
                    "256",
                    "--device-dir",
                    f"/data/local/tmp/qaihm_turboquant/{root.name}_{label}_decoder",
                ],
                root / "logs" / f"probe_decoder_{label}.log",
            )
        execute(
            "summarize_structured_k8_probes.py",
            ["--root", str(root)],
            root / "logs" / "probe_summary.log",
        )
        return
    if args.stage == "summarize":
        summarize(root, protocol)
        return
    for group in args.groups:
        if args.stage == "evaluate":
            if group == GROUPS[1]:
                audit_k8_only(root)
            evaluate(root, group, protocol)
        elif group == GROUPS[1]:
            execute(
                "convert_parts.py",
                [
                    "--split-dir",
                    protocol["split"],
                    "--out",
                    str(root / group),
                    "--profile",
                    group,
                    "--context-length",
                    "1024",
                    "--sequence-lengths",
                    "1",
                    "128",
                    "--native-decoder-package",
                    protocol["mixed_native_override"],
                ],
                root / "logs" / f"build_{group}.log",
            )


if __name__ == "__main__":
    main()
