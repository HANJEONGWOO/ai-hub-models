# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Fresh, fixed-CL1024 Qwen3-1.7B comparison; one run per group/condition.

LM tree + Native LUT / structured K + Native LUT / structured K + BitplaneQK4.
All builds/attempts are preserved. Defaults of existing workflows are untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
from model_identity import sha256_file
from onnx import numpy_helper
from run_device_llm import DEFAULT_ADB
from summarize_native_results import performance, read
from verify_kv_boundary import parse_dlcinfo

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.native_decoder import native_table
from qai_hub_models.models.templates.llm.turboquant.reference import load_codebook
from qai_hub_models.models.templates.llm.turboquant.structured import load_parameters

SCRIPTS = Path(__file__).resolve().parent
PROFILES = {
    "lm": "k4_v4_scaled",
    "structured_lut": "k4s_v4_scaled",
    "bitplane": "k4s_v4_bitplane",
}


def execute(command: list[str], log: Path) -> None:
    print("START", log, flush=True)
    with log.open("x") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print("DONE", log, flush=True)


def environment_snapshot(path: Path) -> None:
    """Read-only observations outside the timed runner; never change thermal state."""
    snapshot = {"host_utc": datetime.now(timezone.utc).isoformat()}
    for service in ("battery", "thermalservice"):
        try:
            result = subprocess.run(
                [str(DEFAULT_ADB), "shell", "dumpsys", service],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            snapshot[service] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            snapshot[service] = {"error": "Read-only snapshot timed out"}
    with path.open("x") as stream:
        json.dump(snapshot, stream, indent=2)


def audit_bundle(bundle: Path, group: str) -> dict:
    meta = read(bundle / "convert_report.json")
    cfg = get_profile(PROFILES[group])
    if (
        meta["config_hash"] != cfg.config_hash()
        or meta["context_buckets"] != [1024]
        or not meta["quantize_current_kv"]
        or not meta["native_decoder"]
    ):
        raise ValueError("Wrong frozen experiment configuration")
    graphs = {}
    for path in bundle.glob("*.onnx"):
        model = onnx.load(path, load_external_data=False)
        native = [n for n in model.graph.node if n.op_type == "Decode4"]
        bp = [n for n in model.graph.node if n.op_type == "BitplaneQK4"]
        constants = {t.name: t for t in model.graph.initializer}
        for kind in ("value",) if group == "bitplane" else ("key", "value"):
            table = native_table(cfg, kind)
            expected = load_codebook(4, 128, getattr(cfg, kind).codebook).astype(
                np.float16
            )
            if not np.array_equal(
                numpy_helper.to_array(constants[table]).ravel(), expected
            ):
                raise ValueError(f"Wrong {kind} Native LUT")
            if not any(n.input[2] == table for n in native):
                raise ValueError(f"Unused {kind} Native LUT")
        if group == "bitplane":
            if not bp or any("key" in n.name for n in native):
                raise ValueError("Missing packed QK or live restored K")
            beta = (
                numpy_helper.to_array(constants["tq_bitplane_beta_f32_bytes"])
                .ravel()
                .view("<f4")
            )
            if not np.array_equal(
                beta, np.asarray(load_parameters()["beta"], np.float32)
            ):
                raise ValueError("Exported beta differs from frozen codebook")
        elif bp or not native:
            raise ValueError("Wrong LUT reference graph")
        dlc = path.with_suffix(".dlcinfo.txt").read_text()
        if group == "bitplane" and "BitplaneQK4" not in dlc:
            raise ValueError("Packed QK absent from compiled DLC")
        if group == "bitplane":
            compiled = parse_dlcinfo(path.with_suffix(".dlcinfo.txt"))
            compiled_bp = [op for op in compiled.ops if op.op_type == "BitplaneQK4"]
            if len(compiled_bp) != len(bp) or any(
                [t.dtype for t in op.inputs]
                != ["Uint_8", "Float_16", "Float_16", "Uint_8"]
                or [t.dtype for t in op.outputs] != ["Float_16"]
                for op in compiled_bp
            ):
                raise ValueError("Compiled packed QK dtype/count contract changed")
        graphs[path.stem] = {
            "native_lut_ops": len(native),
            "bitplane_qk_ops": len(bp),
            "onnx_sha256": sha256_file(path),
            "dlc_sha256": sha256_file(path.with_suffix(".dlc")),
        }
    if len(graphs) != 6:  # embedding-only part 1 has no surgery ONNX.
        raise ValueError(f"Expected six edited 1.7B graphs; found {len(graphs)}")
    return {
        "passed": True,
        "graphs": graphs,
        "configuration": meta["config"],
        "config_hash": meta["config_hash"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["build", "audit", "push", "performance", "quality", "summarize"],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("/mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_w4a16_split"),
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("/mnt/d/ai-hub-models/binaries/turboquant/device_assets_cl1024"),
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument(
        "--groups", choices=list(PROFILES), nargs="+", default=list(PROFILES)
    )
    parser.add_argument("--name", default="structured_bitplane_20261006")
    args = parser.parse_args()
    root, groups = args.root.resolve(), args.groups
    if len(groups) != len(set(groups)):
        parser.error("Duplicate groups would repeat measurements")
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if args.stage == "build":
        for group in groups:
            execute(
                [
                    sys.executable,
                    str(SCRIPTS / "convert_parts.py"),
                    "--split-dir",
                    str(args.split_dir),
                    "--out",
                    str(root / group),
                    "--profile",
                    PROFILES[group],
                    "--native-decoder-package",
                    str(args.package),
                    "--context-length",
                    "1024",
                    "--sequence-lengths",
                    "128",
                    "1",
                ],
                reports / f"{group}_build.log",
            )
        return
    if args.stage == "audit":
        for group in groups:
            with (reports / f"{group}_audit.json").open("x") as stream:
                json.dump(audit_bundle(root / group, group), stream, indent=2)
        return
    if args.runner is None:
        parser.error("--runner is required for frozen measurements")
    assets = read(args.assets / "assets.json")
    for name, digest in assets["sha256"].items():
        if sha256_file(args.assets / name) != digest:
            raise ValueError("Asset identity changed")
    if (
        assets["context_length"] != 1024
        or len(assets["wikitext_windows"]) != 4
        or len({assets["sha256"][w] for w in assets["wikitext_windows"]}) != 4
    ):
        raise ValueError("Expected CL1024 and four distinct PPL windows")
    identity = {
        "runner_sha256": sha256_file(args.runner),
        "assets": assets["sha256"],
        "split_manifest_sha256": sha256_file(args.split_dir / "split_manifest.json"),
        "groups": {},
    }
    metadata = {}
    for group, profile_name in PROFILES.items():
        meta = read(root / group / "convert_report.json")
        metadata[group] = meta
        if not read(reports / f"{group}_audit.json")["passed"]:
            raise ValueError("Graph audit must pass")
        if meta["config_hash"] != get_profile(profile_name).config_hash():
            raise ValueError("Codebook/config changed since build")
        identity["groups"][group] = {
            "config_hash": meta["config_hash"],
            "conversion_sha256": sha256_file(root / group / "convert_report.json"),
            "bins": {
                p.name: sha256_file(p) for p in sorted((root / group).glob("part*.bin"))
            },
            "native": meta["native_decoder"],
        }
        if set(identity["groups"][group]["bins"]) != {
            f"part{i}_of_4.bin" for i in range(1, 5)
        }:
            raise ValueError("Missing/extra context binaries")
    if (
        len(
            {
                m["native_decoder"]["libraries"]["hexagon-v81"]["sha256"]
                for m in metadata.values()
            }
        )
        != 1
    ):
        raise ValueError("All groups must use the same Native package")
    frozen = reports / "experiment.json"
    if frozen.exists():
        if read(frozen) != identity:
            raise ValueError("Frozen experiment inputs changed")
    elif args.stage == "push":
        with frozen.open("x") as stream:
            json.dump(identity, stream, indent=2)
    else:
        raise ValueError("Push/freeze before measurements")

    def run(group: str, label: str, options: list[str]) -> dict:
        path = reports / f"{group}_{label}.json"
        if path.exists():
            raise FileExistsError(path)
        environment_snapshot(reports / f"{group}_{label}.environment_before.json")
        execute(
            [
                sys.executable,
                str(SCRIPTS / "run_device_llm.py"),
                "run",
                "--name",
                args.name + "_" + group,
                "--assets",
                str(args.assets),
                "--context-buckets",
                "1024",
                "--report",
                str(path),
                "--remote-report-tag",
                args.name + "_" + path.stem,
                *options,
            ],
            # run_device_llm owns <report>.log for raw device output. Keep
            # launcher stdout separate so two writers cannot overwrite it.
            reports / f"{group}_{label}.stdout.log",
        )
        environment_snapshot(reports / f"{group}_{label}.environment_after.json")
        data = read(path)
        if (
            data["config_hash"] != metadata[group]["config_hash"]
            or data["context_buckets"] != [1024]
            or data["kv_store_bytes"] != 28 * 2 * 8 * 1024 * 66
        ):
            raise ValueError("Wrong runtime cache/configuration")
        if (
            data["native_decoder"]["sha256"]
            != metadata[group]["native_decoder"]["libraries"]["hexagon-v81"]["sha256"]
        ):
            raise ValueError("Runtime package differs from build")
        if (
            data["assets"]["tokens_sha256"]
            != assets["sha256"][data["assets"]["tokens_file"]]
        ):
            raise ValueError("Wrong runtime tokens")
        return data

    if args.stage == "push":
        for group in groups:
            execute(
                [
                    sys.executable,
                    str(SCRIPTS / "run_device_llm.py"),
                    "push",
                    "--bundle-dir",
                    str(root / group),
                    "--name",
                    args.name + "_" + group,
                    "--runner",
                    str(args.runner),
                ],
                reports / f"{group}_push.log",
            )
    elif args.stage == "performance":
        for condition, tokens in (
            ("short", assets["prompt_ids"]),
            ("long", assets["boundary_prompt"]),
        ):
            for group in groups:
                run(
                    group,
                    condition + "_once",
                    [
                        "--mode",
                        "generate",
                        "--tokens",
                        tokens,
                        "--n-gen",
                        "128",
                        "--sessions",
                        "1",
                    ],
                )
    elif args.stage == "quality":
        for window, tokens in enumerate(assets["wikitext_windows"]):
            for group in groups:
                data = run(
                    group, f"score_w{window}", ["--mode", "score", "--tokens", tokens]
                )
                if data["scored_tokens"] != 1023 or not math.isfinite(data["nll_sum"]):
                    raise ValueError("Invalid quality run")
    else:
        summary = {
            "runs_per_group_condition": 1,
            "model": "Qwen3-1.7B W4A16",
            "context_length": 1024,
            "groups": {},
            "notes": "Single runs, no variance estimate; unprofiled timing excludes model load. V/AV unchanged. Process RSS/HWM is not total HTP memory. Four separate PPL windows, not four repetitions.",
        }
        devices = set()
        for group in PROFILES:
            windows = [read(reports / f"{group}_score_w{i}.json") for i in range(4)]
            nll, count = (
                sum(w["nll_sum"] for w in windows),
                sum(w["scored_tokens"] for w in windows),
            )
            summary["groups"][group] = {
                "short": performance(
                    reports / f"{group}_short_once.json", assets["prompt_tokens"]
                ),
                "long": performance(reports / f"{group}_long_once.json", 897),
                "ppl": math.exp(nll / count),
                "scored_tokens": count,
                "windows": [
                    {k: w[k] for k in ("ppl", "nll_sum", "scored_tokens")}
                    for w in windows
                ],
            }
            for path in [
                reports / f"{group}_{c}_once.json" for c in ("short", "long")
            ] + [reports / f"{group}_score_w{i}.json" for i in range(4)]:
                devices.add(json.dumps(read(path)["device"], sort_keys=True))
        if len(devices) != 1:
            raise ValueError("Device identity changed between groups/conditions")
        summary["device"] = json.loads(devices.pop())
        with (reports / "comparison.json").open("x") as stream:
            json.dump(summary, stream, indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
