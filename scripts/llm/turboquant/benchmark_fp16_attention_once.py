# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""One-run CL1024 comparison of FP16 attention and current TQ, optionally int16.

Use explicit existing bundles; never rebuild or modify historical artifacts.
Attempts are exclusive-create, including failed runs. All groups use the same
runner, assets and fixed C1024 graph selection. Functional runs are not timings.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from model_identity import QWEN_SHAPES, sha256_file, validate_model
from summarize_native_results import performance, read

PROFILES = {
    "int16": "baseline_int16_kv",
    "fp16": "baseline_fp16_kv_fp16_attn",
    "turboquant": "k4_v4_scaled",
}
SCRIPTS = Path(__file__).resolve().parent


def validate_inputs(model_id: str, assets: dict, metadata: dict) -> tuple[int, int]:
    """Validate explicit model identity; retain support for legacy 1.7B bundles."""
    layers, _, _, _, head_dim = QWEN_SHAPES[model_id]
    legacy = model_id == "qwen3_1_7b"
    identities = [assets.get("model"), *(m.get("model") for m in metadata.values())]
    if not legacy and any(identity is None for identity in identities):
        raise ValueError("Explicit model/checkpoint identity is required")
    present = [identity for identity in identities if identity is not None]
    for identity in present:
        validate_model(identity)
        if identity["model_id"] != model_id or identity != present[0]:
            raise ValueError("Model/tokenizer identity differs between groups/assets")
    split_hashes = {m.get("split_manifest_sha256") for m in metadata.values()}
    if not legacy and (None in split_hashes or len(split_hashes) != 1):
        raise ValueError("Groups must share the same split checkpoint")
    total = len(next(iter(metadata.values()))["parts"])
    if total < 2 or (not present and total != 4):
        raise ValueError("Unexpected split part count")
    parts = {f"part{i}_of_{total}" for i in range(1, total + 1)}
    for group, data in metadata.items():
        if (
            data["profile"] != PROFILES[group]
            or data["context_length"] != 1024
            or 1024 not in data.get("context_buckets", [1024])
            or set(data["parts"]) != parts
            or data.get("num_parts", total) != total
        ):
            raise ValueError(f"Unexpected bundle: {group}")
    if assets.get("rope_half", head_dim // 2) != head_dim // 2:
        raise ValueError("Wrong model-specific RoPE dimensions")
    return total, layers


def expected_kv_bytes(model_id: str, group: str) -> int:
    layers, _, _, heads, dim = QWEN_SHAPES[model_id]
    width = dim // 2 + 2 if group == "turboquant" else dim * 2
    return layers * 2 * heads * 1024 * width


def validate_audit(audit: dict, bundle: Path, layers: int) -> None:
    if (
        not audit["passed"]
        or Path(audit["bundle"]).resolve() != bundle
        or not audit["graphs"]
        or any(g.get("violations") for g in audit["graphs"].values())
    ):
        raise ValueError("Compiled graph audit must pass for this bundle")
    for ar in (1, 128):
        observed = [
            layer
            for name, graph in audit["graphs"].items()
            if name.startswith(f"{'token' if ar == 1 else 'prompt'}_ar{ar}_cl1024_")
            for layer in graph["layers"]
        ]
        if sorted(observed) != list(range(layers)):
            raise ValueError("Audit is missing/duplicating model layers")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["push", "functional", "performance", "quality", "summarize"]
    )
    parser.add_argument("--model-id", choices=list(QWEN_SHAPES), default="qwen3_1_7b")
    parser.add_argument(
        "--groups", choices=list(PROFILES), nargs="+", default=list(PROFILES)
    )
    for group in PROFILES:
        parser.add_argument(f"--{group}-bundle", type=Path)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--name", default="fp16_attention_20260929")
    args = parser.parse_args()
    groups = args.groups
    if len(set(groups)) != len(groups) or not {"fp16", "turboquant"} <= set(groups):
        parser.error("Select distinct groups including fp16 and turboquant")
    if any(getattr(args, g + "_bundle") is None for g in groups):
        parser.error("Supply --<group>-bundle for every selected group")
    root = args.reports
    root.mkdir(parents=True, exist_ok=True)
    assets = read(args.assets / "assets.json")
    if (
        assets["context_length"] != 1024
        or not 0 < assets["prompt_tokens"] < 897
        or len(assets["wikitext_windows"]) != 4
    ):
        raise ValueError("Expected CL1024 assets and four quality windows")
    for name, digest in assets["sha256"].items():
        if sha256_file(args.assets / name) != digest:
            raise ValueError(f"Asset identity changed: {name}")
    bundles = {g: getattr(args, g + "_bundle").resolve() for g in groups}
    metadata = {g: read(p / "convert_report.json") for g, p in bundles.items()}
    total, layers = validate_inputs(args.model_id, assets, metadata)
    tq = metadata["turboquant"]
    if (
        not tq.get("quantize_current_kv")
        or not tq.get("native_decoder")
        or tq["config"]["rotation"] != "dense_qr"
        or tq["config"].get("qjl")
        or not tq.get("rotated_attention")
        or tq.get("attention_tile") != 256
    ):
        raise ValueError("Expected Dense+Native K4/V4, QJL-off, quantized current KV")
    identity = {
        "runner_sha256": sha256_file(args.runner),
        "fp16_graph_audit_sha256": sha256_file(root / "fp16_graph_audit.json"),
        "assets_sha256": assets["sha256"],
        "graph_contexts": [1024],
        "runs_per_group_condition": 1,
        "groups": {
            g: {
                "bundle": str(p),
                "conversion_sha256": sha256_file(p / "convert_report.json"),
                "bins_sha256": {
                    f"part{i}_of_{total}.bin": sha256_file(
                        p / f"part{i}_of_{total}.bin"
                    )
                    for i in range(1, total + 1)
                },
            }
            for g, p in bundles.items()
        },
    }
    if args.model_id != "qwen3_1_7b":
        identity["model"] = assets["model"]
        identity["turboquant_graph_audit_sha256"] = sha256_file(
            root / "turboquant_graph_audit.json"
        )
    experiment = root / "experiment.json"
    if experiment.exists():
        if read(experiment) != identity:
            raise ValueError("Frozen experiment inputs changed")
    elif args.stage == "push":
        with experiment.open("x") as stream:
            json.dump(identity, stream, indent=2)
    else:
        raise ValueError("Push/freeze experiment before measuring")

    def execute(options: list[str], log: Path) -> None:
        print("START", log.stem, flush=True)
        with log.open("x") as stream:
            subprocess.run(
                [sys.executable, str(SCRIPTS / "run_device_llm.py"), *options],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print("DONE", log.stem, flush=True)

    def run(group: str, label: str, options: list[str]) -> dict:
        path = root / f"{group}_{label}.json"
        if path.exists() or path.with_suffix(".log").exists():
            raise FileExistsError(f"Refusing repeated measurement: {path}")
        execute(
            [
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
            path.with_suffix(".stdout.log"),
        )
        data = read(path)
        if data["config_hash"] != metadata[group]["config_hash"] or data[
            "context_buckets"
        ] != [1024]:
            raise ValueError("Wrong runtime configuration")
        if (
            data["assets"]["tokens_sha256"]
            != assets["sha256"][data["assets"]["tokens_file"]]
        ):
            raise ValueError("Wrong measurement input")
        if metadata[group].get("model") and (
            data.get("model") != metadata[group]["model"]
            or data.get("split_manifest_sha256")
            != metadata[group].get("split_manifest_sha256")
            or data["assets"].get("rope_sha256") != assets["sha256"][assets["rope"]]
        ):
            raise ValueError("Wrong runtime model/checkpoint/RoPE identity")
        if (
            group == "turboquant"
            and data.get("native_decoder", {}).get("sha256")
            != tq["native_decoder"]["libraries"]["hexagon-v81"]["sha256"]
        ):
            raise ValueError("Native decoder differs from build")
        expected_bytes = expected_kv_bytes(args.model_id, group)
        if data["kv_store_bytes"] != expected_bytes:
            raise ValueError("Unexpected model-specific CL1024 host cache allocation")
        if group == "fp16" and (
            len(data["kv_streams"]) != layers * 2
            or any(s["dtype"] != "float16" for s in data["kv_streams"])
        ):
            raise ValueError(
                "Device cache does not contain FP16 streams for every layer"
            )
        return data

    if args.stage == "push":
        audit = read(root / "fp16_graph_audit.json")
        validate_audit(audit, bundles["fp16"], layers)
        if audit["config_hash"] != metadata["fp16"]["config_hash"]:
            raise ValueError("FP16 audit config differs from bundle")
        if args.model_id != "qwen3_1_7b":
            validate_audit(
                read(root / "turboquant_graph_audit.json"),
                bundles["turboquant"],
                layers,
            )
        for group, bundle in bundles.items():
            # Push writes runtime metadata; keep historical bundles immutable.
            staging = root / "staging" / group
            staging.mkdir(parents=True, exist_ok=True)
            for i in range(1, total + 1):
                target = staging / f"part{i}_of_{total}.bin"
                if not target.exists():
                    target.symlink_to(bundle / target.name)
            conversion = staging / "convert_report.json"
            if not conversion.exists():
                conversion.symlink_to(bundle / conversion.name)
            execute(
                [
                    "push",
                    "--bundle-dir",
                    str(staging),
                    "--name",
                    args.name + "_" + group,
                    "--runner",
                    str(args.runner),
                ],
                root / f"{group}_push.stdout.log",
            )
    elif args.stage == "functional":
        checks = {}
        for group in groups:
            reset = run(
                group,
                "reset",
                ["--mode", "generate", "--n-gen", "8", "--sessions", "2"],
            )
            sessions = reset["sessions"]
            passed = (
                len(sessions) == 2
                and sessions[0]["generated"] == sessions[1]["generated"]
                and all(
                    len(s["generated"]) == 8
                    and s["cached_tokens_at_end"] == assets["prompt_tokens"] + 7
                    for s in sessions
                )
            )
            if not passed:
                raise ValueError(f"Reset/cache functional check failed: {group}")
            eos = run(
                group, "eos", ["--mode", "generate", "--n-gen", "128", "--stop-on-eos"]
            )
            checks[group] = {
                "reset": passed,
                "eos": eos["stop_reason"] == "eos",
                "stop_reason": eos["stop_reason"],
            }
        (root / "functional.json").write_text(json.dumps(checks, indent=2) + "\n")
    elif args.stage == "performance":
        checks = read(root / "functional.json")
        if not all(c["reset"] for c in checks.values()):
            raise ValueError("Reset check must pass")
        for condition, tokens, prompt in (
            ("short", assets["prompt_ids"], assets["prompt_tokens"]),
            ("long", assets["boundary_prompt"], 897),
        ):
            for group in groups:
                run(
                    group,
                    f"{condition}_once",
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
                print(
                    json.dumps(
                        performance(root / f"{group}_{condition}_once.json", prompt)
                    ),
                    flush=True,
                )
    elif args.stage == "quality":
        for window in range(4):
            for group in groups:
                data = run(
                    group,
                    f"score_w{window}",
                    ["--mode", "score", "--tokens", assets["wikitext_windows"][window]],
                )
                if data["scored_tokens"] != 1023 or not math.isfinite(data["nll_sum"]):
                    raise ValueError("Invalid PPL scoring result")
    else:
        summary = {
            "experiment": str(experiment.resolve()),
            "runs_per_group_condition": 1,
            "functional": read(root / "functional.json"),
            "groups": {},
        }
        devices = set()
        for group in groups:
            windows = [read(root / f"{group}_score_w{i}.json") for i in range(4)]
            tokens = sum(w["scored_tokens"] for w in windows)
            nll = sum(w["nll_sum"] for w in windows)
            summary["groups"][group] = {
                "diagnostic_only": not summary["functional"][group]["eos"],
                "short": performance(
                    root / f"{group}_short_once.json", assets["prompt_tokens"]
                ),
                "long": performance(root / f"{group}_long_once.json", 897),
                "quality": {
                    "ppl": math.exp(nll / tokens),
                    "nll_sum": nll,
                    "scored_tokens": tokens,
                    "windows": [
                        {k: w[k] for k in ("ppl", "nll_sum", "scored_tokens")}
                        for w in windows
                    ],
                },
            }
            for path in [root / f"{group}_{c}_once.json" for c in ("short", "long")] + [
                root / f"{group}_score_w{i}.json" for i in range(4)
            ]:
                devices.add(json.dumps(read(path)["device"], sort_keys=True))
        if len(devices) != 1:
            raise ValueError("Device identity changed during comparison")
        summary["device"] = json.loads(devices.pop())
        summary["notes"] = (
            "Single runs; no variance estimate. TTFT excludes model loading. Fixed C1024. FP16 is untiled/unrotated; TQ retains tiled rotated attention and Native decoder. W4A16 and calibrated non-KV boundaries retained; not an all-FP16 model. Failed EOS groups are diagnostic."
        )
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
