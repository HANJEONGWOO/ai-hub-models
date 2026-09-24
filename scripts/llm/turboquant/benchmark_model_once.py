# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Build and compare int16 KV and current TurboQuant for an explicitly selected model.

Default model remains Qwen3-1.7B. Other sizes require explicit --model-id.
--cl1024-only builds fixed C1024 for both groups and measures only 897+128 tokens.
Stages are explicit; performance executes once per group/input condition and
never overwrites attempted measurements. Builds are serial to bound host RAM.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from model_identity import sha256_file, validate_model
from summarize_native_results import performance, read

MODELS = ("qwen3_0_6b", "qwen3_1_7b", "qwen3_4b")
GROUPS = {"baseline_int16": "baseline_int16_kv", "turboquant": "k4_v4_scaled"}
BUCKETS = {"baseline_int16": [1024], "turboquant": [128, 256, 512, 1024]}
SCRIPTS = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "stage",
        choices=[
            "prepare",
            "build",
            "audit",
            "push",
            "functional",
            "performance",
            "quality",
            "summarize",
        ],
    )
    result.add_argument("--model-id", choices=MODELS, default="qwen3_1_7b")
    result.add_argument("--work-dir", type=Path, required=True)
    result.add_argument("--checkpoint", default="DEFAULT")
    result.add_argument(
        "--cl1024-only",
        action="store_true",
        help="Use fixed C1024 for both groups; one long-input performance run each.",
    )
    result.add_argument(
        "--allow-eos-failure",
        action="store_true",
        help="Performance/summarize only: label EOS-failed groups as diagnostic; reset/cache checks remain mandatory.",
    )
    result.add_argument(
        "--runner",
        type=Path,
        default=Path("~/.qaihm/tmp/turboquant/qnn_runner/android-arm64/qnn-llm-runner"),
    )
    result.add_argument("--device-prefix", default=None)
    result.add_argument(
        "--groups", choices=list(GROUPS), nargs="+", default=list(GROUPS)
    )
    result.add_argument(
        "--parts",
        type=int,
        nargs="+",
        default=None,
        help="Build only these parts; still serial.",
    )
    return result


def execute(script: str, options: list[str], log: Path) -> None:
    print("START", log.stem, flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as output:
        subprocess.run(
            [sys.executable, str(SCRIPTS / script), *options],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("DONE", log.stem, flush=True)


def context_buckets(cl1024_only: bool = False) -> dict[str, list[int]]:
    return {g: [1024] if cl1024_only else list(BUCKETS[g]) for g in GROUPS}


def performance_conditions(cl1024_only: bool = False) -> tuple[str, ...]:
    return ("long",) if cl1024_only else ("short", "long")


def validate_bundles(
    work: Path, model_id: str, cl1024_only: bool = False
) -> dict[str, dict[str, Any]]:
    buckets = context_buckets(cl1024_only)
    assets = read(work / "assets/assets.json")
    split_hash = sha256_file(work / "split/split_manifest.json")
    split = read(work / "split/split_manifest.json")
    if split["model_id"] != model_id or split["model"] != assets.get("model"):
        raise ValueError(
            "Split/assets model mismatch; prepare a separate work directory."
        )
    validate_model(split["model"])
    if (
        assets["context_length"] != 1024
        or assets["rope_half"] != split["model"]["architecture"]["head_dim"] // 2
        or len(assets["wikitext_windows"]) != 4
    ):
        raise ValueError(
            "Expected CL1024, model-specific RoPE and four quality windows"
        )
    for name, digest in assets["sha256"].items():
        if sha256_file(work / "assets" / name) != digest:
            raise ValueError(f"Input asset changed: {name}")
    parts = set(split["parts"])
    total = len(parts)
    if parts != {f"part{i}_of_{total}" for i in range(1, total + 1)}:
        raise ValueError("Invalid split part set")
    result = {}
    for group, profile in GROUPS.items():
        bundle = work / group
        metadata = read(bundle / "convert_report.json")
        if (
            metadata.get("model") != split["model"]
            or metadata.get("split_manifest_sha256") != split_hash
            or metadata.get("num_parts") != total
            or metadata["profile"] != profile
            or metadata["context_length"] != 1024
            or metadata["context_buckets"] != buckets[group]
            or bool(metadata.get("quantize_current_kv")) != (group == "turboquant")
            or set(metadata["parts"]) != parts
        ):
            raise ValueError(f"Wrong model, checkpoint or configuration: {bundle}")
        if group == "turboquant" and (
            metadata["config"]["rotation"] != "dense_qr"
            or metadata["config"].get("qjl", False)
            or not metadata.get("native_decoder")
            or metadata["attention_tile"] != 256
            or not metadata["rotated_attention"]
        ):
            raise ValueError("Expected current Dense+Native K4/V4, QJL-off.")
        for part, entry in metadata["parts"].items():
            if "context_s" not in entry or not (bundle / f"{part}.bin").is_file():
                raise ValueError(f"Unfinalized part: {bundle / part}")
        result[group] = metadata
    return result


def validate_audit(audit: dict[str, Any], metadata: dict[str, Any]) -> None:
    graphs = audit["graphs"]
    if (
        not graphs
        or audit.get("passed") is False
        or any(g.get("violations") for g in graphs.values())
    ):
        raise ValueError("Compiled graph audit failed")
    expected_layers = set(range(metadata["model"]["architecture"]["num_hidden_layers"]))
    for context in metadata["context_buckets"]:
        for ar in (1, 128):
            if ar >= context:
                continue
            prefix = f"{'token' if ar == 1 else 'prompt'}_ar{ar}_cl{context}_"
            layers = [
                layer
                for name, graph in graphs.items()
                if name.startswith(prefix)
                for layer in graph.get("layers", [])
            ]
            if set(layers) != expected_layers or len(layers) != len(expected_layers):
                raise ValueError(
                    f"Audit is missing or duplicating model layers: {prefix}"
                )


def expected_kv_bytes(model: dict[str, Any], group: str) -> int:
    arch = model["architecture"]
    width = (
        arch["head_dim"] * 2 if group == "baseline_int16" else arch["head_dim"] // 2 + 2
    )
    return arch["num_hidden_layers"] * 2 * arch["num_key_value_heads"] * 1024 * width


def validate_run(
    data: dict[str, Any], metadata: dict[str, Any], assets: dict[str, Any], group: str
) -> None:
    tokens = data["assets"]["tokens_file"]
    if (
        data.get("model") != metadata["model"]
        or data.get("split_manifest_sha256") != metadata["split_manifest_sha256"]
        or data["config_hash"] != metadata["config_hash"]
        or data["context_buckets"] != metadata["context_buckets"]
        or bool(data.get("quantize_current_kv")) != (group == "turboquant")
        or data["assets"]["tokens_sha256"] != assets["sha256"][tokens]
        or data["assets"].get("rope_sha256") != assets["sha256"][assets["rope"]]
        or data["kv_store_bytes"] != expected_kv_bytes(metadata["model"], group)
    ):
        raise ValueError(f"Wrong measured model/assets/KV layout: {group}")
    if group == "turboquant":
        native = metadata["native_decoder"]
        if (
            data.get("native_decoder", {}).get("sha256")
            != native["libraries"]["hexagon-v81"]["sha256"]
        ):
            raise ValueError("Measured Native library differs from build")


def functional_checks(
    root: Path, group: str, prompt: int, contexts: list[int] | None = None
) -> dict[str, bool]:
    contexts = BUCKETS[group] if contexts is None else contexts
    reset = read(root / f"generation_{group}_reset.json")
    eos = read(root / f"generation_{group}_eos.json")
    switches = read(root / f"generation_{group}_switches.json")
    sessions = reset["sessions"]
    checks = {
        "reset": len(sessions) == 2
        and len(sessions[0]["generated"]) == 8
        and sessions[0]["generated"] == sessions[1]["generated"]
        and all(s["cached_tokens_at_end"] == prompt + 7 for s in sessions),
        "eos": eos["stop_reason"] == "eos" and eos["generated"][-1] in (151645, 151643),
        "switches": len(switches["generated"]) == 600
        and switches["sessions"][0]["cached_tokens_at_end"] == prompt + 599,
    }
    cached = 0
    for step in switches["steps"]:
        expected = min(
            c for c in contexts if step["ar"] < c and cached <= c - step["ar"]
        )
        checks["switches"] &= (
            step["graph_context"] == expected and step["cached_before"] == cached
        )
        cached += step["new_tokens"]
    return checks


def summarize(
    work: Path,
    bundles: dict[str, dict[str, Any]],
    cl1024_only: bool = False,
    allow_eos_failure: bool = False,
) -> dict[str, Any]:
    root = work / "reports"
    assets = read(work / "assets/assets.json")
    conditions = performance_conditions(cl1024_only)
    policy_path = root / "performance_policy.json"
    policy = read(policy_path) if policy_path.exists() else {"allow_eos_failure": False}
    if policy["allow_eos_failure"] != allow_eos_failure:
        raise ValueError("Performance diagnostic policy changed since measurement")
    if policy_path.exists() and (
        policy["cl1024_only"] != cl1024_only
        or policy["performance_conditions"] != list(conditions)
    ):
        raise ValueError("Performance context policy changed since measurement")
    summary: dict[str, Any] = {
        "model": assets["model"],
        "experiment": read(root / "experiment.json"),
        "performance_sessions_per_configuration_condition": 1,
        "performance_conditions": list(conditions),
        "performance_policy": policy,
        "context_buckets": {g: m["context_buckets"] for g, m in bundles.items()},
        "measurement_notes": (
            "Published W4A16 checkpoint shared by both groups. "
            + (
                "Both use fixed C1024; only 897 prompt + 128 generated tokens are timed. "
                if cl1024_only
                else "Int16 uses fixed C1024; current TurboQuant uses C128/256/512/1024. Long decode is C1024 for both. "
            )
            + "Loading excluded from TTFT; profiling off; no warmup exclusion, thermal control or variance estimate. Functional and PPL runs are separate; no historical model results reused."
            + (
                " EOS-failed groups are diagnostic timings, not quality-valid inference results."
                if allow_eos_failure
                else ""
            )
        ),
        **{condition: {} for condition in conditions},
        "quality": {},
        "functional": {},
    }
    device = None
    for group, metadata in bundles.items():
        paths = [
            root / f"perf_{group}_{condition}_once.json" for condition in conditions
        ]
        paths += [root / f"score_{group}_w{i}.json" for i in range(4)]
        paths += [
            root / f"generation_{group}_{label}.json"
            for label in ("reset", "eos", "switches")
        ]
        for path in paths:
            data = read(path)
            validate_run(data, metadata, assets, group)
            if device is not None and data["device"] != device:
                raise ValueError(f"Device mismatch: {path}")
            device = data["device"]
        for condition, prompt in (("short", assets["prompt_tokens"]), ("long", 897)):
            if condition not in conditions:
                continue
            summary[condition][group] = performance(
                root / f"perf_{group}_{condition}_once.json", prompt
            )
        windows = [read(root / f"score_{group}_w{i}.json") for i in range(4)]
        if any(w["mode"] != "score" or w["scored_tokens"] != 1023 for w in windows):
            raise ValueError("Incorrect PPL scoring window")
        nll = sum(w["nll_sum"] for w in windows)
        summary["quality"][group] = {
            "ppl": math.exp(nll / 4092),
            "nll_sum": nll,
            "scored_tokens": 4092,
        }
        summary["functional"][group] = functional_checks(
            root, group, assets["prompt_tokens"], metadata["context_buckets"]
        )
        checks = summary["functional"][group]
        if not functional_acceptable(checks, allow_eos_failure):
            raise ValueError(
                f"Functional checks failed: {summary['functional'][group]}"
            )
        if "functional" in policy and policy["functional"][group] != checks:
            raise ValueError("Functional results changed since performance measurement")
        for condition in conditions:
            summary[condition][group]["diagnostic_only"] = not all(checks.values())
    summary["device"] = device
    for condition in conditions:
        old, new = (
            summary[condition]["baseline_int16"],
            summary[condition]["turboquant"],
        )
        summary[condition + "_change_percent"] = {
            k: (new[k] / old[k] - 1) * 100
            for k in (
                "ttft_ms",
                "prefill_tok_per_s",
                "decode_tok_per_s",
                "host_kv_MiB",
                "end_VmRSS_MiB",
            )
        }
    return summary


def functional_acceptable(checks: dict[str, bool], allow_eos_failure: bool) -> bool:
    """EOS non-emission may be diagnosed; reset/cache failures never qualify."""
    return {"reset", "eos", "switches"} <= checks.keys() and all(
        passed or (name == "eos" and allow_eos_failure)
        for name, passed in checks.items()
    )


def main() -> None:
    args = parser().parse_args()
    if args.allow_eos_failure and args.stage not in ("performance", "summarize"):
        raise ValueError("--allow-eos-failure is only for performance/summarize")
    buckets = context_buckets(args.cl1024_only)
    conditions = performance_conditions(args.cl1024_only)
    work = args.work_dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    reports = work / "reports"
    reports.mkdir(exist_ok=True)
    prefix = args.device_prefix or work.name
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        raise ValueError(
            "Use a simple --device-prefix containing letters, digits, _, . or -"
        )
    runner = args.runner.expanduser().resolve()
    if args.stage == "prepare":
        execute(
            "split_checkpoint.py",
            [
                "--model-id",
                args.model_id,
                "--checkpoint",
                args.checkpoint,
                "--out",
                str(work / "split"),
            ],
            reports / "prepare_split.stdout.log",
        )
        split = read(work / "split/split_manifest.json")
        execute(
            "run_device_llm.py",
            [
                "assets",
                "--model-id",
                args.model_id,
                "--checkpoint-dir",
                split["resolved_checkpoint"],
                "--out",
                str(work / "assets"),
            ],
            reports / "prepare_assets.stdout.log",
        )
        return
    split = read(work / "split/split_manifest.json")
    if split["model_id"] != args.model_id:
        raise ValueError("--model-id does not match this work directory")
    if args.stage == "build":
        parts = args.parts or list(range(1, len(split["parts"]) + 1))
        if len(set(parts)) != len(parts) or not set(parts) <= set(
            range(1, len(split["parts"]) + 1)
        ):
            raise ValueError("Invalid build parts")
        for group in args.groups:
            for part in parts:
                execute(
                    "convert_parts.py",
                    [
                        "--split-dir",
                        str(work / "split"),
                        "--out",
                        str(work / group),
                        "--profile",
                        GROUPS[group],
                        "--context-length",
                        "1024",
                        "--context-buckets",
                        *map(str, buckets[group]),
                        "--parts",
                        str(part),
                    ],
                    reports / f"build_{group}_part{part}.stdout.log",
                )
        return
    bundles = validate_bundles(work, args.model_id, args.cl1024_only)
    assets = read(work / "assets/assets.json")
    if args.stage == "audit":
        for group, script in (
            ("baseline_int16", "verify_kv_boundary.py"),
            ("turboquant", "verify_rotated_attention.py"),
        ):
            execute(
                script,
                [
                    "--bundle",
                    str(work / group),
                    "--report",
                    str(reports / f"boundary_{group}.json"),
                ],
                reports / f"audit_{group}.stdout.log",
            )
            validate_audit(read(reports / f"boundary_{group}.json"), bundles[group])
        return
    experiment_path = reports / "experiment.json"
    identity = {
        "model": split["model"],
        "split_manifest_sha256": sha256_file(work / "split/split_manifest.json"),
        "runner_sha256": sha256_file(runner),
        "device_prefix": prefix,
        "config_hashes": {g: m["config_hash"] for g, m in bundles.items()},
        "assets_sha256": assets["sha256"],
        "performance_sessions_per_configuration_condition": 1,
    }
    if args.cl1024_only:
        identity["cl1024_only"] = True
    if args.stage == "push":
        for group in GROUPS:
            validate_audit(read(reports / f"boundary_{group}.json"), bundles[group])
        identity["binary_sha256"] = {
            g: {
                p.name: sha256_file(p)
                for p in sorted((work / g).glob("part*_of_*.bin"))
            }
            for g in GROUPS
        }
        with experiment_path.open("x") as output:
            json.dump(identity, output, indent=2)
        for group in GROUPS:
            execute(
                "run_device_llm.py",
                [
                    "push",
                    "--bundle-dir",
                    str(work / group),
                    "--runner",
                    str(runner),
                    "--name",
                    prefix + "_" + group,
                ],
                reports / f"push_{group}.stdout.log",
            )
        return
    experiment = read(experiment_path)
    if experiment.get("cl1024_only", False) != args.cl1024_only or any(
        experiment.get(k) != v for k, v in identity.items()
    ):
        raise ValueError("Experiment identity changed since staging")

    def run(group: str, filename: str, options: list[str]) -> None:
        path = reports / filename
        if path.exists() or path.with_suffix(".log").exists():
            raise FileExistsError(f"Refusing to repeat a measurement: {path}")
        execute(
            "run_device_llm.py",
            [
                "run",
                "--name",
                prefix + "_" + group,
                "--assets",
                str(work / "assets"),
                "--report",
                str(path),
                "--remote-report-tag",
                prefix + "_" + path.stem,
                *options,
            ],
            path.with_suffix(".stdout.log"),
        )
        validate_run(read(path), bundles[group], assets, group)

    if args.stage == "functional":
        for group in GROUPS:
            for label, options in (
                ("reset", ["--n-gen", "8", "--sessions", "2"]),
                ("eos", ["--n-gen", "64", "--stop-on-eos"]),
                ("switches", ["--n-gen", "600"]),
            ):
                run(
                    group,
                    f"generation_{group}_{label}.json",
                    ["--mode", "generate", *options],
                )
            checks = functional_checks(
                reports, group, assets["prompt_tokens"], buckets[group]
            )
            if not all(checks.values()):
                raise ValueError(f"Functional checks failed: {group}: {checks}")
    elif args.stage == "performance":
        checks_by_group = {}
        for group in GROUPS:
            checks = functional_checks(
                reports, group, assets["prompt_tokens"], buckets[group]
            )
            checks_by_group[group] = checks
            if not functional_acceptable(checks, args.allow_eos_failure):
                raise ValueError(f"Functional checks must pass first: {group}")
            if not all(checks.values()):
                print(f"DIAGNOSTIC ONLY: {group}: {checks}", flush=True)
        policy_path = reports / "performance_policy.json"
        if policy_path.exists():
            raise FileExistsError(f"Refusing to repeat a measurement: {policy_path}")
        for condition in conditions:
            for group in GROUPS:
                stem = f"perf_{group}_{condition}_once"
                for suffix in (".json", ".log", ".stdout.log"):
                    path = reports / (stem + suffix)
                    if path.exists():
                        raise FileExistsError(
                            f"Refusing to repeat a measurement: {path}"
                        )
        with policy_path.open("x") as output:
            json.dump(
                {
                    "allow_eos_failure": args.allow_eos_failure,
                    "functional": checks_by_group,
                    "cl1024_only": args.cl1024_only,
                    "performance_conditions": list(conditions),
                    "quality_measured_before_performance": all(
                        (reports / f"score_{group}_w{i}.json").exists()
                        for group in GROUPS
                        for i in range(4)
                    ),
                },
                output,
                indent=2,
            )
        for condition, tokens in (
            ("short", assets["prompt_ids"]),
            ("long", assets["boundary_prompt"]),
        ):
            if condition not in conditions:
                continue
            for group in GROUPS:
                run(
                    group,
                    f"perf_{group}_{condition}_once.json",
                    [
                        "--mode",
                        "generate",
                        "--n-gen",
                        "128",
                        "--sessions",
                        "1",
                        "--tokens",
                        tokens,
                    ],
                )
    elif args.stage == "quality":
        for group in GROUPS:
            for window in range(4):
                run(
                    group,
                    f"score_{group}_w{window}.json",
                    ["--mode", "score", "--tokens", assets["wikitext_windows"][window]],
                )
    else:
        result = summarize(work, bundles, args.cl1024_only, args.allow_eos_failure)
        with (reports / "comparison.json").open("x") as output:
            json.dump(result, output, indent=2)
        print(json.dumps({k: result[k] for k in (*conditions, "quality")}, indent=2))


if __name__ == "__main__":
    main()
