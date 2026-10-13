# SPDX-License-Identifier: BSD-3-Clause
"""Single-run mixed-bit comparison; immutable legacy artifacts and input hashes.

prepare -> build (new profiles only) -> evaluate -> summarize. Device runs are
serial; exclusive log/report paths refuse accidental repeated measurements.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from benchmark_asymmetric_hvx_once import package_check
from benchmark_asymmetric_once import (
    ARTIFACTS,
    audit_source,
    digest,
    execute,
    read,
    save_new,
)
from summarize_native_results import performance

from qai_hub_models.models.templates.llm.turboquant.config import (
    TurboQuantConfig,
    get_profile,
)

GROUPS = (
    "k4_v4",
    "k5_v3",
    "k6_v2",
    "mixed_k5_v3_uniform",
    "mixed_k6_v2_uniform",
    "mixed_k5_v3_mse",
    "mixed_k6_v2_mse",
)


def config_for(root: Path, group: str) -> TurboQuantConfig:
    return get_profile(
        group if group.startswith("mixed_") else group + "_scaled",
        layer_plan=read(root / "selection" / f"{group}.json")
        if group.endswith("_mse")
        else None,
    )


def prepare(root: Path) -> None:
    assets = ARTIFACTS / "device_assets_cl1024"
    runner = ARTIFACTS / "asymmetric_kv_20261010/runner/qnn-llm-runner"
    manifest = read(assets / "assets.json")
    if manifest["wikitext_windows"] != [f"wikitext_w{i}.bin" for i in range(4)]:
        raise ValueError("Wrong legacy PPL windows")
    frozen = {str(assets / n): h for n, h in manifest["sha256"].items()}
    for path in (
        runner,
        root / "native/manifest.json",
        root / "selection/selection_protocol.json",
        root / "selection/selection_results.json",
    ):
        frozen[str(path)] = digest(path)
    for group in GROUPS[:3]:
        source = ARTIFACTS / "asymmetric_kv_20261010" / group
        target = root / group
        target.mkdir(exist_ok=False)
        for p in source.iterdir():
            if p.is_file() and (
                p.name.startswith(("prompt_", "token_", "part")) or p.suffix == ".data"
            ):
                (target / p.name).symlink_to(p.resolve())
        meta = read(source / "convert_report.json")
        if meta["config_hash"] != config_for(root, group).config_hash():
            raise ValueError("Baseline configuration changed")
        save_new(target / "convert_report.json", meta)
        for path in (source / "convert_report.json", *source.glob("part*_of_4.bin")):
            frozen[str(path)] = digest(path)
    configs = {}
    for group in GROUPS:
        c = config_for(root, group)
        configs[group] = {
            "profile": c.profile,
            "config": c.to_dict(),
            "config_hash": c.config_hash(),
        }
        if c.layer_bits and sum(sum(p) for p in c.layer_bits) != 28 * 8:
            raise ValueError("Mixed payload budget is not exactly 8 bits")
        if group.endswith("_mse"):
            frozen[str(root / "selection" / f"{group}.json")] = digest(
                root / "selection" / f"{group}.json"
            )
    for path, checksum in frozen.items():
        if digest(Path(path)) != checksum:
            raise ValueError(f"Changed frozen input: {path}")
    save_new(
        root / "protocol.json",
        {
            "model": "Qwen3-1.7B W4A16",
            "context_length": 1024,
            "group_order": GROUPS,
            "profiles": configs,
            "frozen_files": frozen,
            "assets": str(assets),
            "assets_manifest": manifest,
            "runner": str(runner),
            "split": str(ARTIFACTS / "qwen3_1_7b_w4a16_split"),
            "legacy_native_override": str(ARTIFACTS / "asymmetric_hvx_20261010/native"),
            "performance": {
                "sessions_per_group_condition": 1,
                "profiling": False,
                "conditions": {"short": [35, 128], "long": [897, 128]},
                "dispersion": "not estimated from a single run",
            },
            "quality": {
                "windows": 4,
                "scored_tokens_per_window": 1023,
                "aggregation": "exp(sum(nll_sum) / sum(scored_tokens))",
                "runs_per_window": 1,
            },
            "diagnostics": "Separate reset check (two sessions, eight tokens) and HTP stage traces; never reported as throughput samples",
            "strict_encoder_gate": "Known AR128 FP16 effective-scale tolerance failures remain reported, not waived or hidden; zero unexplained index mismatches required",
            "fixed": [
                "Dense K42/V542",
                "exact LM tree",
                "frozen Lloyd-Max",
                "Native HVX",
                "current and past KV compressed",
                "FP16 norm-corrected effective scale",
                "same W4A16 and calibration",
            ],
        },
    )


def validate(root: Path) -> dict:
    p = read(root / "protocol.json")
    for path, checksum in p["frozen_files"].items():
        if digest(Path(path)) != checksum:
            raise ValueError(f"Changed frozen input: {path}")
    for group in GROUPS:
        if config_for(root, group).config_hash() != p["profiles"][group]["config_hash"]:
            raise ValueError("Configuration changed after freeze")
    amendment = root / "native_runtime_amendment.json"
    if amendment.exists():
        update = read(amendment)
        for path, checksum in update["frozen_files"].items():
            if digest(Path(path)) != checksum:
                raise ValueError(f"Changed Native runtime amendment input: {path}")
        package_check(
            read(root / "native/manifest.json"),
            read(Path(update["package"]) / "manifest.json"),
        )
        p["mixed_native_override"] = update["package"]
    return p


def amend_runtime(root: Path, package: Path) -> None:
    """Freeze a header-only HVX improvement before any mixed measurement."""
    validate(root)
    if any((root / "reports").glob("mixed_*_perf_*.json")):
        raise ValueError("Cannot change the runtime after mixed performance evaluation")
    compiled = read(root / "native/manifest.json")
    runtime = read(package / "manifest.json")
    checks = package_check(compiled, runtime)
    paths = [root / "protocol.json", package / "manifest.json"]
    paths.extend(Path(v["path"]) for v in runtime["libraries"].values())
    save_new(
        root / "native_runtime_amendment.json",
        {
            "package": str(package.resolve()),
            "groups": list(GROUPS[3:]),
            "reason": "Exact symmetric 8-bit LM lookup uses the positive half LUT and restores the FP16 sign; original package and diagnostics are preserved",
            "contract_checks": checks,
            "frozen_files": {str(p): digest(p) for p in paths},
            "unchanged": "Encoder, codebooks, graphs, scales, payload, all 2-7-bit kernels and existing baselines",
        },
    )


def runtime_override(group: str, protocol: dict) -> str | None:
    if group in ("k5_v3", "k6_v2"):
        return protocol["legacy_native_override"]
    if group.startswith("mixed_"):
        return protocol.get("mixed_native_override")
    return None


def measure(
    root: Path, group: str, label: str, options: list[str], protocol: dict
) -> dict:
    path = root / "reports" / f"{group}_{label}.json"
    if path.exists():
        raise FileExistsError(f"Refusing repeated measurement {path}")
    execute(
        "run_device_llm.py",
        [
            "run",
            "--name",
            root.name + "_" + group,
            "--assets",
            protocol["assets"],
            "--report",
            str(path),
            "--remote-report-tag",
            root.name + "_" + group + "_" + label,
            *options,
        ],
        path.with_suffix(".stdout.log"),
    )
    data = read(path)
    if (
        data["config_hash"] != protocol["profiles"][group]["config_hash"]
        or not data["quantize_current_kv"]
        or data.get("op_profiles")
    ):
        raise ValueError("Measurement configuration mismatch")
    if (
        data["assets"]["tokens_sha256"]
        != protocol["assets_manifest"]["sha256"][data["assets"]["tokens_file"]]
    ):
        raise ValueError("Input token mismatch")
    if data["kv_store_bytes"] != int(28.875 * 2**20):
        raise ValueError("Unexpected actual cache size")
    return data


def evaluate(root: Path, group: str, protocol: dict) -> None:
    bundle = root / group
    meta = read(bundle / "convert_report.json")
    if (
        meta["config_hash"] != protocol["profiles"][group]["config_hash"]
        or set(meta["parts"]) != {f"part{i}_of_4" for i in range(1, 5)}
        or meta["context_buckets"] != [1024]
        or not meta["quantize_current_kv"]
        or not meta["rotated_attention"]
        or meta["attention_tile"] != 256
    ):
        raise ValueError("Incomplete/incompatible compiled model")
    execute(
        "verify_rotated_attention.py",
        [
            "--bundle",
            str(bundle),
            "--report",
            str(root / "reports" / f"audit_{group}.json"),
        ],
        root / "logs" / f"audit_{group}.log",
    )
    audit_source(root, group)
    runtime = runtime_override(group, protocol)
    override = ["--native-runtime-package", runtime] if runtime else []
    execute(
        "run_device_llm.py",
        [
            "push",
            "--bundle-dir",
            str(bundle),
            "--runner",
            protocol["runner"],
            "--name",
            root.name + "_" + group,
            *override,
        ],
        root / "logs" / f"push_{group}.log",
    )
    data = measure(
        root,
        group,
        "reset",
        ["--mode", "generate", "--n-gen", "8", "--sessions", "2"],
        protocol,
    )
    if (
        len(data["sessions"]) != 2
        or data["sessions"][0]["generated"] != data["sessions"][1]["generated"]
    ):
        raise ValueError("Non-deterministic cache reset")
    for condition, tokens in (
        ("short", "prompt_ids.bin"),
        ("long", "boundary_prompt_cl1024.bin"),
    ):
        measure(
            root,
            group,
            f"perf_{condition}",
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
            protocol,
        )
    for i in range(4):
        measure(
            root,
            group,
            f"score_w{i}",
            ["--mode", "score", "--tokens", f"wikitext_w{i}.bin"],
            protocol,
        )


def summarize(root: Path, protocol: dict) -> None:
    result, rows = {}, []
    for group in GROUPS:
        quality = [
            read(root / "reports" / f"{group}_score_w{i}.json") for i in range(4)
        ]
        if any(
            d["scored_tokens"] != 1023
            or d["config_hash"] != protocol["profiles"][group]["config_hash"]
            for d in quality
        ):
            raise ValueError("Quality evaluation mismatch")
        nll = sum(d["nll_sum"] for d in quality)
        perf = {
            c: performance(root / "reports" / f"{group}_perf_{c}.json", prompt)
            for c, prompt in (("short", 35), ("long", 897))
        }
        result[group] = {
            "config": protocol["profiles"][group],
            "performance": perf,
            "quality": {
                "ppl": math.exp(nll / 4092),
                "mean_nll": nll / 4092,
                "nll_sum": nll,
                "scored_tokens": 4092,
                "windows": [
                    {k: d[k] for k in ("nll_sum", "scored_tokens", "ppl", "assets")}
                    for d in quality
                ],
            },
            "binary_bytes": sum(
                p.stat().st_size for p in (root / group).glob("part*_of_4.bin")
            ),
            "bins_sha256": {
                p.name: digest(p) for p in (root / group).glob("part*_of_4.bin")
            },
        }
        for condition, values in perf.items():
            rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "ppl": result[group]["quality"]["ppl"],
                    **{
                        k: values[k]
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
            "groups": result,
            "offline_attention_mse": read(root / "selection/selection_results.json")[
                "attention_mse"
            ],
        },
    )
    with (root / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "amend-runtime", "build", "evaluate", "summarize")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--native-runtime-package", type=Path)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        prepare(root)
        return
    if args.stage == "amend-runtime":
        if args.native_runtime_package is None:
            parser.error("amend-runtime requires --native-runtime-package")
        amend_runtime(root, args.native_runtime_package.resolve())
        return
    protocol = validate(root)
    if args.stage == "summarize":
        summarize(root, protocol)
        return
    for group in args.groups:
        if args.stage == "evaluate":
            evaluate(root, group, protocol)
        elif group.startswith("mixed_"):
            plan = (
                ["--layer-plan", str(root / "selection" / f"{group}.json")]
                if group.endswith("_mse")
                else []
            )
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
                    str(root / "native"),
                    *plan,
                ],
                root / "logs" / f"build_{group}.log",
            )


if __name__ == "__main__":
    main()
