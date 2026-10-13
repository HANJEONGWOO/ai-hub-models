# SPDX-License-Identifier: BSD-3-Clause
"""Single-run boundary V protection; immutable legacy artifacts and input hashes.

prepare -> build (new profiles only) -> evaluate -> summarize. Device runs are
serial; exclusive log/report paths refuse accidental repeated measurements.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import onnx
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
    "k4_v2",
    "boundary_v8",
    "middle_v8",
)


PROFILE_NAMES = {
    "k4_v4": "k4_v4_scaled",
    "k4_v2": "k4_v2_scaled",
    "boundary_v8": "k4_v8_boundary",
    "middle_v8": "k4_v8_middle",
}


def config_for(root: Path, group: str) -> TurboQuantConfig:
    return get_profile(PROFILE_NAMES[group])


def expected_cache_bytes(c: TurboQuantConfig) -> int:
    # Every layer: 8 KV heads, 1024 tokens, 128 coordinates, two FP16 scales.
    return (
        8
        * 1024
        * sum(
            16 * (c.codec_for("key", i).bits + c.codec_for("value", i).bits) + 4
            for i in range(28)
        )
    )


def prepare(root: Path) -> None:
    assets = ARTIFACTS / "device_assets_cl1024"
    runner = ARTIFACTS / "asymmetric_kv_20261010/runner/qnn-llm-runner"
    split = ARTIFACTS / "qwen3_1_7b_w4a16_split"
    source = ARTIFACTS / "asymmetric_kv_20261010/k4_v4"
    manifest = read(assets / "assets.json")
    if manifest["wikitext_windows"] != [f"wikitext_w{i}.bin" for i in range(4)]:
        raise ValueError("Wrong legacy PPL windows")
    frozen = {str(assets / n): h for n, h in manifest["sha256"].items()}
    native = read(root / "native/manifest.json")
    paths = [
        runner,
        root / "native/manifest.json",
        split / "split_manifest.json",
        source / "convert_report.json",
        *source.glob("part*_of_4.bin"),
    ]
    paths.extend(Path(v["path"]) for v in native["libraries"].values())
    frozen.update({str(p): digest(p) for p in paths})
    baseline = read(source / "convert_report.json")
    if baseline["config_hash"] != config_for(root, "k4_v4").config_hash():
        raise ValueError("Baseline changed")
    target = root / "k4_v4"
    target.mkdir(exist_ok=False)
    for p in source.iterdir():
        if p.is_file() and (
            p.name.startswith(("prompt_", "token_", "part")) or p.suffix == ".data"
        ):
            (target / p.name).symlink_to(p.resolve())
    save_new(target / "convert_report.json", baseline)
    configs = {}
    for group in GROUPS:
        c = config_for(root, group)
        configs[group] = {
            "profile": c.profile,
            "config": c.to_dict(),
            "config_hash": c.config_hash(),
            "v8_layers": [i for i in range(28) if c.codec_for("value", i).bits == 8],
            "expected_cache_bytes": expected_cache_bytes(c),
        }
        if any(c.codec_for("key", i).bits != 4 for i in range(28)):
            raise ValueError("Only V precision may change")
    if (
        configs["boundary_v8"]["expected_cache_bytes"]
        != configs["middle_v8"]["expected_cache_bytes"]
    ):
        raise ValueError("Boundary and Middle budgets differ")
    for path, checksum in frozen.items():
        if digest(Path(path)) != checksum:
            raise ValueError(f"Changed input: {path}")
    save_new(
        root / "protocol.json",
        {
            "model": "Qwen3-1.7B W4A16",
            "context_length": 1024,
            "branch": "exp/boundary-v-quantization",
            "base_commit": "b751124c3",
            "reused_8bit_implementation_commit": "c022fecee",
            "group_order": GROUPS,
            "profiles": configs,
            "frozen_files": frozen,
            "assets": str(assets),
            "assets_manifest": manifest,
            "runner": str(runner),
            "split": str(split),
            "native_package": native,
            "performance": {
                "sessions_per_group_condition": 1,
                "profiling": False,
                "conditions": {"short": [35, 128], "long": [897, 128]},
                "dispersion": "not estimated from one run; user overrode three repetitions",
            },
            "quality": {
                "windows": 4,
                "scored_tokens_per_window": 1023,
                "aggregation": "exp(sum(nll_sum) / sum(scored_tokens))",
                "runs_per_window": 1,
            },
            "diagnostics": "Separate two-session eight-token reset and physical HTP optrace; not throughput samples",
            "strict_encoder_gate": "Unchanged HTP_FP16 tolerances including scale_relative=0.002; failures reported separately, not waived",
            "fixed": [
                "K4 all layers",
                "Dense K42/V542",
                "exact LM tree",
                "frozen Lloyd-Max",
                "Native HVX",
                "current/past KV compressed",
                "FP16 corrected effective scale",
                "same W4A16, weights, calibration, GQA, masks, QK/AV",
            ],
            "scope": "Only the four fixed groups; no training, placement search, or 4+4/6+6 follow-up execution",
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
    return p


def runtime_override(group: str, protocol: dict) -> None:
    # Baseline uses its unchanged package; new graphs compile against new Native.
    return None


def audit_key_path(root: Path, group: str) -> None:
    """Bit-identical K source nodes/constants and Q rotation; V alone may change."""
    checks = {}
    for path in sorted((root / group).glob("*.kv_edits.json")):
        stem = path.name.removesuffix(".kv_edits.json")
        graphs = [
            onnx.load(root / g / (stem + ".onnx"), load_external_data=False).graph
            for g in ("k4_v4", group)
        ]
        nodes, constants = [], []
        for graph in graphs:
            nodes.append(
                {
                    n.name: n.SerializeToString()
                    for n in graph.node
                    if n.name.startswith("tq_key_")
                    or re.fullmatch(
                        r"tq_attn_\d+_head\d+_q\d+_(rotated|scaled)", n.name
                    )
                }
            )
            constants.append(
                {
                    t.name: t.SerializeToString()
                    for t in graph.initializer
                    if t.name.startswith(("tq_key_", "tq_rotation_"))
                }
            )
        checks[stem] = {
            "key_and_query_nodes_unchanged": bool(nodes[0]) and nodes[0] == nodes[1],
            "key_constants_and_all_rotations_unchanged": bool(constants[0])
            and constants[0] == constants[1],
        }
    passed = bool(checks) and all(all(v.values()) for v in checks.values())
    path = root / "reports" / f"k_path_audit_{group}.json"
    record = {"graphs": checks, "passed": passed}
    if path.exists():
        if read(path) != record:
            raise ValueError("Previously audited K path changed")
    else:
        save_new(path, record)
    if not passed:
        raise ValueError("K or rotation changed outside the V-only experiment")


def measure(
    root: Path, group: str, label: str, options: list[str], protocol: dict
) -> dict:
    path = root / "reports" / f"{group}_{label}.json"
    if path.exists():
        raise FileExistsError(f"Refusing repeated measurement {path}")

    def environment() -> dict:
        battery = subprocess.check_output(
            ["/mnt/c/adb/adb.exe", "shell", "dumpsys", "battery"], text=True
        )
        temperature = re.search(r"^\s*temperature:\s*(\d+)", battery, re.MULTILINE)
        return {
            "utc": datetime.now(timezone.utc).isoformat(),
            "battery_temperature_tenths_c": int(temperature.group(1))
            if temperature
            else None,
        }

    before = environment()
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
    save_new(
        path.with_suffix(".environment.json"),
        {"before": before, "after": environment()},
    )
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
    if data["kv_store_bytes"] != protocol["profiles"][group]["expected_cache_bytes"]:
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
    audit_key_path(root, group)
    if group != "k4_v4" and meta["native_decoder"] != protocol["native_package"]:
        raise ValueError("New bundle uses a different Native package")
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
    a, v2, boundary, middle = (result[g]["quality"] for g in GROUPS)
    gap = v2["mean_nll"] - a["mean_nll"]
    analysis = {
        "v2_minus_v4_mean_nll": gap,
        "boundary_minus_middle_mean_nll": boundary["mean_nll"] - middle["mean_nll"],
        "boundary_vs_middle_window_delta_nll": [
            (b["nll_sum"] - m["nll_sum"]) / 1023
            for b, m in zip(boundary["windows"], middle["windows"], strict=True)
        ],
        "vs_v2": {},
        "caveats": "Recovery is descriptive relative to K4/V4, not FP16 ground truth; four reused windows, no new heldout; performance is one run per condition, not a significance estimate",
    }
    for group in GROUPS[2:]:
        q = result[group]["quality"]
        analysis["vs_v2"][group] = {
            "ppl_drop": v2["ppl"] - q["ppl"],
            "ppl_drop_percent": 100 * (1 - q["ppl"] / v2["ppl"]),
            "nll_gap_recovered_fraction": (v2["mean_nll"] - q["mean_nll"]) / gap
            if gap
            else None,
            "cache_increase_bytes": protocol["profiles"][group]["expected_cache_bytes"]
            - protocol["profiles"]["k4_v2"]["expected_cache_bytes"],
            "long_ttft_increase_percent": 100
            * (
                result[group]["performance"]["long"]["ttft_ms"]
                / result["k4_v2"]["performance"]["long"]["ttft_ms"]
                - 1
            ),
            "long_decode_latency_increase_percent": 100
            * (
                result[group]["performance"]["long"]["decode_ms_per_token"]
                / result["k4_v2"]["performance"]["long"]["decode_ms_per_token"]
                - 1
            ),
        }
    save_new(
        root / "comparison.json",
        {
            "sessions_per_condition": 1,
            "dispersion": None,
            "groups": result,
            "protocol": str(root / "protocol.json"),
            "analysis": analysis,
        },
    )
    with (root / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    repo = Path(__file__).resolve().parents[3]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=repo, text=True
    ).splitlines()
    new = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, text=True
    ).splitlines()
    save_new(
        root / "source_provenance.json",
        {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "git_branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=repo, text=True
            ).strip(),
            "files_at_summary": {
                p: digest(repo / p)
                for p in sorted(set(changed + new))
                if (repo / p).is_file()
            },
            "notes": "Uncommitted worktree; build ONNX/DLC/context and runtime hashes are retained in their respective manifests; final README may be completed after summary",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "build", "evaluate", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--build-jobs", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        prepare(root)
        return
    protocol = validate(root)
    if args.stage == "summarize":
        summarize(root, protocol)
        return

    def build(group: str) -> None:
        if group != "k4_v4":
            execute(
                "convert_parts.py",
                [
                    "--split-dir",
                    protocol["split"],
                    "--out",
                    str(root / group),
                    "--profile",
                    PROFILE_NAMES[group],
                    "--context-length",
                    "1024",
                    "--sequence-lengths",
                    "1",
                    "128",
                    "--native-decoder-package",
                    str(root / "native"),
                ],
                root / "logs" / f"build_{group}.log",
            )

    if args.stage == "build":
        with ThreadPoolExecutor(max_workers=args.build_jobs) as workers:
            list(workers.map(build, args.groups))
    else:
        for group in args.groups:
            evaluate(root, group, protocol)


if __name__ == "__main__":
    main()
