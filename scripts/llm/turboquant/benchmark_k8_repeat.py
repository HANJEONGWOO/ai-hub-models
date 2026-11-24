# SPDX-License-Identifier: BSD-3-Clause
"""Frozen Odd/Boundary Structured vs converged-LM8 repeated HTP experiment.

Preparation/build/deployment finish before freeze. No codec or runner changes,
automatic retries, sample filtering, PPL evaluation, or automatic commits.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TextIO

from benchmark_asymmetric_once import ARTIFACTS, digest, execute, read, save_new
from benchmark_nll_allocation import symlink

from qai_hub_models.models.templates.llm.turboquant.config import (
    STRUCTURED_ALLOCATION_PROFILE,
    TurboQuantConfig,
    get_profile,
    structured_allocation_plan,
)

PREVIOUS = ARTIFACTS / "lloyd_max_k8_convergence_20261020"
BOUNDARY_SOURCE = ARTIFACTS / "structured_k8_nll_allocation_20261013/boundary"
NATIVE = ARTIFACTS / "structured_k8_decoder_20261013/native"
RUNNER = ARTIFACTS / "structured_k8_nll_allocation_20261013/runner/qnn-llm-runner"
SPLIT = ARTIFACTS / "qwen3_1_7b_w4a16_split"
ASSETS = ARTIFACTS / "device_assets_cl1024"
SDK = Path("/home/hjw/qairt/2.48.0.260626")
ADB = "/mnt/c/adb/adb.exe"
SERIAL = "R5KYB0654BW"
GROUPS = ("odd_structured", "odd_lm", "boundary_structured", "boundary_lm")
MASKS = {"odd": list(range(1, 28, 2)), "boundary": list(range(7)) + list(range(21, 28))}
METRICS = ("ttft_ms", "prefill_tok_per_s", "decode_tok_per_s")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def schedule() -> list[list[str]]:
    # Two Williams blocks: each configuration occurs twice in each position and
    # each directed within-round adjacency occurs twice. Two reverse orders add
    # one observation per config in complementary positions. Every pair has
    # exactly 5/5 precedence. Exact position balance is impossible with n=10.
    rows = [(0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1)] * 2
    rows += [(0, 2, 1, 3), (3, 1, 2, 0)]
    return [[GROUPS[i] for i in row] for row in rows]


def config(group: str) -> TurboQuantConfig:
    if group == "boundary_structured":
        return get_profile(
            STRUCTURED_ALLOCATION_PROFILE,
            layer_plan=structured_allocation_plan(MASKS["boundary"]),
        )
    return get_profile(
        {
            "odd_structured": "mixed_k6_v2_uniform_structured_k8",
            "odd_lm": "mixed_k6_v2_uniform_lm8_converged",
            "boundary_lm": "mixed_k6_v2_boundary_lm8_converged",
        }[group]
    )


def stage_bundle(source: Path, target: Path) -> None:
    target.mkdir(exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and (
            path.name.startswith(("part", "prompt_", "token_"))
            or path.suffix == ".data"
        ):
            symlink(path, target / path.name)
    save_new(target / "convert_report.json", read(source / "convert_report.json"))


def prepare(root: Path) -> None:
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    sources = {
        "odd_structured": PREVIOUS / "structured",
        "odd_lm": PREVIOUS / "lm8_converged",
        "boundary_structured": BOUNDARY_SOURCE,
    }
    for group, source in sources.items():
        if (
            read(source / "convert_report.json")["config_hash"]
            != config(group).config_hash()
        ):
            raise ValueError(f"Historical config mismatch: {group}")
        stage_bundle(source, root / group)
    paths = [
        RUNNER,
        NATIVE / "manifest.json",
        SPLIT / "split_manifest.json",
        ASSETS / "assets.json",
    ]
    paths += [
        Path(v["path"]) for v in read(NATIVE / "manifest.json")["libraries"].values()
    ]
    paths += [ASSETS / name for name in read(ASSETS / "assets.json")["sha256"]]
    paths += [source / "convert_report.json" for source in sources.values()]
    paths += [p for source in sources.values() for p in source.glob("part*_of_4.bin")]
    pdf = Path("ICEIC2027_Overleaf_Ready.pdf").resolve()
    if pdf.exists():
        paths.append(pdf)
    save_new(
        root / "protocol.json",
        {
            "created_utc": now(),
            "groups": GROUPS,
            "masks": MASKS,
            "schedule": schedule(),
            "model": "Qwen3-1.7B W4A16",
            "sdk": str(SDK),
            "device_serial": SERIAL,
            "context_length": 1024,
            "prompt_tokens": 897,
            "generated_tokens": 128,
            "decode_iterations": 127,
            "maximum_cached_tokens": 897 + 128 - 1,
            "kv_store_bytes": 30277632,
            "runner": str(RUNNER),
            "assets": str(ASSETS),
            "native": str(NATIVE),
            "split": str(SPLIT),
            "source_bundles": {k: str(v) for k, v in sources.items()},
            "profiles": {
                g: {
                    "config": config(g).to_dict(),
                    "config_hash": config(g).config_hash(),
                }
                for g in GROUPS
            },
            "warmup": "Every invocation uses existing --sessions 2: session 0 is a full 897+128 warm-up; reset; only session 1 is measured. 40 warm-up + 40 measured sessions.",
            "timing": "Uninstrumented HTP burst-power runner; model load/reset/warm-up excluded. Prefill=897/prefill_s; decode=127/sum(127 decode_step_s); first generated token is in prefill.",
            "order": "10 rounds; each pair precedes the other 5 times; position counts 2 or 3. Schedule fixed before any new timing.",
            "inter_invocation_pause_s": 10,
            "thermal": "HAL current temperatures/status every 2 seconds during runner invocation, plus before/after; battery and device clock before/after. Cached HAL temperatures are not used. Android status is not direct HTP clock/throttling telemetry.",
            "abnormal_policy": "Retain every attempt and all finite metrics, including thermal warnings/outliers. No speed-based deletion/retry. Stop on execution failure or protocol mismatch; no automatic replacement.",
            "analysis": "Mean, sample SD(ddof=1), median/min/max; improvement from arithmetic group means. Paired rounds/temperature/order exploratory, not causal or multi-device inference.",
            "frozen_inputs": {str(p): digest(p) for p in paths},
            "fixed": "Dense K42/V542; existing codebooks, norm correction/FP16 scales, current+past compressed, tile256, weights/calibration, K4/V2; no performance-code change.",
            "historical_encoder_gate": "Existing AR128 encoder tolerance failures remain separate limitations; this is not a new numerical certification.",
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )
    print("Prepared", root, flush=True)


def check_inputs(root: Path) -> dict:
    p = read(root / "protocol.json")
    for name, expected in p["frozen_inputs"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for group in GROUPS:
        if config(group).config_hash() != p["profiles"][group]["config_hash"]:
            raise ValueError(f"Config changed: {group}")
    return p


def build(root: Path) -> None:
    check_inputs(root)
    if (root / "measurement_freeze.json").exists():
        raise ValueError("No build after measurement freeze")

    def part(number: int) -> None:
        execute(
            "convert_parts.py",
            [
                "--split-dir",
                str(SPLIT),
                "--out",
                str(root / "builds" / f"part{number}"),
                "--profile",
                config("boundary_lm").profile,
                "--context-length",
                "1024",
                "--sequence-lengths",
                "1",
                "128",
                "--native-decoder-package",
                str(NATIVE),
                "--parts",
                str(number),
            ],
            root / "logs" / f"build_boundary_lm_part{number}.log",
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(part, (2, 3, 4)))
    assemble(root)


def assemble(root: Path) -> None:
    target = root / "boundary_lm"
    target.mkdir(exist_ok=False)
    combined = copy.deepcopy(read(root / "builds/part2/convert_report.json"))
    combined["parts"] = {}
    for number in (1, 2, 3, 4):
        source = root / "odd_lm" if number == 1 else root / "builds" / f"part{number}"
        meta = read(source / "convert_report.json")
        name = f"part{number}_of_4"
        if number > 1 and meta["config_hash"] != config("boundary_lm").config_hash():
            raise ValueError("Unexpected build profile")
        if number == 1 and any(
            g.get("surgery", {}).get("codec_io")
            for g in meta["parts"][name]["graphs"].values()
        ):
            raise ValueError("Shared part1 must not contain KV codec")
        combined["parts"][name] = meta["parts"][name]
        symlink(source / f"{name}.bin", target / f"{name}.bin")
        for graph in meta["parts"][name]["graphs"]:
            for path in source.glob(graph + ".*"):
                if path.is_file():
                    symlink(path, target / path.name)
        for path in source.glob("*.data"):
            symlink(path, target / path.name)
    save_new(target / "convert_report.json", combined)


def audit(root: Path) -> None:
    """Check the compiled I/O/attention plus frozen encoder/decoder constants."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    from qai_hub_models.models.templates.llm.turboquant.lloyd_max_k8 import codebook

    check_inputs(root)
    native = read(NATIVE / "manifest.json")
    results = {}
    for group in GROUPS:
        bundle = root / group
        meta = read(bundle / "convert_report.json")
        c = config(group)
        checks = {
            "config": meta["config"] == c.to_dict()
            and meta["config_hash"] == c.config_hash(),
            "native": meta["native_decoder"] == native,
            "sdk": meta["sdk"] == str(SDK),
            "context": meta["context_buckets"] == [1024],
            "current_past": meta["quantize_current_kv"] is True,
            "attention": meta["attention_tile"] == 256 and meta["rotated_attention"],
            "mask": [i for i, p in enumerate(c.layer_bits) if p[0] == 8]
            == MASKS[group.split("_")[0]],
            "budget": sum(sum(p) for p in c.layer_bits) == 28 * 8
            and all(p[1] == 2 for p in c.layer_bits),
            "decoder": meta.get("structured_k8_decoder", "lut")
            == ("arithmetic" if group.endswith("structured") else "lut"),
            "parts": set(meta["parts"]) == {f"part{i}_of_4" for i in range(1, 5)},
        }
        execute(
            "verify_rotated_attention.py",
            [
                "--bundle",
                str(bundle),
                "--report",
                str(root / "audits" / f"{group}.json"),
            ],
            root / "logs" / f"audit_{group}.log",
        )
        codebooks = {}
        for path in sorted(bundle.glob("*.kv_edits.json")):
            name = path.name.removesuffix(".kv_edits.json")
            graph = onnx.load(bundle / (name + ".onnx"), load_external_data=False).graph
            if group.endswith("lm"):
                arrays = {t.name: t for t in graph.initializer}
                # A K4-only split needs no K8 constant.
                has_k8 = any(
                    io["kind"] == "key" and io["packed_bytes"] == 128
                    for io in read(path)["codec_io"]
                )
                if has_k8:
                    lut = numpy_helper.to_array(
                        arrays["tq_native_converged_k8_centroids_fp16"]
                    )
                    expected = codebook()
                    codebooks[name] = {
                        "lut_exact": np.array_equal(
                            lut.ravel(), expected.astype(np.float16)
                        ),
                        "boundaries_exact": all(
                            float(
                                numpy_helper.to_array(
                                    arrays[f"tq_scalar_b8_d128_threshold_{i}"]
                                )
                            )
                            == float(x)
                            for i, x in enumerate(
                                ((expected[:-1] + expected[1:]) / 2).astype(np.float32)
                            )
                        ),
                    }
        checks["converged_tables"] = all(all(v.values()) for v in codebooks.values())
        results[group] = {"checks": checks, "codebooks": codebooks}
        if not all(checks.values()):
            save_new(root / "audits" / f"failed_{group}.json", results[group])
            raise ValueError(f"Failed bundle audit: {group}")
    # Each pair starts from precisely the same external model weights/encodings.
    pairs = {}
    for placement in MASKS:
        left, right = [root / (placement + suffix) for suffix in ("_structured", "_lm")]
        checks = {}
        for p in sorted(left.glob("*.kv_edits.json")):
            name = p.name.removesuffix(".kv_edits.json")
            graphs = [
                onnx.load(b / (name + ".onnx"), load_external_data=False).graph
                for b in (left, right)
            ]
            unchanged = [
                {
                    t.name: t.SerializeToString()
                    for t in g.initializer
                    if not t.name.startswith("tq_")
                }
                for g in graphs
            ]
            checks[name] = {
                "non_codec_initializers_identical": unchanged[0] == unchanged[1],
                "encodings_identical": digest(left / (name + ".encodings"))
                == digest(right / (name + ".encodings")),
            }
        pairs[placement] = checks
    passed = all(all(c.values()) for p in pairs.values() for c in p.values())
    save_new(
        root / "audits/summary.json",
        {"groups": results, "pairs": pairs, "passed": passed},
    )
    if not passed:
        raise ValueError("Paired weight/encoding audit failed")


def adb(command: str, timeout: int = 20) -> dict:
    start = now()
    result = subprocess.run(
        [ADB, "-s", SERIAL, "shell", command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "host_start_utc": start,
        "host_end_utc": now(),
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def parse_thermal(text: str) -> dict:
    status = re.search(r"^Thermal Status:\s*(\d+)", text, re.MULTILINE)
    current = text.split("Current temperatures from HAL:", 1)
    body = (
        current[1].split("Current cooling devices from HAL:", 1)[0]
        if len(current) == 2
        else ""
    )
    sensors = {}
    for value, kind, name, level in re.findall(
        r"Temperature\{mValue=([^,]+), mType=(\d+), mName=([^,]+), mStatus=(\d+)\}",
        body,
    ):
        numeric = float(value)
        if math.isfinite(numeric):
            sensors[name] = {
                "celsius": numeric,
                "type": int(kind),
                "status": int(level),
            }
    return {
        "android_status": int(status.group(1)) if status else None,
        "current_sensors": sensors,
        "htp_specific_throttling": "unknown; no HTP clock/throttle counter is exposed by this telemetry",
    }


def snapshot(full: bool = False) -> dict:
    sample = {"host_utc": now()}
    try:
        sample["thermal_raw"] = adb("dumpsys thermalservice")
        sample["thermal"] = parse_thermal(sample["thermal_raw"]["stdout"])
        if full:
            sample["battery_raw"] = adb("dumpsys battery")
            match = re.search(
                r"^\s*temperature:\s*(-?\d+)",
                sample["battery_raw"]["stdout"],
                re.MULTILINE,
            )
            sample["battery_c"] = int(match.group(1)) / 10 if match else None
            sample["device_clock"] = adb("date -u '+%Y-%m-%dT%H:%M:%SZ'")
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        sample["telemetry_error"] = repr(error)
    return sample


def deploy(root: Path) -> None:
    check_inputs(root)
    if (root / "measurement_freeze.json").exists():
        raise ValueError("Deployment must precede freeze")
    if not read(root / "audits/summary.json")["passed"]:
        raise ValueError("Audit not passed")
    if adb("pidof qnn-llm-runner")["stdout"].strip():
        raise RuntimeError(
            "Do not replace shared runtime files while a runner is active"
        )
    os.environ["ANDROID_SERIAL"] = SERIAL
    for group in GROUPS:
        execute(
            "run_device_llm.py",
            [
                "push",
                "--bundle-dir",
                str(root / group),
                "--runner",
                str(RUNNER),
                "--name",
                root.name + "_" + group,
            ],
            root / "logs" / f"push_{group}.log",
        )
    save_new(
        root / "device_preflight.json",
        {
            "thermal": snapshot(full=True),
            "free_space": adb("df -h /data/local/tmp"),
            "device": adb("getprop ro.product.model"),
            "soc": adb("getprop ro.soc.model"),
            "fingerprint": adb("getprop ro.build.fingerprint"),
            "processes": adb("pidof qnn-llm-runner"),
            "devfreq_interfaces": adb("ls /sys/class/devfreq"),
        },
    )


def device_identity(root: Path, paths: list[str]) -> dict:
    response = adb("sha256sum " + " ".join(paths), timeout=300)
    if response["returncode"]:
        raise ValueError("Device identity read failed: " + response["stderr"])
    found = {
        line.split()[1]: line.split()[0] for line in response["stdout"].splitlines()
    }
    if set(found) != set(paths):
        raise ValueError("Incomplete device hash manifest")
    return {"files": found, "raw": response}


def freeze(root: Path) -> None:
    from run_device_llm import DEVICE_LIBS, DEVICE_ROOT

    check_inputs(root)
    if (ASSETS / "boundary_prompt_cl1024.bin").stat().st_size != 897 * 4:
        raise ValueError("Expected exactly 897 int32 input token IDs")
    if (
        not read(root / "audits/summary.json")["passed"]
        or not (root / "device_preflight.json").exists()
    ):
        raise ValueError("Complete audit/deployment before measurement freeze")
    files = [
        root / "protocol.json",
        root / "audits/summary.json",
        Path(__file__).resolve(),
        Path(__file__).with_name("report_k8_repeat.py"),
        PREVIOUS / "comparison.csv",
        BOUNDARY_SOURCE.parent / "reports/boundary_perf.json",
    ]
    files += list(
        Path(__file__)
        .resolve()
        .parents[3]
        .glob("src/qai_hub_models/models/templates/llm/turboquant/*.py")
    )
    files += [
        Path(__file__).with_name(n)
        for n in (
            "run_device_llm.py",
            "verify_rotated_attention.py",
            "convert_parts.py",
        )
    ]
    for part in read(SPLIT / "split_manifest.json")["parts"].values():
        directory = Path(part["bundle_dir"])
        files += (
            list(directory.glob("*.onnx"))
            + list(directory.glob("*.encodings"))
            + list(directory.glob("*.data"))
        )
    device_expected = {f"{DEVICE_ROOT}/bin/qnn-llm-runner": digest(RUNNER)}
    for rel in DEVICE_LIBS:
        files.append(SDK / rel)
        device_expected[f"{DEVICE_ROOT}/bin/{Path(rel).name}"] = digest(SDK / rel)
    for group in GROUPS:
        bundle = root / group
        files += [bundle / "convert_report.json", bundle / "runtime_manifest.json"]
        native = read(bundle / "convert_report.json")["native_decoder"]["libraries"][
            "hexagon-v81"
        ]
        remote = f"{DEVICE_ROOT}/bundles/{root.name}_{group}"
        device_expected[f"{remote}/{Path(native['path']).name}"] = native["sha256"]
        device_expected[f"{remote}/runtime_manifest.json"] = digest(
            bundle / "runtime_manifest.json"
        )
        for path in bundle.glob("part*_of_4.bin"):
            files.append(path)
            device_expected[f"{remote}/{path.name}"] = digest(path)
    actual = device_identity(root, list(device_expected))
    if actual["files"] != device_expected:
        raise ValueError("Device files do not match frozen host artifacts")
    save_new(
        root / "measurement_freeze.json",
        {
            "created_utc": now(),
            "host_files": {str(p): digest(p) for p in files},
            "device": actual,
        },
    )
    print(
        "All four immutable configurations frozen; no further build/deploy.", flush=True
    )


def verify_freeze(root: Path, remote: bool = True) -> None:
    check_inputs(root)
    frozen = read(root / "measurement_freeze.json")
    for name, expected in frozen["host_files"].items():
        if digest(Path(name)) != expected:
            raise ValueError("Measured implementation changed: " + name)
    if (
        remote
        and device_identity(root, list(frozen["device"]["files"]))["files"]
        != frozen["device"]["files"]
    ):
        raise ValueError("Device files changed since freeze")


def extract_metrics(data: dict, group: str, token_hash: str) -> tuple[dict, list[str]]:
    """Top-level fields already describe the final session in the existing runner."""
    sessions = data["sessions"]
    decode_steps = [s for s in data["steps"] if s["kind"] == "decode"]
    checks = {
        "generate": data["mode"] == "generate",
        "prompt": data["prompt_tokens"] == 897,
        "context": data["context_length"] == 1024 and data["context_buckets"] == [1024],
        "warmup_then_measured": len(sessions) == 2
        and [s["session"] for s in sessions] == [0, 1],
        "generated": len(data["generated"]) == 128
        and all(len(s["generated"]) == 128 for s in sessions),
        "cache": all(s["cached_tokens_at_end"] == 1024 for s in sessions),
        "decode": len(data["decode_step_s"]) == len(decode_steps) == 127
        and all(s["graph_context"] == 1024 for s in decode_steps),
        "unprofiled": not data.get("op_profiles") and not data.get("diagnostic_only"),
        "power": data["burst_power"] is True,
        "kv_bytes": data["kv_store_bytes"] == 30277632,
        "config": data["config_hash"] == config(group).config_hash()
        and data["quantize_current_kv"],
        "tokens": data["assets"]["tokens_sha256"] == token_hash,
        "prefill_rate": math.isclose(
            data["prefill_tok_per_s"], 897 / data["prefill_s"], rel_tol=1e-7
        ),
        "decode_rate": math.isclose(
            data["decode_tok_per_s"], 127 / sum(data["decode_step_s"]), rel_tol=1e-7
        ),
        "last_session": all(
            data[k] == sessions[-1][k]
            for k in ("ttft_s", "prefill_tok_per_s", "decode_tok_per_s")
        ),
    }
    if not all(checks.values()):
        raise ValueError("Measurement contract failed: " + str(checks))
    metrics = {
        "ttft_ms": data["ttft_s"] * 1000,
        "prefill_tok_per_s": data["prefill_tok_per_s"],
        "decode_tok_per_s": data["decode_tok_per_s"],
    }
    if not all(math.isfinite(x) and x > 0 for x in metrics.values()):
        raise ValueError("Nonfinite/nonpositive performance metric")
    flags = []
    if sessions[0]["generated"] != sessions[1]["generated"]:
        flags.append("warmup_reset_generated_tokens_differ")
    metrics.update(
        {
            "warmup_ttft_ms": sessions[0]["ttft_s"] * 1000,
            "warmup_prefill_tok_per_s": sessions[0]["prefill_tok_per_s"],
            "warmup_decode_tok_per_s": sessions[0]["decode_tok_per_s"],
            "kv_store_bytes": data["kv_store_bytes"],
            "io_buffer_bytes": data["io_buffer_bytes"],
            "cached_tokens_at_end": 1024,
            "decode_iterations": 127,
        }
    )
    return metrics, flags


def thermal_summary(samples: list[dict]) -> dict:
    statuses = [
        s["thermal"]["android_status"]
        for s in samples
        if s.get("thermal", {}).get("android_status") is not None
    ]
    values = {}
    sensor_statuses = []
    for sample in samples:
        for name, item in sample.get("thermal", {}).get("current_sensors", {}).items():
            values.setdefault(name, []).append(item["celsius"])
            sensor_statuses.append(item["status"])
    output = {
        "thermal_samples": len(samples),
        "thermal_status_max": max(statuses) if statuses else None,
        "sensor_status_max": max(sensor_statuses) if sensor_statuses else None,
        "android_thermal_warning": any(s > 0 for s in statuses + sensor_statuses),
        "htp_throttling": "unknown",
        "telemetry_errors": sum("telemetry_error" in s for s in samples),
    }
    for name in ("AP", "BAT", "SKIN"):
        v = values.get(name, [])
        output.update(
            {
                f"{name.lower()}_temp_{label}_c": func(v) if v else None
                for label, func in (
                    ("min", min),
                    ("max", max),
                    ("mean", statistics.mean),
                )
            }
        )
    return output


def instrument_adb(
    original: Callable[..., str],
    record: dict,
    attempt: Path,
    samples: list[dict],
    stream: TextIO,
) -> Callable[..., str]:
    """Instrument the launch externally; no changes inside the timed runner."""
    stop = threading.Event()

    def capture(full: bool = False) -> None:
        sample = snapshot(full)
        samples.append(sample)
        stream.write(json.dumps(sample) + "\n")
        stream.flush()

    def poll() -> None:
        while not stop.wait(2):
            capture()

    def tracked(args: argparse.Namespace, *cmd: str, **kwargs: Any) -> str:
        if (
            cmd
            and cmd[0] == "shell"
            and any("__RC__" in x and "/bin/qnn-llm-runner" in x for x in cmd)
        ):
            capture(full=True)
            record["runner_start_utc"] = now()
            save_new(
                attempt / "runner_command.json",
                {"command": list(cmd), "host_utc": record["runner_start_utc"]},
            )
            thread = threading.Thread(target=poll, daemon=True)
            thread.start()
            try:
                return original(args, *cmd, **kwargs)
            finally:
                record["runner_end_utc"] = now()
                stop.set()
                thread.join()
                capture(full=True)
        return original(args, *cmd, **kwargs)

    return tracked


def measure(root: Path) -> None:
    import fcntl

    import run_device_llm as device

    # Advisory host lock; also reject an already-running device runner.
    with Path(f"/tmp/tq_repeat_{SERIAL}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if adb("pidof qnn-llm-runner")["stdout"].strip():
            raise RuntimeError("Another runner is already executing")
        verify_freeze(root)
        protocol = read(root / "protocol.json")
        token_hash = read(ASSETS / "assets.json")["sha256"][
            "boundary_prompt_cl1024.bin"
        ]
        os.environ["ANDROID_SERIAL"] = SERIAL
        for round_index, order in enumerate(protocol["schedule"], 1):
            for position, group in enumerate(order, 1):
                label = f"r{round_index:02d}_p{position}_{group}"
                attempt = root / "attempts" / label
                if attempt.exists():
                    # Resume completed attempts without executing them again.
                    if (attempt / "result.json").exists() and read(
                        attempt / "result.json"
                    ).get("success"):
                        continue
                    raise FileExistsError(
                        f"Incomplete/failed attempt retained, not retried: {attempt}"
                    )
                attempt.mkdir(parents=True)
                record = {
                    "round": round_index,
                    "position": position,
                    "ordinal": (round_index - 1) * 4 + position,
                    "group": group,
                    "placement": group.split("_")[0],
                    "quantizer": "Structured K8"
                    if group.endswith("structured")
                    else "Converged Lloyd-Max",
                    "attempt_start_utc": now(),
                }
                save_new(attempt / "start.json", record)
                samples = []
                original_adb = device.adb
                raw_telemetry = (attempt / "thermal.jsonl").open("x")
                device.adb = instrument_adb(
                    original_adb, record, attempt, samples, raw_telemetry
                )
                try:
                    # Reuse the untouched existing CLI and C++ timing/reset path.
                    argv = sys.argv
                    sys.argv = [
                        "run_device_llm.py",
                        "run",
                        "--name",
                        root.name + "_" + group,
                        "--assets",
                        str(ASSETS),
                        "--mode",
                        "generate",
                        "--n-gen",
                        "128",
                        "--sessions",
                        "2",
                        "--tokens",
                        "boundary_prompt_cl1024.bin",
                        "--report",
                        str(attempt / "runner.json"),
                        "--remote-report-tag",
                        root.name + "_" + label,
                    ]
                    try:
                        device.main()
                    finally:
                        sys.argv = argv
                    metrics, flags = extract_metrics(
                        read(attempt / "runner.json"), group, token_hash
                    )
                    thermal = thermal_summary(samples)
                    if thermal["android_thermal_warning"]:
                        flags.append("android_thermal_warning")
                    if (
                        thermal["telemetry_errors"]
                        or not thermal["thermal_samples"]
                        or thermal["thermal_status_max"] is None
                    ):
                        flags.append("incomplete_thermal_telemetry")
                    record.update(
                        metrics,
                        **thermal,
                        flags=flags,
                        success=True,
                        attempt_end_utc=now(),
                        report=str(attempt / "runner.json"),
                    )
                    save_new(attempt / "result.json", record)
                    print(
                        f"MEASURED {record['ordinal']}/40 {group}: {metrics}",
                        flush=True,
                    )
                except Exception as error:
                    record.update(
                        success=False,
                        error=repr(error),
                        attempt_end_utc=now(),
                        **thermal_summary(samples),
                    )
                    save_new(attempt / "failure.json", record)
                    raise
                finally:
                    device.adb = original_adb
                    raw_telemetry.close()
                time.sleep(protocol["inter_invocation_pause_s"])
        verify_freeze(root)
        save_new(
            root / "measurement_complete.json",
            {
                "completed_utc": now(),
                "measured": 40,
                "warmup": 40,
                "post_measurement_hash_check": "passed",
            },
        )


def describe(values: list[float]) -> dict:
    if not values:
        return dict(n=0, mean=None, sample_sd=None, median=None, min=None, max=None)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def improvement(lm: float, structured: float, metric: str) -> float:
    return 100 * (
        (lm - structured) / lm if metric == "ttft_ms" else (structured - lm) / lm
    )


def analyze(root: Path) -> None:
    import numpy as np

    rows = sorted(
        [read(p) for p in root.glob("attempts/*/result.json")],
        key=lambda r: r["ordinal"],
    )
    failures = [read(p) for p in root.glob("attempts/*/failure.json")]
    if len(rows) != 40 or failures or not (root / "measurement_complete.json").exists():
        raise ValueError(
            "Require all 40 completed samples; do not silently discard failures"
        )
    if any(Counter(r["group"] for r in rows)[g] != 10 for g in GROUPS):
        raise ValueError("Unbalanced sample counts")
    fields = list(rows[0])
    with (root / "raw.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**r, "flags": ";".join(r["flags"])} for r in rows)
    summary = {
        g: {m: describe([r[m] for r in rows if r["group"] == g]) for m in METRICS}
        for g in GROUPS
    }
    gains = {
        p: {
            m: improvement(
                summary[p + "_lm"][m]["mean"], summary[p + "_structured"][m]["mean"], m
            )
            for m in METRICS
        }
        for p in MASKS
    }
    # Paired-round bootstrap is supplemental, not a substitute for the full table.
    rng = np.random.default_rng(20261027)
    bootstrap = {}
    for placement in MASKS:
        bootstrap[placement] = {}
        for metric in METRICS:
            lm = np.array([r[metric] for r in rows if r["group"] == placement + "_lm"])
            structured = np.array(
                [r[metric] for r in rows if r["group"] == placement + "_structured"]
            )
            draw = rng.integers(0, 10, size=(10000, 10))
            estimates = improvement(
                lm[draw].mean(axis=1), structured[draw].mean(axis=1), metric
            )
            bootstrap[placement][metric] = list(
                map(float, np.quantile(estimates, [0.025, 0.975]))
            )
    diagnostics = {}
    for group in GROUPS:
        group_rows = [r for r in rows if r["group"] == group]
        diagnostics[group] = {
            "position_counts": dict(Counter(r["position"] for r in group_rows)),
            "flags": [r["ordinal"] for r in group_rows if r["flags"]],
            "thermal_status_max": max(
                (
                    r["thermal_status_max"]
                    for r in group_rows
                    if r["thermal_status_max"] is not None
                ),
                default=None,
            ),
            "ap_temp_c": describe(
                [
                    r["ap_temp_mean_c"]
                    for r in group_rows
                    if r["ap_temp_mean_c"] is not None
                ]
            ),
            "by_position": {
                str(p): {
                    m: describe([r[m] for r in group_rows if r["position"] == p])
                    for m in METRICS
                }
                for p in range(1, 5)
            },
            "correlations": {},
        }
        for predictor in ("round", "position", "ap_temp_mean_c"):
            eligible = [r for r in group_rows if r[predictor] is not None]
            x = [r[predictor] for r in eligible]
            diagnostics[group]["correlations"][predictor] = {
                m: float(np.corrcoef(x, [r[m] for r in eligible])[0, 1])
                if len(x) > 2 and np.std(x) > 0 and np.std([r[m] for r in eligible]) > 0
                else None
                for m in METRICS
            }
    old = list(csv.DictReader((PREVIOUS / "comparison.csv").open()))
    historical = {
        r["group"]: {m: float(r[m]) for m in METRICS}
        for r in old
        if r["group"] in ("lm8_converged", "structured")
    }
    boundary_old = read(BOUNDARY_SOURCE.parent / "reports/boundary_perf.json")
    output = {
        "samples": 40,
        "warmups_excluded": 40,
        "summary": summary,
        "improvement_pct": gains,
        "boundary_minus_odd_improvement_pp": {
            m: gains["boundary"][m] - gains["odd"][m] for m in METRICS
        },
        "paired_round_bootstrap_95pct": bootstrap,
        "bootstrap_note": "10000 paired-round resamples, seed20261027; small n, exploratory; shared thermal drift/autocorrelation can limit intervals.",
        "diagnostics": diagnostics,
        "historical_single_run_odd": historical,
        "historical_single_run_boundary_structured": {
            "ttft_ms": boundary_old["ttft_s"] * 1000,
            **{m: boundary_old[m] for m in METRICS[1:]},
            "source": str(BOUNDARY_SOURCE.parent / "reports/boundary_perf.json"),
            "lm_boundary_historical": None,
        },
        "limitations": [
            "Historical single-run results had no same-process warm-up; direct historical deltas also reflect session state/date/device conditions.",
            "Battery/AP/SKIN and Android thermal status are observations, not direct HTP frequency or throttling counters.",
            "No samples excluded; no causal attribution to temperature/order from n=10.",
            "Complete implementations differ in quantizer and decoder; not a decoder-only or algorithm-independent LM claim.",
            "No PPL rerun; historical Odd PPL remains historical and no Boundary PPL is inferred.",
        ],
    }
    save_new(root / "summary.json", output)
    with (root / "summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "group",
                "metric",
                "n",
                "mean",
                "sample_sd",
                "median",
                "min",
                "max",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {"group": g, "metric": m, **value}
            for g, metrics in summary.items()
            for m, value in metrics.items()
        )
    print(
        json.dumps({"summary": summary, "improvement_pct": gains}, indent=2), flush=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "build",
            "assemble",
            "audit",
            "deploy",
            "freeze",
            "measure",
            "analyze",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.root.resolve())


if __name__ == "__main__":
    main()
