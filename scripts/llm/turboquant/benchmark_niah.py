# SPDX-License-Identifier: BSD-3-Clause
"""Frozen, one-pass NIAH orchestration using the unchanged HTP LLM runner.

Stages: assets, deploy, pilot (FP16 only), probe (separate smoke example),
measure (test only). No automatic retries, fallback, rebuilds or deletion.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmark_k8_repeat import ADB, RUNNER, SDK, SERIAL
from niah_data import digest, save_new

GROUPS = ("fp16", "k4_v4", "boundary_lm", "boundary_structured")
DEVICE_ROOT = "/data/local/tmp/qaihm_turboquant/llm"
EOS = {151643, 151645}  # Unchanged runner defaults; checked against tokenizer.


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def valid_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe experiment/file identifier: {value!r}")
    return value


def name(root: Path, capacity: int, group: str) -> str:
    if group not in GROUPS or capacity < 1:
        raise ValueError("Unknown profile/capacity")
    return valid_name(f"{valid_name(root.name)}_cl{capacity}_{group}")


def protocol(root: Path, capacity: int) -> tuple[dict, dict]:
    p, data = read(root / "protocol.json"), read(root / "data/data_manifest.json")
    if list(p["groups"]) != list(GROUPS):
        raise ValueError("The frozen four-profile comparison changed")
    if capacity not in p["context_capacities"]:
        raise ValueError("Capacity was not authorized by the frozen protocol")
    if list(data["protocol"]["capacities"]) != list(p["context_capacities"]):
        raise ValueError("Data/build context capacities do not match")
    if p["max_generated_tokens"] != 64 or data["protocol"]["max_new_tokens"] != 64:
        raise ValueError("Generation budget changed")
    return p, data


def assets_path(root: Path, capacity: int) -> Path:
    return root / "runtime_assets" / f"cl{capacity}"


def remote_assets(assets: dict) -> str:
    identity = hashlib.sha256(
        json.dumps(assets["sha256"], sort_keys=True).encode()
    ).hexdigest()[:16]
    return f"{DEVICE_ROOT}/assets/cl{assets['context_length']}_{identity}"


def assets(root: Path, capacity: int) -> None:
    """Generate only RoPE, symlink frozen prompt IDs; no dataset/network calls."""
    _, data = protocol(root, capacity)
    target = assets_path(root, capacity)
    if target.exists():
        verify_asset_files(target)
        print(f"Verified existing immutable assets: {target}", flush=True)
        return
    from transformers import AutoConfig

    from qai_hub_models.models.templates.lm_driver.utils.rope_embedding import (
        RopeEmbedding,
    )

    checkpoint = Path(data["tokenizer"]["checkpoint_dir"])
    model_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    rope = RopeEmbedding(
        model=SimpleNamespace(config=model_config), context_length=capacity
    )
    cos = rope.cos[0, 0].numpy().astype("<f4")
    sin = rope.sin[0, 0].numpy().astype("<f4")
    target.mkdir(parents=True, exist_ok=False)
    rope_name = f"rope_cl{capacity}.bin"
    with (target / rope_name).open("xb") as stream:
        stream.write(cos.tobytes() + sin.tobytes())
    hashes = {rope_name: digest(target / rope_name)}
    cases = [case for case in data["cases"] if case["capacity"] == capacity]
    for case in cases:
        source = root / "data" / case["tokens_file"]
        if (
            digest(source) != case["tokens_sha256"]
            or source.stat().st_size != case["input_tokens"] * 4
        ):
            raise ValueError(f"Frozen case changed: {case['case_id']}")
        basename = valid_name(source.name)
        (target / basename).symlink_to(source.resolve(strict=True))
        hashes[basename] = case["tokens_sha256"]
    if not cases:
        raise ValueError("No frozen cases for capacity")
    save_new(
        target / "assets.json",
        {
            "checkpoint_dir": str(checkpoint),
            "context_length": capacity,
            "rope": rope_name,
            "rope_half": int(cos.shape[1]),
            "prompt": "Frozen NIAH chat prompt; use per-case manifest for actual text and token count.",
            "prompt_ids": Path(cases[0]["tokens_file"]).name,
            "prompt_tokens": cases[0]["input_tokens"],
            "wikitext_windows": [],
            "sha256": hashes,
            "data_manifest_sha256": digest(root / "data/data_manifest.json"),
        },
    )
    print(
        f"Prepared {len(cases)} immutable inputs and RoPE for CL{capacity}", flush=True
    )


def verify_asset_files(directory: Path) -> dict:
    a = read(directory / "assets.json")
    for file, expected in a["sha256"].items():
        if digest(directory / file) != expected:
            raise ValueError(f"Asset changed after preparation: {file}")
    return a


def freeze_sources(root: Path, capacity: int) -> dict:
    p, data = protocol(root, capacity)
    path = root / "execution_freeze.json"
    if path.exists():
        frozen = read(path)
    else:
        files = [root / "protocol.json", root / "data/data_manifest.json", RUNNER]
        script = Path(__file__).resolve().parent
        files += [
            script / n
            for n in (
                "benchmark_niah.py",
                "niah_data.py",
                "niah_report.py",
                "niah_build.py",
                "niah_audit.py",
                "run_device_llm.py",
                "model_identity.py",
                "convert_parts.py",
            )
        ]
        files += sorted(
            (
                script.parents[2] / "src/qai_hub_models/models/templates/llm/turboquant"
            ).rglob("*.py")
        )
        checkpoint = Path(data["tokenizer"]["checkpoint_dir"])
        files += [checkpoint / "config.json"]
        files += [Path(file) for file in data["tokenizer"]["files_sha256"]]
        frozen = {
            "utc": now(),
            "files_sha256": {str(file.resolve()): digest(file) for file in files},
            "runner": str(RUNNER),
            "sdk": str(SDK),
            "serial": SERIAL,
            "sessions": 1,
            "n_gen": 64,
            "stop_on_eos": True,
            "generation_decoding": "greedy; decode generated continuation with skip_special_tokens=True",
            "gate": "both FP16 short pilots exact; per-group capacity smoke must satisfy execution contract (correctness recorded, not selected)",
            "order": "caller-selected capacity/profile blocks; within each block immutable manifest order, no outcome-dependent reordering",
            "failure_policy": p["failure_policy"],
        }
        save_new(path, frozen)
    for file, expected in frozen["files_sha256"].items():
        if digest(Path(file)) != expected:
            raise ValueError(f"Execution source changed after freeze: {file}")
    for file, expected in data["tokenizer"]["files_sha256"].items():
        if digest(Path(file)) != expected:
            raise ValueError(f"Data tokenizer changed: {file}")
    return frozen


def namespace(root: Path, capacity: int, group: str) -> argparse.Namespace:
    return argparse.Namespace(
        adb=Path(ADB),
        sdk=SDK,
        runner=RUNNER,
        name=name(root, capacity, group),
        bundle_dir=root / "bundles" / f"cl{capacity}" / group,
        native_runtime_package=None,
        assets=assets_path(root, capacity),
        mode="generate",
        tokens="",
        n_gen=64,
        stop_on_eos=True,
        graph_suffix="",
        context_buckets=[],
        dump_logits=False,
        profile_decode_step=-1,
        profile_prefill=False,
        profile_prefill_all=False,
        profile_decode_steps=[],
        runner_name="qnn-llm-runner",
        optrace_out=None,
        sessions=1,
        report=Path(),
        remote_report_tag="",
    )


def assert_idle(device: Any, args: argparse.Namespace) -> None:
    value = device.adb(
        args, "shell", "if pidof qnn-llm-runner; then true; else echo __IDLE__; fi"
    ).strip()
    if value != "__IDLE__":
        raise RuntimeError(
            f"Another runner is active or device is unavailable: {value!r}"
        )


def remote_hashes(
    device: Any, args: argparse.Namespace, paths: list[str]
) -> dict[str, str]:
    if not paths:
        return {}
    raw = device.adb(
        args, "shell", "sha256sum " + " ".join(shlex.quote(p) for p in paths)
    )
    result = {}
    for line in raw.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise RuntimeError(f"Invalid/missing device hash line: {line}")
        result[parts[1].lstrip("*")] = parts[0]
    if set(result) != set(paths):
        raise RuntimeError("Incomplete device hash set")
    return result


def verify_file_map(device: Any, args: argparse.Namespace, files: list[dict]) -> None:
    expected = {}
    for item in files:
        if digest(Path(item["local"])) != item["sha256"]:
            raise ValueError(f"Frozen host file changed: {item['local']}")
        expected[item["remote"]] = item["sha256"]
    if remote_hashes(device, args, list(expected)) != expected:
        raise ValueError("Device files differ from frozen host files")


def deployment_path(root: Path, capacity: int, group: str) -> Path:
    return root / "deployments" / f"cl{capacity}_{group}.json"


def deploy(root: Path, capacity: int, group: str) -> None:
    import run_device_llm as device

    os.environ["ANDROID_SERIAL"] = SERIAL
    p, _ = protocol(root, capacity)
    freeze_sources(root, capacity)
    args = namespace(root, capacity, group)
    a = verify_asset_files(args.assets)
    target = deployment_path(root, capacity, group)
    if target.exists():
        verify_file_map(device, args, read(target)["files"])
        print(f"Verified preserved deployment {args.name}", flush=True)
        return
    meta = read(args.bundle_dir / "convert_report.json")
    audit_file = root / "audits" / f"cl{capacity}" / f"{group}.json"
    audit = read(audit_file)
    if not audit.get("passed") or not audit.get("compiled_io", {}).get("passed"):
        raise ValueError("Compiled graph/source/I/O audit must pass before deployment")
    if meta["config_hash"] != p["profiles"][group]["config_hash"] or meta[
        "context_buckets"
    ] != [capacity]:
        raise ValueError("Bundle profile/context differs from frozen protocol")
    if (args.bundle_dir / "runtime_manifest.json").is_symlink():
        raise ValueError("Runtime metadata must be a private experiment-local file")
    assert_idle(device, args)
    logs = root / "deployment_logs"
    logs.mkdir(exist_ok=True)
    with (
        (logs / f"cl{capacity}_{group}.log").open("x") as log,
        contextlib.redirect_stdout(log),
    ):
        device.cmd_push(args)
        remote = remote_assets(a)
        device.adb(args, "shell", "mkdir", "-p", remote)
        for file in a["sha256"]:
            device.push_if_changed(args, args.assets / file, remote + "/" + file)
    runtime = read(args.bundle_dir / "runtime_manifest.json")
    remote_bundle = f"{DEVICE_ROOT}/bundles/{args.name}"
    pairs = [
        (args.bundle_dir / f"part{i}_of_4.bin", f"{remote_bundle}/part{i}_of_4.bin")
        for i in range(1, 5)
    ]
    pairs += [
        (
            args.bundle_dir / "runtime_manifest.json",
            remote_bundle + "/runtime_manifest.json",
        )
    ]
    pairs += [
        (SDK / rel, f"{DEVICE_ROOT}/bin/{Path(rel).name}") for rel in device.DEVICE_LIBS
    ]
    pairs += [(RUNNER, f"{DEVICE_ROOT}/bin/qnn-llm-runner")]
    if meta.get("native_decoder"):
        native = meta["native_decoder"]["libraries"]["hexagon-v81"]
        if digest(Path(native["path"])) != native["sha256"]:
            raise ValueError("Compiled Native library changed")
        pairs += [
            (
                Path(native["path"]),
                remote_bundle + "/" + runtime["native_decoder"]["library"],
            )
        ]
    pairs += [(args.assets / file, remote + "/" + file) for file in a["sha256"]]
    files = [
        {
            "local": str(local.absolute()),
            "remote": remote,
            "sha256": digest(local),
            "bytes": local.stat().st_size,
        }
        for local, remote in pairs
    ]
    verify_file_map(device, args, files)
    target.parent.mkdir(exist_ok=True)
    save_new(
        target,
        {
            "utc": now(),
            "name": args.name,
            "capacity": capacity,
            "group": group,
            "config_hash": meta["config_hash"],
            "native_decoder": runtime.get("native_decoder"),
            "convert_report_sha256": digest(args.bundle_dir / "convert_report.json"),
            "compiled_audit_file": str(audit_file.relative_to(root)),
            "compiled_audit_sha256": digest(audit_file),
            "compiled_io": audit["compiled_io"],
            "assets_manifest_sha256": digest(args.assets / "assets.json"),
            "execution_freeze_sha256": digest(root / "execution_freeze.json"),
            "files": files,
            "device": {
                "serial": SERIAL,
                "soc": device.adb(args, "shell", "getprop", "ro.soc.model").strip(),
            },
        },
    )
    print(f"Deployed and hash-verified {args.name}", flush=True)


def partial_first_chunks(count: int) -> list[tuple[int, int]]:
    if count < 1:
        raise ValueError("Empty prompt")
    ar = 1 if count == 1 else 128
    remainder = count % ar
    chunks = [(0, remainder)] if remainder else []
    chunks += [(start, ar) for start in range(remainder, count, ar)]
    return chunks


def validate_report(
    report: dict, case: dict, deployment: dict, bundle_name: str
) -> dict:
    capacity, count = case["capacity"], case["input_tokens"]
    generated = report.get("generated", [])
    sessions = report.get("sessions", [])
    steps = report.get("steps", [])
    prefill = [step for step in steps if step.get("kind") == "prefill"]
    decode = [step for step in steps if step.get("kind") == "decode"]
    expected = partial_first_chunks(count)
    cached = count + len(generated) - 1
    native = deployment.get("native_decoder")
    reason = report.get("stop_reason")
    bin_files = [
        item
        for item in deployment["files"]
        if re.fullmatch(r"part\d+_of_4.bin", Path(item["remote"]).name)
    ]
    checks = {
        "generate": report.get("mode") == "generate",
        "htp_backend": "htp" in report.get("backend_api", "").lower(),
        "context": report.get("context_length") == capacity
        and report.get("context_buckets") == [capacity],
        "prompt": report.get("prompt_tokens") == count and count + 64 <= capacity,
        "tokens": report.get("assets", {}).get("tokens_sha256") == case["tokens_sha256"]
        and report.get("assets", {}).get("tokens_file")
        == Path(case["tokens_file"]).name,
        "bundle": report.get("bundle_name") == bundle_name
        and report.get("config_hash") == deployment["config_hash"],
        "native": report.get("native_decoder") == native,
        "memory": math.isclose(
            report.get("kv_store_bytes", -1),
            capacity * (114688 if deployment["group"] == "fp16" else 29568),
            rel_tol=1e-8,
        )
        and math.isclose(
            report.get("kv_store_bytes", -1),
            deployment["compiled_io"]["host_kv_bytes"],
            rel_tol=1e-8,
        )
        and math.isclose(
            report.get("io_buffer_bytes", -1),
            deployment["compiled_io"]["runtime_io_bytes"],
            rel_tol=1e-8,
        ),
        "finite_times": all(
            math.isfinite(report.get(field, float("nan")))
            and report.get(field, -1) >= 0
            for field in ("load_s", "ttft_s", "prefill_s")
        ),
        "current_kv": report.get("quantize_current_kv")
        == (deployment["group"] != "fp16"),
        "single_session": len(sessions) == 1
        and sessions[0].get("session") == 0
        and sessions[0].get("generated") == generated
        and sessions[0].get("cached_tokens_at_end") == cached <= capacity,
        "generation_count": 1 <= len(generated) <= 64
        and all(isinstance(token, int) and token >= 0 for token in generated),
        "stop": (
            reason == "eos"
            and bool(generated)
            and generated[-1] in EOS
            and len(generated) < 64
        )
        or (reason == "length" and len(generated) == 64),
        "no_early_eos": not any(token in EOS for token in generated[:-1]),
        "partial_first_prefill": [
            (step["cached_before"], step["new_tokens"]) for step in prefill
        ]
        == expected
        and all(
            step["ar"] == (1 if count == 1 else 128)
            and step["graph_context"] == capacity
            for step in prefill
        ),
        "decode_cache": len(decode) == max(0, len(generated) - 1)
        and all(
            step["ar"] == step["new_tokens"] == 1
            and step["cached_before"] == count + i
            and step["graph_context"] == capacity
            for i, step in enumerate(decode)
        ),
        "no_other_steps": len(steps) == len(prefill) + len(decode),
        "unprofiled": not report.get("diagnostic_only")
        and not report.get("op_profiles"),
        "binaries": {item["path"]: item["bytes"] for item in report.get("bins", [])}
        == {item["remote"]: item["bytes"] for item in bin_files},
    }
    if not all(checks.values()):
        raise ValueError(f"NIAH execution contract failed: {checks}")
    return {
        "checks": checks,
        "cached_tokens_at_end": cached,
        "decode_iterations": len(decode),
    }


def result_path(root: Path, group: str, case: dict) -> Path:
    return root / "results" / f"{group}_{valid_name(case['case_id'])}.json"


def check_existing(
    root: Path, group: str, case: dict, deployment_hash: str
) -> dict | None:
    path = result_path(root, group, case)
    if not path.exists():
        return None
    result = read(path)
    if result.get("status") != "ok":
        raise RuntimeError(
            f"Previous failed attempt retained; no automatic retry: {path}"
        )
    if (
        result.get("case_id") != case["case_id"]
        or result.get("group") != group
        or result.get("tokens_sha256") != case["tokens_sha256"]
        or result.get("deployment_sha256") != deployment_hash
    ):
        raise ValueError(
            "Existing result belongs to a different frozen input/deployment"
        )
    if digest(root / result["runner_report"]) != result["runner_report_sha256"]:
        raise ValueError("Prior successful runner report changed")
    return result


@contextlib.contextmanager
def efficient_device(
    device: Any,
    args: argparse.Namespace,
    tokenizer: Any,
    assets: dict,
    command_log: Path,
) -> Iterator[None]:
    """Reuse tokenization and already hash-verified assets, not inference results."""
    original_push, original_tokenizer, original_adb = (
        device.push_if_changed,
        device.AutoTokenizer,
        device.adb,
    )
    allowed = {
        remote_assets(assets) + "/" + file: (args.assets / file).resolve()
        for file in assets["sha256"]
    }
    stat_ids = {
        remote: (path.stat().st_size, path.stat().st_mtime_ns)
        for remote, path in allowed.items()
    }

    def cached_push(unused: Any, local: Path, remote: str) -> bool:
        resolved = local.resolve()
        if (
            remote not in allowed
            or resolved != allowed[remote]
            or (resolved.stat().st_size, resolved.stat().st_mtime_ns)
            != stat_ids[remote]
        ):
            raise ValueError("Unexpected/changed asset during frozen execution")
        return False

    def logged_adb(ns: Any, *command: str, check: bool = True) -> str:
        start = now()
        try:
            output = original_adb(ns, *command, check=check)
        except Exception as error:
            with command_log.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "start_utc": start,
                            "end_utc": now(),
                            "command": command,
                            "error": repr(error),
                        }
                    )
                    + "\n"
                )
            raise
        with command_log.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "start_utc": start,
                        "end_utc": now(),
                        "command": command,
                        "stdout": output,
                    }
                )
                + "\n"
            )
        return output

    device.push_if_changed = cached_push
    device.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
    device.adb = logged_adb
    try:
        yield
    finally:
        device.push_if_changed, device.AutoTokenizer, device.adb = (
            original_push,
            original_tokenizer,
            original_adb,
        )


def gates(root: Path, capacity: int, group: str, stage: str, data: dict) -> None:
    if stage == "pilot":
        if group != "fp16" or capacity != min(data["protocol"]["capacities"]):
            raise ValueError("Pilot must use FP16 at the smallest frozen capacity")
        return
    pilots = [case for case in data["cases"] if case["split"] == "pilot"]
    if len(pilots) != 2:
        raise ValueError("Expected exactly two frozen FP16 prompt-format pilots")
    completion = (
        root
        / "stage_completion"
        / f"pilot_cl{min(data['protocol']['capacities'])}_fp16.json"
    )
    if not completion.exists() or not read(completion).get(
        "pre_post_host_device_hashes_verified"
    ):
        raise ValueError(
            "Both frozen FP16 prompt-format pilots must pass pre/post verification before probe/test"
        )
    for case in pilots:
        path = result_path(root, "fp16", case)
        if (
            not path.exists()
            or read(path).get("status") != "ok"
            or not read(path).get("exact_match")
        ):
            raise ValueError(
                "Both frozen FP16 prompt-format pilots must pass before probe/test"
            )
        check_existing(
            root, "fp16", case, digest(deployment_path(root, case["capacity"], "fp16"))
        )
    if stage == "measure":
        smoke = [
            case
            for case in data["cases"]
            if case["split"] == "smoke" and case["capacity"] == capacity
        ]
        if len(smoke) != 1:
            raise ValueError("Expected one separate smoke case per capacity")
        completion = root / "stage_completion" / f"probe_cl{capacity}_{group}.json"
        if not completion.exists() or not read(completion).get(
            "pre_post_host_device_hashes_verified"
        ):
            raise ValueError(
                "This profile/capacity needs a post-verified operational smoke probe first"
            )
        path = result_path(root, group, smoke[0])
        if not path.exists() or read(path).get("status") != "ok":
            raise ValueError(
                "This profile/capacity needs a successful operational smoke probe first"
            )
        check_existing(
            root, group, smoke[0], digest(deployment_path(root, capacity, group))
        )


def require_new_remote_report(
    device: Any, args: argparse.Namespace, remote_report: str
) -> None:
    exists = device.adb(
        args,
        "shell",
        f"if test -e {shlex.quote(remote_report)}; then echo EXISTS; else echo NEW; fi",
    ).strip()
    if exists != "NEW":
        raise FileExistsError("Remote report exists or device state unavailable")


def run(root: Path, capacity: int, group: str, stage: str) -> None:
    import run_device_llm as device
    from transformers import AutoTokenizer

    os.environ["ANDROID_SERIAL"] = SERIAL
    _, data = protocol(root, capacity)
    freeze_sources(root, capacity)
    gates(root, capacity, group, stage, data)
    args = namespace(root, capacity, group)
    deployment_file = deployment_path(root, capacity, group)
    dep = read(deployment_file)
    deployment_hash = digest(deployment_file)
    a = verify_asset_files(args.assets)
    if (
        digest(args.assets / "assets.json") != dep["assets_manifest_sha256"]
        or digest(args.bundle_dir / "convert_report.json")
        != dep["convert_report_sha256"]
        or digest(root / dep["compiled_audit_file"]) != dep["compiled_audit_sha256"]
        or digest(root / "execution_freeze.json") != dep["execution_freeze_sha256"]
    ):
        raise ValueError("Bundle/asset metadata changed after deployment")
    verify_file_map(device, args, dep["files"])
    assert_idle(device, args)
    tokenizer = AutoTokenizer.from_pretrained(
        data["tokenizer"]["checkpoint_dir"], local_files_only=True
    )
    if tokenizer.eos_token_id not in EOS:
        raise ValueError("Tokenizer EOS differs from unchanged runner defaults")
    selected = [
        case
        for case in data["cases"]
        if case["capacity"] == capacity
        and case["split"]
        == {"pilot": "pilot", "probe": "smoke", "measure": "test"}[stage]
    ]
    if not selected:
        raise ValueError("No cases selected")
    (root / "results").mkdir(exist_ok=True)
    for case in selected:
        existing = check_existing(root, group, case, deployment_hash)
        if existing is not None:
            print(
                f"Verified completed case, no repeat: {group}/{case['case_id']}",
                flush=True,
            )
            continue
        label = valid_name(f"{group}_{case['case_id']}")
        attempt = root / "attempts" / label
        if attempt.exists():
            raise FileExistsError(
                f"Incomplete attempt preserved; no automatic retry: {attempt}"
            )
        attempt.mkdir(parents=True)
        args.tokens = Path(case["tokens_file"]).name
        args.report = attempt / "runner.json"
        args.remote_report_tag = valid_name(root.name + "_" + label)
        start = now()
        save_new(
            attempt / "start.json",
            {
                "utc": start,
                "stage": stage,
                "case_id": case["case_id"],
                "group": group,
                "deployment_sha256": deployment_hash,
                "settings": {
                    "n_gen": 64,
                    "sessions": 1,
                    "stop_on_eos": True,
                    "tokens": args.tokens,
                    "bundle": args.name,
                },
            },
        )
        try:
            remote_report = f"{DEVICE_ROOT}/reports/{args.remote_report_tag}.json"
            require_new_remote_report(device, args, remote_report)
            assert_idle(device, args)
            with (
                (attempt / "host_stdout.log").open("x") as log,
                contextlib.redirect_stdout(log),
                efficient_device(
                    device, args, tokenizer, a, attempt / "commands.jsonl"
                ),
            ):
                device.cmd_run(args)
            report = read(args.report)
            validation = validate_report(report, case, dep, args.name)
            response = tokenizer.decode(report["generated"], skip_special_tokens=True)
            # Scoring is intentionally minimal here; niah_report supplies the
            # secondary extraction metric and the aggregate confidence intervals.
            result = {
                "status": "ok",
                "case_id": case["case_id"],
                "group": group,
                "split": case["split"],
                "capacity": capacity,
                "base_id": case["base_id"],
                "depth": case["depth"],
                "input_tokens": case["input_tokens"],
                "tokens_sha256": case["tokens_sha256"],
                "expected_answer": case["expected_answer"],
                "response": response,
                "exact_match": response.strip() == case["expected_answer"],
                "generated_ids": report["generated"],
                "generated_tokens": len(report["generated"]),
                "max_new_tokens": 64,
                "raw_generated_text": report["generated_text"],
                "stop_reason": report["stop_reason"],
                "finish_reason": report["stop_reason"],
                "start_utc": start,
                "end_utc": now(),
                "config_hash": dep["config_hash"],
                "deployment_sha256": deployment_hash,
                "runner_report": str(args.report.relative_to(root)),
                "runner_report_sha256": digest(args.report),
                "validation": validation,
                "kv_store_bytes": report.get("kv_store_bytes"),
                "runtime_io_bytes": report.get("io_buffer_bytes"),
                "timing_scope": "quality run only; not controlled performance comparison",
            }
            save_new(result_path(root, group, case), result)
            print(
                f"{stage} {group} {case['case_id']}: exact={result['exact_match']} response={response!r}",
                flush=True,
            )
        except Exception as error:
            error_result = {
                "status": "error",
                "case_id": case["case_id"],
                "group": group,
                "capacity": capacity,
                "split": case["split"],
                "base_id": case["base_id"],
                "depth": case["depth"],
                "tokens_sha256": case["tokens_sha256"],
                "deployment_sha256": deployment_hash,
                "start_utc": start,
                "end_utc": now(),
                "error": repr(error),
                "failure_class": "infrastructure_or_contract_failure_not_model_wrong_answer",
            }
            if not result_path(root, group, case).exists():
                save_new(result_path(root, group, case), error_result)
            raise
    verify_file_map(device, args, dep["files"])
    freeze_sources(root, capacity)
    completion = root / "stage_completion" / f"{stage}_cl{capacity}_{group}.json"
    if not completion.exists():
        completion.parent.mkdir(exist_ok=True)
        save_new(
            completion,
            {
                "utc": now(),
                "cases": [c["case_id"] for c in selected],
                "deployment_sha256": deployment_hash,
                "pre_post_host_device_hashes_verified": True,
            },
        )
    if stage == "pilot" and not all(
        read(result_path(root, group, case))["exact_match"] for case in selected
    ):
        raise RuntimeError(
            "FP16 prompt-format pilot failed; preserve outputs and stop before formal evaluation"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("assets", "deploy", "pilot", "probe", "measure")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--group", choices=GROUPS, default="fp16")
    args = parser.parse_args()
    root = args.root.expanduser().absolute()
    if args.stage == "assets":
        assets(root, args.capacity)
    elif args.stage == "deploy":
        deploy(root, args.capacity, args.group)
    else:
        run(root, args.capacity, args.group, args.stage)


if __name__ == "__main__":
    main()
