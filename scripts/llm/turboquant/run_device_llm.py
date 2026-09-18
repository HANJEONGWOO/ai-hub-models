# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Stage and run split LLM context binaries on an Android HTP device with qnn-llm-runner.

Stages:
  assets  RoPE float32 table (cos then sin, first half of the rotary dims, as
          HubCompatibleGenerator builds it), a chat prompt and WikiText windows
          as int32 token files.
  push    Runner, QNN runtime libraries and one bundle's part*_of_N.bin files,
          skipping files whose sha256 already matches on the device.
  run     Execute the runner (generate or score) and pull its JSON report.

Usage:

    PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py assets \
        --checkpoint-dir ~/.qaihm/qai-hub-models/models/qwen3_1_7b/v2/qwen3_1_7b_w4a16 \
        --context-length 1024 --out ~/.qaihm/tmp/turboquant/device_assets_cl1024
    PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py push \
        --bundle-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_baseline_int8_cl1024 --name baseline_int8_cl1024
    PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py run --name baseline_int8_cl1024 \
        --assets ~/.qaihm/tmp/turboquant/device_assets_cl1024 --mode generate --n-gen 128 \
        --report ~/.qaihm/tmp/turboquant/reports/baseline_generate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from qai_hub_models.models.templates.lm_driver.utils.rope_embedding import (
    RopeEmbedding,
)

DEFAULT_SDK = Path("~/qairt/2.48.0.260626").expanduser()
DEFAULT_ADB = Path("/mnt/c/adb/adb.exe")
DEFAULT_RUNNER = Path(
    "~/.qaihm/tmp/turboquant/qnn_runner/android-arm64/qnn-llm-runner"
).expanduser()
DEVICE_ROOT = "/data/local/tmp/qaihm_turboquant/llm"
DEVICE_LIBS = (
    "lib/aarch64-android/libQnnHtp.so",
    "lib/aarch64-android/libQnnHtpV81Stub.so",
    "lib/aarch64-android/libQnnSystem.so",
    "lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so",
)
PROMPT = "What is gravity? Keep the answer under ten words."


def windows_path(path: Path) -> str:
    return subprocess.run(
        ["wslpath", "-w", str(path)], check=True, capture_output=True, text=True
    ).stdout.strip()


def adb(args: argparse.Namespace, *cmd: str, check: bool = True) -> str:
    result = subprocess.run(
        [str(args.adb), *cmd], capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"adb {' '.join(cmd)} failed: {result.stderr}{result.stdout}"
        )
    return result.stdout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def push_if_changed(args: argparse.Namespace, local: Path, remote: str) -> bool:
    remote_hash = adb(
        args, "shell", f"sha256sum {remote} 2>/dev/null", check=False
    ).split()
    if remote_hash and remote_hash[0] == sha256_file(local):
        return False
    adb(args, "push", windows_path(local), remote)
    return True


def cmd_assets(args: argparse.Namespace) -> None:
    out: Path = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint_dir.expanduser()
    config = AutoConfig.from_pretrained(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    rope = RopeEmbedding(
        model=SimpleNamespace(config=config), context_length=args.context_length
    )
    cos = rope.cos[0, 0].numpy().astype(np.float32)
    sin = rope.sin[0, 0].numpy().astype(np.float32)
    (out / f"rope_cl{args.context_length}.bin").write_bytes(
        cos.tobytes() + sin.tobytes()
    )

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": PROMPT},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=True
    )
    if not isinstance(prompt_ids, list):
        prompt_ids = list(prompt_ids["input_ids"])
    np.asarray(prompt_ids, dtype=np.int32).tofile(out / "prompt_ids.bin")

    path = hf_hub_download(
        repo_id="Salesforce/wikitext",
        repo_type="dataset",
        filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
    )
    text = "\n\n".join(pd.read_parquet(path)["text"].tolist())
    all_ids = tokenizer(text, add_special_tokens=False).input_ids
    windows = []
    for w in range(args.num_windows):
        window = np.asarray(
            all_ids[w * args.context_length : (w + 1) * args.context_length],
            dtype=np.int32,
        )
        window.tofile(out / f"wikitext_w{w}.bin")
        windows.append(f"wikitext_w{w}.bin")
    # Prompt + 128 greedy tokens fill the cache exactly to the context length.
    boundary = f"boundary_prompt_cl{args.context_length}.bin"
    np.asarray(all_ids[: args.context_length - 127], dtype=np.int32).tofile(
        out / boundary
    )
    manifest = {
        "checkpoint_dir": str(ckpt),
        "context_length": args.context_length,
        "rope": f"rope_cl{args.context_length}.bin",
        "rope_half": int(cos.shape[1]),
        "prompt": PROMPT,
        "prompt_tokens": len(prompt_ids),
        "prompt_ids": "prompt_ids.bin",
        "wikitext_windows": windows,
        "boundary_prompt": boundary,
        "sha256": {
            p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.suffix == ".bin"
        },
    }
    (out / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, indent=2))


def cmd_push(args: argparse.Namespace) -> None:
    bundle = args.bundle_dir.expanduser()
    bins = sorted(
        bundle.glob("part*_of_*.bin"),
        key=lambda p: int(re.search(r"part(\d+)", p.name).group(1)),
    )
    if not bins:
        raise FileNotFoundError(f"No part*_of_*.bin in {bundle}")
    remote_bundle = f"{DEVICE_ROOT}/bundles/{args.name}"
    adb(args, "shell", "mkdir", "-p", f"{DEVICE_ROOT}/bin", remote_bundle)
    for rel in DEVICE_LIBS:
        push_if_changed(args, args.sdk / rel, f"{DEVICE_ROOT}/bin/{Path(rel).name}")
    push_if_changed(args, args.runner.expanduser(), f"{DEVICE_ROOT}/bin/qnn-llm-runner")
    adb(args, "shell", "chmod", "+x", f"{DEVICE_ROOT}/bin/qnn-llm-runner")
    manifest = {"name": args.name, "bundle_dir": str(bundle), "bins": []}
    for local in bins:
        changed = push_if_changed(args, local, f"{remote_bundle}/{local.name}")
        manifest["bins"].append(
            {"name": local.name, "sha256": sha256_file(local), "pushed": changed}
        )
        print(f"{local.name}: {'pushed' if changed else 'up to date'}", flush=True)
    conversion = bundle / "convert_report.json"
    metadata = json.loads(conversion.read_text()) if conversion.exists() else {}
    runtime_metadata = {
        k: metadata[k]
        for k in ("context_length", "context_buckets", "config_hash", "config")
        if k in metadata
    }
    if native := metadata.get("native_decoder"):
        lib = native["libraries"]["hexagon-v81"]
        local = Path(lib["path"])
        if sha256_file(local) != lib["sha256"]:
            raise ValueError(
                "Native package changed since conversion; rebuild the bundle"
            )
        push_if_changed(args, local, f"{remote_bundle}/{local.name}")
        runtime_metadata["native_decoder"] = {
            "package": native["package"],
            "interface": native["interface"],
            "library": local.name,
            "sha256": lib["sha256"],
        }
    runtime = bundle / "runtime_manifest.json"
    runtime.write_text(
        json.dumps(
            runtime_metadata,
            indent=2,
        )
        + "\n"
    )
    push_if_changed(args, runtime, f"{remote_bundle}/runtime_manifest.json")
    (bundle / f"device_push_{args.name}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def cmd_run(args: argparse.Namespace) -> None:
    assets_dir = args.assets.expanduser()
    assets = json.loads((assets_dir / "assets.json").read_text())
    remote_assets = f"{DEVICE_ROOT}/assets/cl{assets['context_length']}"
    adb(args, "shell", "mkdir", "-p", remote_assets, f"{DEVICE_ROOT}/reports")
    for name in assets["sha256"]:
        push_if_changed(args, assets_dir / name, f"{remote_assets}/{name}")

    remote_bundle = f"{DEVICE_ROOT}/bundles/{args.name}"
    listing = adb(args, "shell", f"ls {remote_bundle}").split()
    runtime = (
        json.loads(adb(args, "shell", "cat", f"{remote_bundle}/runtime_manifest.json"))
        if "runtime_manifest.json" in listing
        else {}
    )
    if (
        runtime.get("context_length", assets["context_length"])
        != assets["context_length"]
    ):
        raise ValueError("Bundle and assets context lengths differ.")
    bins = sorted((b for b in listing if re.fullmatch(r"part\d+_of_\d+\.bin", b)),
                  key=lambda b: int(re.search(r"part(\d+)", b).group(1)))  # fmt: skip
    tokens = args.tokens or (
        assets["prompt_ids"]
        if args.mode == "generate"
        else assets["wikitext_windows"][0]
    )
    tag = args.report.stem
    remote_report = f"{DEVICE_ROOT}/reports/{tag}.json"
    runner_args = [
        f"{DEVICE_ROOT}/bin/qnn-llm-runner",
        "--backend libQnnHtp.so --system libQnnSystem.so",
        "--bins " + ",".join(f"{remote_bundle}/{b}" for b in bins),
        f"--context-length {assets['context_length']}",
        f"--rope {remote_assets}/{assets['rope']} --rope-half {assets['rope_half']}",
        f"--mode {args.mode} --tokens {remote_assets}/{tokens}",
        f"--n-gen {args.n_gen} --report {remote_report}",
    ]
    native = runtime.get("native_decoder")
    if native:
        package_path = f"{remote_bundle}/{native['library']}"
        digest = adb(args, "shell", "sha256sum", package_path).split()[0]
        if digest != native["sha256"]:
            raise ValueError("Device native decoder does not match the context bundle")
        runner_args.append(
            f"--op-package {package_path} --op-package-provider {native['interface']}"
        )
    if args.stop_on_eos:
        runner_args.append("--stop-on-eos")
    buckets = args.context_buckets or runtime.get("context_buckets", [])
    if buckets:
        buckets = sorted({*buckets, assets["context_length"]})
        available = set(runtime.get("context_buckets", [assets["context_length"]]))
        if runtime and not set(buckets).issubset(available):
            raise ValueError(
                "Requested context buckets are not present in this bundle."
            )
    if buckets:
        runner_args.append("--context-buckets " + ",".join(map(str, buckets)))
    if args.graph_suffix:
        runner_args.append(f"--graph-suffix {args.graph_suffix}")
    if args.dump_logits:
        runner_args.append(f"--dump-logits {DEVICE_ROOT}/reports/{tag}.logits.bin")
    if args.profile_decode_step >= 0:
        runner_args.append(f"--profile-decode-step {args.profile_decode_step}")
    if args.profile_prefill:
        runner_args.append("--profile-prefill")
    if args.sessions > 1:
        runner_args.append(f"--sessions {args.sessions}")
    script = (
        f"cd {DEVICE_ROOT}/bin && export LD_LIBRARY_PATH={DEVICE_ROOT}/bin:/vendor/lib64 "
        f"&& export ADSP_LIBRARY_PATH='{remote_bundle}:{DEVICE_ROOT}/bin:/vendor/dsp/cdsp:/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp' "
        f"&& {' '.join(runner_args)} 2>&1; echo __RC__$?"
    )
    out = adb(args, "shell", script, check=False)
    rc = re.search(r"__RC__(\d+)", out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.with_suffix(".log").write_text(out)
    if not rc or rc.group(1) != "0":
        raise RuntimeError(f"runner failed; log in {args.report.with_suffix('.log')}")
    adb(args, "pull", remote_report, windows_path(args.report.parent))
    if args.dump_logits:
        adb(
            args,
            "pull",
            f"{DEVICE_ROOT}/reports/{tag}.logits.bin",
            windows_path(args.report.parent),
        )
    report = json.loads(args.report.read_text())
    report["device"] = {
        "soc_model": adb(args, "shell", "getprop", "ro.soc.model").strip(),
        "fingerprint": adb(args, "shell", "getprop", "ro.build.fingerprint").strip(),
    }
    report["bundle_name"] = args.name
    report["context_buckets"] = buckets or [assets["context_length"]]
    if native:
        report["native_decoder"] = native
    if runtime.get("config_hash"):
        report["config_hash"] = runtime["config_hash"]
    report["assets"] = {
        "tokens_file": tokens,
        "tokens_sha256": sha256_file(assets_dir / tokens),
        **{k: assets[k] for k in ("context_length", "prompt", "prompt_tokens")},
    }
    if args.mode == "generate":
        tokenizer = AutoTokenizer.from_pretrained(assets["checkpoint_dir"])
        report["generated_text"] = tokenizer.decode(report["generated"])
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: report.get(k) for k in ("mode", "ttft_s", "prefill_tok_per_s", "decode_tok_per_s", "ppl",
                                          "scored_tokens", "stop_reason", "generated_text", "load_s")}  # fmt: skip
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for p in (parser,):
        p.add_argument("--adb", type=Path, default=DEFAULT_ADB)
        p.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    assets = sub.add_parser("assets")
    assets.add_argument("--checkpoint-dir", type=Path, required=True)
    assets.add_argument("--context-length", type=int, default=1024)
    assets.add_argument("--num-windows", type=int, default=4)
    assets.add_argument("--out", type=Path, required=True)
    push = sub.add_parser("push")
    push.add_argument("--bundle-dir", type=Path, required=True)
    push.add_argument("--name", required=True)
    push.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    run = sub.add_parser("run")
    run.add_argument("--name", required=True)
    run.add_argument("--assets", type=Path, required=True)
    run.add_argument("--mode", choices=["generate", "score"], default="generate")
    run.add_argument("--tokens", default="")
    run.add_argument("--n-gen", type=int, default=128)
    run.add_argument("--stop-on-eos", action="store_true")
    run.add_argument("--graph-suffix", default="")
    run.add_argument("--context-buckets", type=int, nargs="+", default=[])
    run.add_argument("--dump-logits", action="store_true")
    run.add_argument("--profile-decode-step", type=int, default=-1)
    run.add_argument("--profile-prefill", action="store_true")
    run.add_argument("--sessions", type=int, default=1)
    run.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.sdk = args.sdk.expanduser()
    {"assets": cmd_assets, "push": cmd_push, "run": cmd_run}[args.stage](args)


if __name__ == "__main__":
    main()
