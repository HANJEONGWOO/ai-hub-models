# SPDX-License-Identifier: BSD-3-Clause
"""Freeze document-disjoint tokens, capture actual W4A16 HTP FP16-KV Q/K/V/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

SCRIPTS = Path(__file__).resolve().parent
DATA_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def digest(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: str | Path, data: Any) -> None:
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2)


def experiment_name(root: Path) -> str:
    return "krotation_" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]


def run_logged(command: list[str], path: Path) -> None:
    print("START", path, flush=True)
    with path.open("x") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    print("DONE", path, flush=True)


def prepare(root: Path, source_assets: Path) -> None:
    out = root / "assets"
    out.mkdir(parents=True, exist_ok=False)
    base = json.loads((source_assets / "assets.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        base["checkpoint_dir"], local_files_only=True
    )
    rng = np.random.default_rng(20261006)
    used = set()
    windows = {}
    datasets = {}
    for split, count in (("train", 4), ("validation", 2), ("test", 4)):
        path = hf_hub_download(
            "Salesforce/wikitext",
            f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=DATA_REVISION,
        )
        datasets[split] = {"path": path, "sha256": digest(path)}
        documents, title, lines = [], None, []
        for line in pd.read_parquet(path)["text"]:
            if re.fullmatch(r"\s*= [^=]+ =\s*", line):
                if title is not None:
                    documents.append((title, "\n".join(lines)))
                title, lines = line.strip(), [line]
            elif title is not None:
                lines.append(line)
        if title is not None:
            documents.append((title, "\n".join(lines)))
        chosen = []
        for index in rng.permutation(len(documents)):
            title, text = documents[index]
            title_hash = hashlib.sha256(title.encode()).hexdigest()
            if title_hash in used:
                continue
            tokens = tokenizer(text, add_special_tokens=False).input_ids
            if len(tokens) < 1024:
                continue
            start = int(rng.integers(0, len(tokens) - 1024 + 1))
            name = f"{split}_{len(chosen)}.bin"
            np.asarray(tokens[start : start + 1024], dtype="<i4").tofile(out / name)
            used.add(title_hash)
            chosen.append(
                {
                    "file": name,
                    "title": title,
                    "document_id": title_hash,
                    "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "offset": start,
                    "tokens_sha256": digest(out / name),
                }
            )
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError("Insufficient disjoint documents")
        windows[split] = chosen
    for name in (base["rope"], base["prompt_ids"], base["boundary_prompt"]):
        shutil.copy2(source_assets / name, out / name)
    base["wikitext_windows"] = [w["file"] for w in windows["test"]]
    base["sha256"] = {p.name: digest(p) for p in sorted(out.glob("*.bin"))}
    write_json(out / "assets.json", base)
    write_json(
        root / "data_manifest.json",
        {
            "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
            "revision": DATA_REVISION,
            "seed": 20261006,
            "context": 1024,
            "chunk": 128,
            "datasets": datasets,
            "windows": windows,
            "selection": "fixed document-disjoint 4 train / 2 validation / 4 heldout test",
            "source": "actual W4A16 HTP FP16-KV path, not HF float-model activations",
        },
    )


def capture(root: Path, splits: list[str] | None = None) -> None:
    manifest = json.loads((root / "data_manifest.json").read_text())
    split = Path("/mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_w4a16_split")
    sources = {}
    for part in json.loads((split / "split_manifest.json").read_text())[
        "parts"
    ].values():
        directory = Path(part["bundle_dir"])
        for file in sorted(directory.iterdir()):
            if file.suffix in (".onnx", ".data", ".encodings"):
                sources[str(file.resolve())] = digest(file)
    write_json(
        root / "capture_identity.json",
        {
            "source_files_sha256": sources,
            "split_manifest_sha256": digest(split / "split_manifest.json"),
            "conversion_sha256": digest(root / "capture_bundle/convert_report.json"),
            "runner_sha256": digest(root / "runner/qnn-llm-runner"),
            "bins_sha256": {
                p.name: digest(p)
                for p in sorted((root / "capture_bundle").glob("part*_of_4.bin"))
            },
        },
    )
    run_logged(
        [
            sys.executable,
            str(SCRIPTS / "run_device_llm.py"),
            "push",
            "--name",
            experiment_name(root) + "_capture",
            "--bundle-dir",
            str(root / "capture_bundle"),
            "--runner",
            str(root / "runner/qnn-llm-runner"),
        ],
        root / "reports/capture_push.log",
    )
    for split_name, windows in manifest["windows"].items():
        if splits is not None and split_name not in splits:
            continue
        for window in windows:
            tag = Path(window["file"]).stem
            run_logged(
                [
                    sys.executable,
                    str(SCRIPTS / "run_device_llm.py"),
                    "run",
                    "--name",
                    experiment_name(root) + "_capture",
                    "--assets",
                    str(root / "assets"),
                    "--mode",
                    "score",
                    "--tokens",
                    window["file"],
                    "--context-buckets",
                    "1024",
                    "--capture-attention-out",
                    str(root / "captures" / tag),
                    "--report",
                    str(root / "reports" / (tag + "_capture.json")),
                    "--remote-report-tag",
                    experiment_name(root) + "_" + tag,
                ],
                root / "reports" / (tag + "_capture.stdout.log"),
            )


def pack(root: Path, splits: list[str] | None = None) -> None:
    manifest = json.loads((root / "data_manifest.json").read_text())
    out = root / "samples"
    out.mkdir(exist_ok=False)
    encodings = {}
    for path in sorted((root / "capture_bundle").glob("prompt*.kv_edits.json")):
        metadata = json.loads(path.read_text())
        if not metadata.get("fp16_attention"):
            continue
        model = onnx.load(
            str(path).replace(".kv_edits.json", ".onnx"), load_external_data=False
        )
        nodes = {o: n for n in model.graph.node for o in n.output}
        acts = {
            e["name"]: e
            for e in json.loads(
                Path(str(path).replace(".kv_edits.json", ".encodings")).read_text()
            )["activation_encodings"]
        }
        for item in metadata["fp16_attention"]:
            prob = nodes[item["av"]["lhs"]].input[0]
            masked = nodes[prob].input[0]
            if (
                nodes[prob].op_type != "Softmax"
                or nodes[masked].op_type != "Add"
                or "key_scaled" in item["qk"]["rhs"]
            ):
                raise ValueError("Unsupported captured attention pattern")
            encodings.setdefault(str(item["layer"]), []).append(
                {
                    "head": item["head"],
                    "group": item["group"],
                    "score": acts[item["qk"]["boundary"]],
                    "masked": acts[masked],
                    "prob": acts[prob],
                }
            )
    for split_name, windows in manifest["windows"].items():
        if splits is not None and split_name not in splits:
            continue
        for window in windows:
            tag = Path(window["file"]).stem
            captured = root / "captures" / tag
            pieces = {i: {k: [] for k in ("q", "k", "v", "o")} for i in range(28)}
            for start in range(0, 1024, 128):
                data = json.loads((captured / f"s0_t{start}.json").read_text())
                if (data["ar"], data["new_tokens"], data["cached_after"]) != (
                    128,
                    128,
                    start + 128,
                ):
                    raise ValueError("Capture context/reset mismatch")
                for layer in pieces:
                    for kind, name in (
                        ("q", f"capture_attn_{layer}_query"),
                        ("o", f"capture_attn_{layer}_output"),
                        ("k", f"past_key_{layer}_out"),
                        ("v", f"past_value_{layer}_out"),
                    ):
                        item = data["tensors"][name]
                        x = np.fromfile(captured / item["file"], dtype="<f4").reshape(
                            item["shape"]
                        )[:, 0]
                        if kind == "k":
                            x = x.swapaxes(-1, -2)
                        if not np.isfinite(x).all():
                            raise ValueError("Nonfinite captured tensor")
                        pieces[layer][kind].append(x.astype(np.float16))
            for layer, fields in pieces.items():
                np.savez(
                    out / f"{tag}_layer{layer:02d}.npz",
                    **{k: np.concatenate(v, axis=1) for k, v in fields.items()},
                )
    for heads in encodings.values():
        heads.sort(key=lambda x: (x["head"], x["group"]))
    write_json(out / "encodings.json", encodings)
    write_json(
        out / "manifest.json",
        {
            "data_manifest_sha256": digest(root / "data_manifest.json"),
            "capture_identity_sha256": digest(root / "capture_identity.json"),
            "capture_conversion_sha256": digest(
                root / "capture_bundle/convert_report.json"
            ),
            "sha256": {p.name: digest(p) for p in sorted(out.glob("*.npz"))},
            "encodings_sha256": digest(out / "encodings.json"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "capture", "pack"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"))
    parser.add_argument(
        "--source-assets",
        type=Path,
        default=Path("/mnt/d/ai-hub-models/binaries/turboquant/device_assets_cl1024"),
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.root, args.source_assets)
    else:
        globals()[args.stage](args.root, args.splits)


if __name__ == "__main__":
    main()
