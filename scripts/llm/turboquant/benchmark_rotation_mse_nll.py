# SPDX-License-Identifier: BSD-3-Clause
"""Single-pass validation NLL for immutable A/B/P, reusing saved Attention MSE."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_key_rotation import read, run
from benchmark_rotation_followup import identity as followup_identity
from rotation_data import SCRIPTS, digest, experiment_name, run_logged, write_json
from run_device_llm import DEFAULT_ADB, DEFAULT_SDK, DEVICE_LIBS, DEVICE_ROOT
from transformers import AutoTokenizer

GROUPS = ("A", "B", "P")
TITLES = (
    "= Slammiversary ( 2008 ) =",
    "= Sorry ( Madonna song ) =",
    "= Meridian , Mississippi =",
    "= Fort Scott National Historic Site =",
)
PROTOCOL = {
    "model": "Qwen3-1.7B W4A16",
    "groups": list(GROUPS),
    "context_length": 1024,
    "scored_tokens_per_document": 1023,
    "repeats": 1,
    "document_orders": ["ABP", "BPA", "PAB", "ABP"],
    "measurement": "HTP teacher-forced score; existing runner; no profiling",
    "mse": "Reuse cpu_quality.json groups A/B/P on the same validation windows",
    "restrictions": "No selection, training, build, calibration, codebook/default changes, test inputs, or automatic retries",
    "analysis": "Paired NLL deltas and aggregate direction; no inferential correlation from three configurations",
}


def shell(*args: str) -> str:
    return subprocess.run(
        [str(DEFAULT_ADB), "shell", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_documents(old: Path) -> list[dict]:
    """Validate pinned text, exact offsets and re-tokenization, validation only."""
    manifest = read(old / "data_manifest.json")
    windows = manifest["windows"]["validation"]
    if manifest["context"] != 1024 or tuple(w["title"] for w in windows) != TITLES:
        raise ValueError("Unexpected validation documents/context")
    dataset = manifest["datasets"]["validation"]
    if digest(dataset["path"]) != dataset["sha256"]:
        raise ValueError("Pinned validation dataset changed")
    documents, title, lines = {}, None, []
    for line in pd.read_parquet(dataset["path"])["text"]:
        if re.fullmatch(r"\s*= [^=]+ =\s*", line):
            if title is not None:
                documents[title] = "\n".join(lines)
            title, lines = line.strip(), [line]
        elif title is not None:
            lines.append(line)
    if title is not None:
        documents[title] = "\n".join(lines)
    tokenizer = AutoTokenizer.from_pretrained(
        read(old / "assets/assets.json")["checkpoint_dir"], local_files_only=True
    )
    for i, window in enumerate(windows):
        text = documents[window["title"]]
        if (
            window["file"] != f"validation_{i}.bin"
            or hashlib.sha256(text.encode()).hexdigest() != window["document_sha256"]
            or hashlib.sha256(window["title"].encode()).hexdigest()
            != window["document_id"]
        ):
            raise ValueError("Document identity changed")
        offset = window["offset"]
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if offset < 0 or offset + 1024 > len(ids):
            raise ValueError("Invalid frozen token offset")
        expected = np.asarray(ids[offset : offset + 1024], dtype="<i4").tobytes()
        actual = (old / "assets" / window["file"]).read_bytes()
        if (
            expected != actual
            or hashlib.sha256(actual).hexdigest() != window["tokens_sha256"]
        ):
            raise ValueError("Frozen tokens/offset/tokenizer mismatch")
    return windows


def freeze(root: Path, old: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    (root / "reports").mkdir()
    write_json(root / "protocol.json", PROTOCOL)
    previous = read(old / "reports/experiment.json")
    if followup_identity(old) != previous:
        raise ValueError("Original binaries, source identity, config or device changed")
    windows = verify_documents(old)
    assets = read(old / "assets/assets.json")
    (root / "assets").mkdir()
    names = [w["file"] for w in windows] + [assets["rope"]]
    for name in names:
        (root / "assets" / name).symlink_to((old / "assets" / name).resolve())
    assets["sha256"] = {n: digest(root / "assets" / n) for n in names}
    assets["wikitext_windows"] = [w["file"] for w in windows]
    assets["prompt_ids"] = windows[0]["file"]
    assets["prompt"] = "Frozen validation scoring only; no generation prompt"
    assets["prompt_tokens"] = 1024
    assets.pop("boundary_prompt", None)
    write_json(root / "assets/assets.json", assets)
    manifest = read(old / "data_manifest.json")
    write_json(
        root / "data_manifest.json",
        {
            "source_manifest_sha256": digest(old / "data_manifest.json"),
            "dataset": manifest["dataset"],
            "revision": manifest["revision"],
            "context": 1024,
            "windows": {"validation": windows},
            "document_offset_retokenization_verified": True,
            "heldout_inputs_included": False,
        },
    )
    (root / "runner").mkdir()
    (root / "runner/qnn-llm-runner").symlink_to(
        (old / "runner/qnn-llm-runner").resolve()
    )
    for group in GROUPS:
        (root / group).mkdir()
        for name in ["convert_report.json"] + [
            f"part{i}_of_4.bin" for i in range(1, 5)
        ]:
            (root / group / name).symlink_to((old / group / name).resolve())
    preserved = set(old.glob("reports/*.json"))
    preserved.update(
        old / name
        for name in (
            "protocol.json",
            "source_identity.json",
            "capture_identity.json",
            "data_manifest.json",
            "cpu_quality.json",
            "rotation_validation.json",
            "layer_selection.json",
            "assets/assets.json",
            "runner/qnn-llm-runner",
            "reports/extra_validation_layers.csv",
        )
    )
    for group in GROUPS:
        preserved.add(old / f"rotations/{group}.json")
        preserved.add(old / group / "convert_report.json")
        preserved.update(old / group / f"part{i}_of_4.bin" for i in range(1, 5))
    write_json(
        root / "source_identity.json",
        {
            "root": str(old.resolve()),
            "sha256": {str(p.relative_to(old)): digest(p) for p in sorted(preserved)},
        },
    )
    write_json(
        root / "attention_mse.json",
        {
            "source_sha256": digest(old / "cpu_quality.json"),
            "groups": {g: read(old / "cpu_quality.json")["groups"][g] for g in GROUPS},
            "recomputed": False,
            "scope": "Four documents, 28 layers, 32 positions per window (31,63,...1023), all heads, fixed FP16-KV captures",
            "document_mse_available": False,
        },
    )
    files = [
        Path(__file__),
        SCRIPTS / "benchmark_key_rotation.py",
        SCRIPTS / "run_device_llm.py",
        SCRIPTS / "rotation_data.py",
    ]
    repo = SCRIPTS.parents[2]
    write_json(
        root / "reports/experiment.json",
        {
            **previous,
            "source_experiment_sha256": digest(old / "reports/experiment.json"),
            "protocol_sha256": digest(root / "protocol.json"),
            "data_manifest_sha256": digest(root / "data_manifest.json"),
            "assets_sha256": assets["sha256"],
            "source_root": str(old.resolve()),
            "source_preservation_sha256": digest(root / "source_identity.json"),
            "attention_mse_sha256": digest(root / "attention_mse.json"),
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=repo, text=True
            ).strip(),
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "code_sha256": {str(p): digest(p) for p in files},
            "runtime_sha256": {
                str(DEFAULT_SDK / p): digest(DEFAULT_SDK / p) for p in DEVICE_LIBS
            },
        },
    )
    print("FROZEN: validation only, 12 single-pass runs", root, flush=True)


def verify_local(root: Path) -> None:
    identity = read(root / "reports/experiment.json")
    if read(root / "protocol.json") != PROTOCOL:
        raise ValueError("Protocol changed")
    for name, field in (
        ("protocol.json", "protocol_sha256"),
        ("data_manifest.json", "data_manifest_sha256"),
        ("source_identity.json", "source_preservation_sha256"),
        ("attention_mse.json", "attention_mse_sha256"),
    ):
        if digest(root / name) != identity[field]:
            raise ValueError(f"Frozen identity changed: {name}")
    source = read(root / "source_identity.json")
    for name, expected in source["sha256"].items():
        if digest(Path(source["root"]) / name) != expected:
            raise ValueError(f"Preserved source changed: {name}")
    assets = read(root / "assets/assets.json")
    if assets["sha256"] != identity["assets_sha256"]:
        raise ValueError("Asset identity changed")
    for name, expected in identity["assets_sha256"].items():
        if digest(root / "assets" / name) != expected:
            raise ValueError(f"Asset changed: {name}")
    for field in ("code_sha256", "runtime_sha256"):
        for name, expected in identity[field].items():
            if digest(name) != expected:
                raise ValueError(f"Implementation/runtime changed: {name}")


def device_hashes(root: Path) -> dict:
    identity = read(root / "reports/experiment.json")
    expected = {f"{DEVICE_ROOT}/bin/qnn-llm-runner": identity["runner_sha256"]}
    for path, checksum in identity["runtime_sha256"].items():
        expected[f"{DEVICE_ROOT}/bin/{Path(path).name}"] = checksum
    for group in GROUPS:
        remote = f"{DEVICE_ROOT}/bundles/{experiment_name(root)}_{group}"
        expected.update(
            {
                f"{remote}/{n}": h
                for n, h in identity["groups"][group]["bins_sha256"].items()
            }
        )
        expected[f"{remote}/runtime_manifest.json"] = digest(
            root / group / "runtime_manifest.json"
        )
        native = read(root / group / "runtime_manifest.json")["native_decoder"]
        expected[f"{remote}/{native['library']}"] = native["sha256"]
    actual = {}
    for line in shell("sha256sum", *expected).splitlines():
        checksum, name = line.split(maxsplit=1)
        actual[name.strip()] = checksum
    if actual != expected:
        raise ValueError("Device executable/library/binary/config hash mismatch")
    return actual


def push(root: Path) -> None:
    verify_local(root)
    for group in GROUPS:
        run_logged(
            [
                sys.executable,
                str(SCRIPTS / "run_device_llm.py"),
                "push",
                "--name",
                experiment_name(root) + "_" + group,
                "--bundle-dir",
                str(root / group),
                "--runner",
                str(root / "runner/qnn-llm-runner"),
            ],
            root / f"reports/{group}_push.log",
        )
    write_json(root / "reports/device_before.json", device_hashes(root))


def validate_score(data: dict, window: dict) -> None:
    if (
        data["mode"] != "score"
        or data["context_length"] != 1024
        or data["prompt_tokens"] != 1024
        or data["scored_tokens"] != 1023
        or data["assets"]["tokens_file"] != window["file"]
        or data["assets"]["tokens_sha256"] != window["tokens_sha256"]
        or not data["quantize_current_kv"]
        or data.get("diagnostic_only")
        or data["context_buckets"] != [1024]
        or "htp" not in data["backend_api"]
    ):
        raise ValueError("Wrong input, scoring policy or execution conditions")
    nll = data["nll_sum"]
    if (
        not math.isfinite(nll)
        or nll < 0
        or not math.isclose(data["ppl"], math.exp(nll / 1023), rel_tol=1e-7)
    ):
        raise ValueError("Inconsistent/nonfinite NLL/PPL")


def score(root: Path) -> None:
    verify_local(root)
    if device_hashes(root) != read(root / "reports/device_before.json"):
        raise ValueError("Device state changed after push")
    windows = read(root / "data_manifest.json")["windows"]["validation"]
    for i, (window, order) in enumerate(
        zip(windows, PROTOCOL["document_orders"], strict=True)
    ):
        for group in order:
            data = run(
                root,
                group,
                f"{group}_validation_w{i}",
                [
                    "--mode",
                    "score",
                    "--tokens",
                    window["file"],
                ],
            )
            validate_score(data, window)
            print(
                "SCORED",
                group,
                window["title"],
                data["nll_sum"] / 1023,
                data["ppl"],
                flush=True,
            )
    write_json(root / "reports/device_after.json", device_hashes(root))
    verify_local(root)
    write_json(
        root / "reports/preservation.json",
        {
            "original_files_unchanged": True,
            "local_and_device_hashes_passed": True,
            "attempts": 12,
            "repeats_per_document_group": 1,
            "builds": 0,
            "mse_reruns": 0,
            "heldout_input_runs": 0,
        },
    )


def aggregate(reports: list[dict]) -> dict:
    count = sum(r["scored_tokens"] for r in reports)
    nll = math.fsum(r["nll_sum"] for r in reports)
    if count <= 0 or not math.isfinite(nll) or nll < 0:
        raise ValueError("Invalid NLL aggregation")
    return {
        "scored_tokens": count,
        "nll_sum": nll,
        "mean_nll": nll / count,
        "ppl": math.exp(nll / count),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path) -> None:
    if not read(root / "reports/preservation.json")["original_files_unchanged"]:
        raise ValueError("Required successful measurement/preservation audit missing")
    windows = read(root / "data_manifest.json")["windows"]["validation"]
    mse = read(root / "attention_mse.json")["groups"]
    identity = read(root / "reports/experiment.json")
    rows, documents, by_group = [], [], {g: [] for g in GROUPS}
    for i, window in enumerate(windows):
        item = {
            "window": i,
            "title": window["title"],
            "offset": window["offset"],
            "tokens_sha256": window["tokens_sha256"],
        }
        document_identity = dict(item)
        for group in GROUPS:
            path = root / f"reports/{group}_validation_w{i}.json"
            report = read(path)
            validate_score(report, window)
            if report["config_hash"] != identity["groups"][group]["config_hash"]:
                raise ValueError("Report configuration mismatch")
            summary = aggregate([report])
            by_group[group].append(report)
            rows.append(
                {
                    **document_identity,
                    "group": group,
                    **summary,
                    "report_sha256": digest(path),
                }
            )
            for key, value in summary.items():
                item[f"{group}_{key}"] = value
        for group, baseline in (("B", "A"), ("P", "A"), ("P", "B")):
            delta = item[f"{group}_mean_nll"] - item[f"{baseline}_mean_nll"]
            item[f"{group}_minus_{baseline}_mean_nll"] = delta
        documents.append(item)
    summary_rows = [
        {"group": g, "attention_mse": mse[g]["mse"], **aggregate(by_group[g])}
        for g in GROUPS
    ]
    groups = {row["group"]: row for row in summary_rows}
    pairs = {}
    for group, baseline in (("B", "A"), ("P", "A"), ("P", "B")):
        a, b = groups[baseline], groups[group]
        pairs[f"{group}_vs_{baseline}"] = {
            "mse_relative_change": b["attention_mse"] / a["attention_mse"] - 1,
            "mean_nll_delta": b["mean_nll"] - a["mean_nll"],
            "ppl_relative_change": b["ppl"] / a["ppl"] - 1,
            "documents_nll_improved": sum(
                d[f"{group}_minus_{baseline}_mean_nll"] < 0 for d in documents
            ),
        }
    layer_rows = []
    for layer in range(28):
        row = {
            "layer": layer,
            **{f"{g}_mse": mse[g]["layers"][str(layer)]["mse"] for g in GROUPS},
        }
        row["P_minus_B_mse_contribution"] = (row["P_mse"] - row["B_mse"]) / 28
        row["B_absolute_mse_share"] = row["B_mse"] / (28 * mse["B"]["mse"])
        layer_rows.append(row)
    result = {
        "protocol": PROTOCOL,
        "groups": groups,
        "documents": documents,
        "pairs": pairs,
        "layers": layer_rows,
        "limitations": [
            "MSE aggregates 32 query positions/window; NLL scores all 1023 next-token targets/window.",
            "Local MSE uses frozen FP16-KV layer inputs; full TQ inference propagates upstream errors.",
            "Per-document Attention MSE was not saved; no documentwise MSE-NLL coefficient or layer-NLL causal attribution.",
            "Three fixed configurations and four selected validation windows do not establish population correlation or causality.",
            "Existing encoder numerical validation failures remain; no new encoder/probe evaluation.",
            "Validation was previously observed for P promotion; this is not a new heldout evaluation.",
        ],
    }
    write_json(root / "reports/comparison.json", result)
    write_json(root / "reports/document_nll.json", rows)
    write_csv(root / "reports/document_nll.csv", rows)
    write_csv(root / "reports/document_deltas.csv", documents)
    write_csv(root / "reports/summary.csv", summary_rows)
    write_csv(root / "reports/layer_mse_contributions.csv", layer_rows)
    print(json.dumps({"groups": groups, "pairs": pairs}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "push", "score", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.source is None:
            parser.error("freeze requires --source")
        freeze(args.root.resolve(), args.source.resolve())
    else:
        globals()[args.stage](args.root.resolve())


if __name__ == "__main__":
    main()
