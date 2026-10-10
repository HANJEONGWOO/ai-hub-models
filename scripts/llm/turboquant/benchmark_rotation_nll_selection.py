# SPDX-License-Identifier: BSD-3-Clause
"""Build/audit only missing shared rotations, then evaluate a frozen NLL protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnx
from benchmark_key_rotation import PACKAGE, SPLIT, compiled_signature, read, snapshot
from benchmark_rotation_followup import graph_signature, normalized_name
from benchmark_rotation_mse_nll import aggregate, shell, validate_score
from onnx import numpy_helper
from rotation_data import SCRIPTS, digest, experiment_name, run_logged, write_json
from rotation_nll_selection import (
    PROTOCOL,
    SEEDS,
    choose_seed,
    fold_analysis,
    paired_bootstrap,
    verify_frozen,
    write_csv,
)
from run_device_llm import DEFAULT_SDK, DEVICE_LIBS, DEVICE_ROOT
from summarize_native_results import performance as performance_result
from verify_rotated_attention import verify_graph

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.rotation_artifact import (
    with_key_rotation,
)


def normalize_rms_display_name(name: str) -> str:
    # QNN may append a second generated numeric suffix after a name collision.
    # Apply only to operation display names, never tensor names or graph edges.
    return re.sub(
        r"^rms_norm_node___\d+(?:__\d+)*$", "rms_norm_node___GENERATED_ID", name
    )


def compiled_audit_signature(path: Path) -> dict:
    result = compiled_signature(path)
    for op in result["ops"]:
        op["name"] = normalize_rms_display_name(op["name"])
    return result


def build(root: Path) -> None:
    verify_frozen(root, sources=True)
    missing = [
        s for s in SEEDS if not read(root / "candidates.json")[str(s)]["source_bundle"]
    ]

    def candidate(seed: int) -> None:
        out = root / f"builds/seed{seed}"
        if out.exists() or (root / f"seed{seed}").exists():
            raise FileExistsError("Preserve existing build attempt; no automatic retry")
        run_logged(
            [
                sys.executable,
                str(SCRIPTS / "convert_parts.py"),
                "--split-dir",
                str(SPLIT),
                "--out",
                str(out),
                "--profile",
                "k4_v4_scaled",
                "--context-length",
                "1024",
                "--native-decoder-package",
                str(PACKAGE),
                "--key-rotation-file",
                str(root / f"rotations/seed{seed}.json"),
                "--parts",
                "2",
                "3",
                "4",
            ],
            root / f"reports/seed{seed}_build.log",
        )
        base = root / f"builds/seed{seed}_embedding"
        base.mkdir()
        original = root / "seed42"
        report = copy.deepcopy(read(original / "convert_report.json"))
        graph_part = report["parts"]["part1_of_4"]
        if any(
            g.get("surgery", {}).get("codec_io") for g in graph_part["graphs"].values()
        ):
            raise ValueError("Embedding reuse must not contain KV operations")
        config = with_key_rotation(
            get_profile("k4_v4_scaled"), root / f"rotations/seed{seed}.json"
        )
        report.update(
            {
                "parts": {"part1_of_4": graph_part},
                "config": config.to_dict(),
                "config_hash": config.config_hash(),
            }
        )
        report.pop("part_sources", None)
        write_json(base / "convert_report.json", report)
        for path in original.iterdir():
            if path.is_file() and (
                path.name.startswith("part1_of_4.")
                or any(path.name.startswith(g + ".") for g in graph_part["graphs"])
                or path.suffix == ".data"
            ):
                (base / path.name).symlink_to(path.resolve())
        run_logged(
            [
                sys.executable,
                str(SCRIPTS / "assemble_native_bundle.py"),
                "--base",
                str(base),
                "--parts",
                str(out),
                "--out",
                str(root / f"seed{seed}"),
            ],
            root / f"reports/seed{seed}_assemble.log",
        )
        for part in range(1, 5):
            source = base if part == 1 else out
            (root / f"seed{seed}/part{part}_of_4.json").symlink_to(
                (source / f"part{part}_of_4.json").resolve()
            )
        print("BUILT", seed, flush=True)

    with ThreadPoolExecutor(max_workers=PROTOCOL["max_build_workers"]) as executor:
        list(executor.map(candidate, missing))


def audit(root: Path) -> None:
    verify_frozen(root, sources=True)
    reference, io_reference = {}, {}
    candidates = read(root / "candidates.json")
    report = {
        "passed": True,
        "candidates": {},
        "scope": "Full non-K ONNX structure/constants and original external weight files, DLC ops/dtypes/shapes/params/calibration, final context I/O; internal HTP scheduling not exposed",
        "encoder_status": "Existing encoder numerical validation failures remain; no claim of new numerical validation success",
    }
    for seed in SEEDS:
        bundle = root / f"seed{seed}"
        conversion = read(bundle / "convert_report.json")
        if conversion["config_hash"] != candidates[str(seed)]["config_hash"]:
            raise ValueError("Candidate/config mismatch")
        entries = {}
        for path in sorted(bundle.glob("*.kv_edits.json")):
            name = path.name.removesuffix(".kv_edits.json")
            result = verify_graph(bundle, name)
            model = onnx.load(bundle / f"{name}.onnx", load_external_data=False)
            compiled = compiled_audit_signature(bundle / f"{name}.dlcinfo.txt")
            signature = (
                graph_signature(model.graph),
                hashlib.sha256(
                    normalized_name(json.dumps(compiled, sort_keys=True)).encode()
                ).hexdigest(),
                digest(bundle / f"{name}.encodings"),
            )
            if seed == 42:
                reference[name] = signature
            elif signature != reference[name]:
                result["violations"].append(
                    "Non-K graph/compiled/calibration difference"
                )
            constants = [
                t
                for t in model.graph.initializer
                if t.name == "tq_rotation_t_dense_qr_s42_d128"
            ]
            hashes = [
                hashlib.sha256(
                    numpy_helper.to_array(t).T.astype("<f4").tobytes()
                ).hexdigest()
                for t in constants
            ]
            if hashes != [candidates[str(seed)]["matrix_f32_sha256"]]:
                result["violations"].append("Wrong or non-shared K matrix")
            result.update(
                {
                    "signature": signature,
                    "source_matmul_count": sum(
                        n.op_type == "MatMul" for n in model.graph.node
                    ),
                    "compiled_matmul_count": sum(
                        o["type"] == "MatMul" for o in compiled["ops"]
                    ),
                    "K_matrix_sha256": hashes,
                }
            )
            entries[name] = result
        if len(entries) != 6:
            raise ValueError("Expected six attention graphs")
        for part in range(1, 5):
            context = read(bundle / f"part{part}_of_4.json")
            layouts = [
                (
                    g["info"]["graphName"],
                    side,
                    {k: v for k, v in t["info"].items() if k != "id"},
                )
                for g in context["info"]["graphs"]
                for side in ("graphInputs", "graphOutputs")
                for t in g["info"][side]
            ]
            if seed == 42:
                io_reference[part] = layouts
            elif layouts != io_reference[part]:
                raise ValueError("Final context I/O/cache ABI changed")
        if digest(bundle / "part1_of_4.bin") != digest(root / "seed42/part1_of_4.bin"):
            raise ValueError("Embedding binary changed")
        report["candidates"][str(seed)] = {
            "graphs": entries,
            "binary_bytes": sum(
                (bundle / f"part{i}_of_4.bin").stat().st_size for i in range(1, 5)
            ),
            "config_hash": conversion["config_hash"],
            "conversion_sha256": digest(bundle / "convert_report.json"),
            "bins_sha256": {
                f"part{i}_of_4.bin": digest(bundle / f"part{i}_of_4.bin")
                for i in range(1, 5)
            },
        }
        report["passed"] &= all(not r["violations"] for r in entries.values())
        print("AUDITED", seed, report["passed"], flush=True)
    write_json(root / "reports/graph_audit.json", report)
    if not report["passed"]:
        raise ValueError("Graph audit failed; no scoring")


def device_hashes(root: Path) -> dict:
    audit_report = read(root / "reports/graph_audit.json")
    expected = {
        f"{DEVICE_ROOT}/bin/qnn-llm-runner": digest(root / "runner/qnn-llm-runner")
    }
    for path in DEVICE_LIBS:
        expected[f"{DEVICE_ROOT}/bin/{Path(path).name}"] = digest(DEFAULT_SDK / path)
    for seed in SEEDS:
        remote = f"{DEVICE_ROOT}/bundles/{experiment_name(root)}_seed{seed}"
        expected.update(
            {
                f"{remote}/{n}": h
                for n, h in audit_report["candidates"][str(seed)]["bins_sha256"].items()
            }
        )
        runtime = read(root / f"seed{seed}/runtime_manifest.json")
        expected[f"{remote}/runtime_manifest.json"] = digest(
            root / f"seed{seed}/runtime_manifest.json"
        )
        expected[f"{remote}/{runtime['native_decoder']['library']}"] = runtime[
            "native_decoder"
        ]["sha256"]
    actual = {
        line.split(maxsplit=1)[1].strip(): line.split()[0]
        for line in shell("sha256sum", *expected).splitlines()
    }
    if actual != expected:
        raise ValueError("Device executable/context/runtime mismatch")
    return actual


def push(root: Path) -> None:
    verify_frozen(root)
    audit_report = read(root / "reports/graph_audit.json")
    if not audit_report["passed"]:
        raise ValueError("Audit required")
    for seed in SEEDS:
        run_logged(
            [
                sys.executable,
                str(SCRIPTS / "run_device_llm.py"),
                "push",
                "--name",
                experiment_name(root) + f"_seed{seed}",
                "--bundle-dir",
                str(root / f"seed{seed}"),
                "--runner",
                str(root / "runner/qnn-llm-runner"),
            ],
            root / f"reports/seed{seed}_push.log",
        )
    files = [
        Path(__file__),
        SCRIPTS / "rotation_nll_selection.py",
        SCRIPTS / "run_device_llm.py",
        SCRIPTS / "benchmark_key_rotation.py",
        SCRIPTS / "benchmark_rotation_mse_nll.py",
    ]
    write_json(
        root / "reports/execution_identity.json",
        {
            "code_sha256": {str(p): digest(p) for p in files},
            "device_files_sha256": device_hashes(root),
            "graph_audit_sha256": digest(root / "reports/graph_audit.json"),
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], text=True
            ).strip(),
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )


def verify_execution(root: Path) -> None:
    verify_frozen(root)
    identity = read(root / "reports/execution_identity.json")
    for path, checksum in identity["code_sha256"].items():
        if digest(path) != checksum:
            raise ValueError("Frozen execution code changed")
    if digest(root / "reports/graph_audit.json") != identity["graph_audit_sha256"]:
        raise ValueError("Audit changed")
    if device_hashes(root) != identity["device_files_sha256"]:
        raise ValueError("Device payload changed")


def run(root: Path, seed: int, tag: str, role: str, options: list[str]) -> dict:
    path = root / f"reports/{tag}.json"
    with path.with_suffix(".attempt").open("x") as stream:
        stream.write("Single predeclared attempt; no automatic repeat.\n")
    snapshot(path.with_suffix(".before.json"))
    assets = root / f"assets_{role}"
    run_logged(
        [
            sys.executable,
            str(SCRIPTS / "run_device_llm.py"),
            "run",
            "--name",
            experiment_name(root) + f"_seed{seed}",
            "--assets",
            str(assets),
            "--context-buckets",
            "1024",
            "--report",
            str(path),
            "--remote-report-tag",
            experiment_name(root) + "_" + tag,
            *options,
        ],
        path.with_suffix(".stdout.log"),
    )
    snapshot(path.with_suffix(".after.json"))
    data = read(path)
    if data["device"] != read(root / "source_identity.json")["device"]:
        raise ValueError("Device changed")
    if (
        data["config_hash"] != read(root / "candidates.json")[str(seed)]["config_hash"]
        or data["kv_store_bytes"] != 30277632
        or data.get("diagnostic_only")
    ):
        raise ValueError("Configuration/storage/instrumentation mismatch")
    if (
        data["assets"]["tokens_sha256"]
        != read(assets / "assets.json")["sha256"][data["assets"]["tokens_file"]]
    ):
        raise ValueError("Tokens changed")
    return data


def validation(root: Path) -> None:
    verify_execution(root)
    windows = read(root / "data_manifest.json")["windows"]["validation"]
    for i, window in enumerate(windows):
        order = SEEDS[i % 8 :] + SEEDS[: i % 8]
        for seed in order:
            data = run(
                root,
                seed,
                f"seed{seed}_validation_{i:02d}",
                "validation",
                ["--mode", "score", "--tokens", window["file"]],
            )
            validate_score(data, window)
            print("VALIDATION", i, seed, data["nll_sum"] / 1023, flush=True)
    verify_execution(root)
    write_json(
        root / "reports/validation_complete.json",
        {"runs": 128, "repeats": 1, "heldout_scored": False},
    )


def selection(root: Path) -> None:
    verify_frozen(root)
    read(root / "reports/validation_complete.json")
    manifest = read(root / "data_manifest.json")
    rows, nll = [], []
    hashes = {}
    for seed in SEEDS:
        values = []
        for i, window in enumerate(manifest["windows"]["validation"]):
            path = root / f"reports/seed{seed}_validation_{i:02d}.json"
            data = read(path)
            validate_score(data, window)
            if (
                data["config_hash"]
                != read(root / "candidates.json")[str(seed)]["config_hash"]
            ):
                raise ValueError("Candidate report mismatch")
            hashes[path.name] = digest(path)
            values.append(data["nll_sum"])
            rows.append(
                {
                    "seed": seed,
                    "document": i,
                    "title": window["title"],
                    **aggregate([data]),
                }
            )
        nll.append(values)
    table = np.asarray(nll)
    counts = np.full(16, 1023)
    chosen = choose_seed(table, counts, SEEDS)
    totals = [
        {
            "seed": s,
            "nll_sum": float(table[i].sum()),
            "scored_tokens": int(counts.sum()),
            "mean_nll": float(table[i].sum() / counts.sum()),
            "ppl": math.exp(float(table[i].sum() / counts.sum())),
        }
        for i, s in enumerate(SEEDS)
    ]
    result = {
        "selected_seed": chosen,
        "aliases": {"A": 42, "B": 48, "S": chosen},
        "unique_final_seeds": list(dict.fromkeys([42, 48, chosen])),
        "criterion": PROTOCOL["selection"],
        "validation": totals,
        "fold_analysis": fold_analysis(
            table, counts, SEEDS, manifest["validation_folds"]
        ),
        "validation_reports_sha256": hashes,
        "heldout_access": "No heldout device execution before this file is frozen",
    }
    if list((root / "reports").glob("seed*_heldout_*.attempt")):
        raise ValueError("Selection cannot follow heldout inspection")
    write_json(root / "selection.json", result)
    write_csv(root / "reports/validation_nll.csv", rows)
    write_csv(root / "reports/validation_summary.csv", totals)
    write_json(root / "reports/folds.json", result["fold_analysis"])
    write_json(
        root / "selection_identity.json",
        {
            "selection_sha256": digest(root / "selection.json"),
            "data_manifest_sha256": digest(root / "data_manifest.json"),
        },
    )
    print("SELECTED", chosen, json.dumps(totals), flush=True)


def selected(root: Path) -> list[int]:
    frozen = read(root / "selection_identity.json")
    if (
        digest(root / "selection.json") != frozen["selection_sha256"]
        or digest(root / "data_manifest.json") != frozen["data_manifest_sha256"]
    ):
        raise ValueError("Selection/data changed after lock")
    result = read(root / "selection.json")
    for name, checksum in result["validation_reports_sha256"].items():
        if digest(root / "reports" / name) != checksum:
            raise ValueError("Selection evidence changed")
    return result["unique_final_seeds"]


def heldout(root: Path) -> None:
    verify_execution(root)
    seeds = selected(root)
    windows = read(root / "data_manifest.json")["windows"]["heldout"]
    for i, window in enumerate(windows):
        start = i % len(seeds)
        for seed in seeds[start:] + seeds[:start]:
            data = run(
                root,
                seed,
                f"seed{seed}_heldout_{i:02d}",
                "heldout",
                ["--mode", "score", "--tokens", window["file"]],
            )
            validate_score(data, window)
            print("HELDOUT", i, seed, data["nll_sum"] / 1023, flush=True)
    verify_execution(root)
    write_json(
        root / "reports/heldout_complete.json",
        {
            "runs": 16 * len(seeds),
            "unique_seeds": seeds,
            "selection_sha256": digest(root / "selection.json"),
        },
    )


def performance(root: Path) -> None:
    verify_execution(root)
    seeds = selected(root)
    for repeat in range(3):
        start = repeat % len(seeds)
        order = seeds[start:] + seeds[:start]
        for condition in ("short", "long"):
            for seed in order:
                options = ["--mode", "generate", "--n-gen", "128"]
                if condition == "long":
                    options += ["--tokens", "boundary_prompt_cl1024.bin"]
                run(
                    root,
                    seed,
                    f"seed{seed}_{condition}_r{repeat}",
                    "performance",
                    options,
                )
    verify_execution(root)
    write_json(
        root / "reports/performance_complete.json",
        {"repeats": 3, "unique_seeds": seeds},
    )


def summarize(root: Path) -> None:
    read(root / "reports/heldout_complete.json")
    read(root / "reports/performance_complete.json")
    seeds = selected(root)
    chosen = read(root / "selection.json")["selected_seed"]
    windows = read(root / "data_manifest.json")["windows"]["heldout"]
    counts = np.full(16, 1023)
    rows, nll, groups = [], {}, {}
    for seed in seeds:
        reports = []
        for i, window in enumerate(windows):
            data = read(root / f"reports/seed{seed}_heldout_{i:02d}.json")
            validate_score(data, window)
            reports.append(data)
            rows.append(
                {
                    "seed": seed,
                    "document": i,
                    "title": window["title"],
                    **aggregate([data]),
                }
            )
        nll[seed] = np.asarray([r["nll_sum"] for r in reports])
        groups[str(seed)] = {"quality": aggregate(reports), "performance": {}}
        for condition, prompt in (("short", 35), ("long", 897)):
            runs = [
                performance_result(
                    root / f"reports/seed{seed}_{condition}_r{i}.json", prompt
                )
                for i in range(3)
            ]
            fields = (
                "ttft_ms",
                "decode_tok_per_s",
                "prefill_tok_per_s",
                "host_kv_MiB",
                "io_buffer_MiB",
                "end_VmRSS_MiB",
                "process_VmHWM_MiB",
            )
            groups[str(seed)]["performance"][condition] = {
                field: {
                    "median": statistics.median(values := [r[field] for r in runs]),
                    "min": min(values),
                    "max": max(values),
                    "samples": values,
                }
                for field in fields
            }
        groups[str(seed)]["binary_bytes"] = read(root / "reports/graph_audit.json")[
            "candidates"
        ][str(seed)]["binary_bytes"]
    pairs = {
        label: {
            **paired_bootstrap(nll[chosen], nll[baseline], counts),
            "same_candidate": chosen == baseline,
        }
        for label, baseline in (("S_minus_A", 42), ("S_minus_B", 48))
    }
    pairs["B_minus_A"] = paired_bootstrap(nll[48], nll[42], counts)
    delta_rows = [
        {
            "document": i,
            "title": w["title"],
            "A_mean_nll": float(nll[42][i] / 1023),
            "B_mean_nll": float(nll[48][i] / 1023),
            "S_mean_nll": float(nll[chosen][i] / 1023),
            "S_minus_A": float((nll[chosen][i] - nll[42][i]) / 1023),
            "S_minus_B": float((nll[chosen][i] - nll[48][i]) / 1023),
        }
        for i, w in enumerate(windows)
    ]
    result = {
        "protocol": PROTOCOL,
        "reproduction_only": (root / "replay.json").exists(),
        "selection": read(root / "selection.json"),
        "groups": groups,
        "heldout_pairs": pairs,
        "heldout_documents": delta_rows,
        "primary_decision": pairs["S_minus_A"],
        "limitations": "CI conditional on fixed selected S and sampled documents; not timing variability or all-domain guarantee. B selected on different historical data, no general superiority claim for NLL versus MSE selection. Encoder numerical failures remain. Default42 unchanged.",
    }
    write_json(root / "reports/comparison.json", result)
    write_csv(root / "reports/heldout_nll.csv", rows)
    write_csv(root / "reports/heldout_deltas.csv", delta_rows)
    verify_frozen(root, sources=True)
    write_json(
        root / "reports/preservation.json",
        {
            "originals_unchanged": True,
            "default_promoted": False,
            "training_runs": 0,
            "heldout_now_previously_observed": True,
        },
    )
    print(
        json.dumps(
            {
                "selected_seed": chosen,
                "heldout_pairs": pairs,
                "groups": {s: g["quality"] for s, g in groups.items()},
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "build",
            "audit",
            "push",
            "validation",
            "selection",
            "heldout",
            "performance",
            "summarize",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.root.resolve())


if __name__ == "__main__":
    main()
