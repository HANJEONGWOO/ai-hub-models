# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Frozen Structured K8 allocation experiment; immutable artifacts and single runs."""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnx
from benchmark_asymmetric_once import ARTIFACTS, digest, execute, read, save_new

from qai_hub_models.models.templates.llm.turboquant.config import (
    STRUCTURED_ALLOCATION_PROFILE,
    get_profile,
    structured_allocation_plan,
)

PREVIOUS = ARTIFACTS / "structured_k8_decoder_20261013"
SOURCE = PREVIOUS / "structured_arithmetic"
CALIBRATION = ARTIFACTS / "k_rotation_nll_selection_20261009"
STATIC = {
    "uniform_odd": list(range(1, 28, 2)),
    "uniform_even": list(range(0, 28, 2)),
    "early": list(range(14)),
    "late": list(range(14, 28)),
    "middle": list(range(7, 21)),
    "boundary": list(range(7)) + list(range(21, 28)),
}


def symlink(source: Path, target: Path) -> None:
    if target.exists():
        if target.resolve() != source.resolve():
            raise ValueError(f"Conflicting existing artifact {target}")
    else:
        target.symlink_to(source.resolve(strict=True))


def prepare(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    for sub in ("plans", "reports", "logs", "assets", "runner", "offline", "builds"):
        (root / sub).mkdir()
    subprocess.run(
        [
            "bash",
            "scripts/llm/turboquant/qnn_runner/build_android.sh",
            str(root / "runner"),
        ],
        check=True,
    )
    meta = read(SOURCE / "convert_report.json")
    assert meta["structured_k8_decoder"] == "arithmetic"
    assert (
        meta["config_hash"]
        == get_profile("mixed_k6_v2_uniform_structured_k8").config_hash()
    )
    assets = ARTIFACTS / "device_assets_cl1024"
    manifest = read(assets / "assets.json")
    copied = copy.deepcopy(manifest)
    for name, checksum in manifest["sha256"].items():
        assert digest(assets / name) == checksum
        symlink(assets / name, root / "assets" / name)
    data = read(CALIBRATION / "data_manifest.json")
    documents = []
    for i, item in enumerate(data["windows"]["validation"][:4]):
        source = CALIBRATION / "assets_validation" / item["file"]
        assert digest(source) == item["tokens_sha256"]
        name = f"calibration_{i}.bin"
        symlink(source, root / "assets" / name)
        copied["sha256"][name] = digest(source)
        documents.append(
            {
                **item,
                "file": name,
                "role": "allocation_calibration",
                "source": str(source),
            }
        )

    # Reject accidental exact/contiguous overlap with the historical test inputs.
    def grams(path: Path) -> set[bytes]:
        values = np.fromfile(path, dtype="<i4")
        return {values[i : i + 16].tobytes() for i in range(len(values) - 15)}

    calibration_grams = set().union(
        *(grams(root / "assets" / d["file"]) for d in documents)
    )
    test_grams = set().union(
        *(grams(assets / name) for name in manifest["wikitext_windows"])
    )
    assert not (calibration_grams & test_grams), (
        "Calibration and test share a 16-token span"
    )
    save_new(root / "assets/assets.json", copied)
    save_new(
        root / "data_manifest.json",
        {
            "calibration": documents,
            "test": {
                name: manifest["sha256"][name] for name in manifest["wikitext_windows"]
            },
            "dataset": data["dataset"],
            "revision": data["revision"],
            "calibration_source_manifest_sha256": digest(
                CALIBRATION / "data_manifest.json"
            ),
            "overlap_16grams": 0,
            "limits": "Historical test windows, excluded from this search but previously reported; not a new unseen generalization benchmark. Four calibration documents are reused from earlier rotation validation, not activation recalibration.",
        },
    )
    for group, layers in STATIC.items():
        save_new(root / "plans" / f"{group}.json", structured_allocation_plan(layers))
    paths = [
        SOURCE / "convert_report.json",
        PREVIOUS / "native/manifest.json",
        root / "runner/qnn-llm-runner",
        root / "data_manifest.json",
    ]
    paths += list(SOURCE.glob("part*_of_4.bin"))
    paths += [
        Path(v["path"])
        for v in read(PREVIOUS / "native/manifest.json")["libraries"].values()
    ]
    paths += list((root / "plans").glob("*.json"))
    paths += [root / "assets" / name for name in copied["sha256"]]
    split = ARTIFACTS / "qwen3_1_7b_w4a16_split"
    paths += [split / "split_manifest.json"]
    paths += (
        list(split.glob("*.aimet/*.onnx"))
        + list(split.glob("*.aimet/*.encodings"))
        + list(split.glob("*.aimet/*.data"))
    )
    save_new(
        root / "protocol.json",
        {
            "base_branch": "exp/structured-k8-decoder",
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "branch": "exp/structured-k8-nll-allocation",
            "model": "Qwen3-1.7B W4A16",
            "context": 1024,
            "static_masks": STATIC,
            "native": str(PREVIOUS / "native"),
            "split": str(split),
            "runner": str(root / "runner/qnn-llm-runner"),
            "assets": str(root / "assets"),
            "frozen_files": {str(p): digest(p) for p in paths},
            "search": {
                "objective": "Full-model teacher-forced mean NLL, never Attention MSE",
                "initial": "uniform_odd",
                "max_iterations": 3,
                "shortlist_per_direction": 4,
                "accept_min_nll_decrease": 1e-6,
                "tie": "lexicographic (NLL, promotion_layer, demotion_layer)",
                "single_flips": "Offline sensitivity diagnostics may have 13/15 K8 layers; deployable masks and every swap must have exactly 14",
                "calibration_windows": 4,
                "tokens_per_window": 1024,
                "max_swaps_per_iteration": 16,
                "selection_gate": "Score offline candidate and Uniform-Odd on all four calibration documents on HTP; retain Uniform-Odd if candidate does not improve mean NLL by >1e-6. No test-based selection.",
            },
            "evaluation": {
                "test_windows": 4,
                "scored_tokens_per_window": 1023,
                "performance_prompt": 897,
                "generated": 128,
                "sessions_per_configuration": 1,
                "order": [*STATIC, "nll_selected"],
                "ppl": "exp(total_nll / total_scored_tokens)",
                "kl": "KL(Uniform-Odd || candidate), per target; mean/P95/P99; not FP16-reference KL",
                "performance_profiled": False,
            },
            "fixed": "K8 structured+HVX arithmetic, K4/V2 LM Native, K8x14/K4x14/V2x28, Dense K42/V542, norm correction FP16 scales, tile256, current+past compressed, unchanged weights/calibration; 30277632 KV bytes",
            "limits": "Offline W4A16 fake-quantized emulator is not HTP bit-exact; compare NLL/logits on calibration before trusting transfer. Single performance observation, no significance claim. Historical encoder scale gate failures are not waived.",
        },
    )
    # The numerical graph/cache contract is identical for Uniform-Odd; only the
    # opt-in profile's descriptive identity differs. Reuse immutable binaries.
    group = root / "uniform_odd"
    group.mkdir()
    for path in SOURCE.iterdir():
        if path.is_file() and (
            path.name.startswith(("part", "token_", "prompt_"))
            or path.suffix == ".data"
        ):
            symlink(path, group / path.name)
    config = get_profile(
        STRUCTURED_ALLOCATION_PROFILE, layer_plan=read(root / "plans/uniform_odd.json")
    )
    updated = copy.deepcopy(meta)
    updated.update(
        profile=config.profile,
        config=config.to_dict(),
        config_hash=config.config_hash(),
        layer_plan=str(root / "plans/uniform_odd.json"),
    )
    updated["allocation_reuse"] = {
        "source": str(SOURCE),
        "source_config_hash": meta["config_hash"],
        "reason": "Identical layer bits, quantizer, matrices, scales, decoder, current/past and attention; metadata-only allocation identity",
    }
    assert (
        meta["config"]["layer_plan"]["bits"] == updated["config"]["layer_plan"]["bits"]
    )
    save_new(group / "convert_report.json", updated)


def validate(root: Path) -> dict:
    protocol = read(root / "protocol.json")
    for name, checksum in protocol["frozen_files"].items():
        if digest(Path(name)) != checksum:
            raise ValueError("Frozen input changed: " + name)
    return protocol


def build_group(root: Path, group: str, protocol: dict) -> None:
    target = root / group
    if (target / "convert_report.json").exists():
        raise FileExistsError(target)
    plan = read(root / "plans" / f"{group}.json")
    get_profile(STRUCTURED_ALLOCATION_PROFILE, layer_plan=plan)

    def part(number: int) -> None:
        execute(
            "convert_parts.py",
            [
                "--split-dir",
                protocol["split"],
                "--out",
                str(root / "builds" / group / f"part{number}"),
                "--profile",
                STRUCTURED_ALLOCATION_PROFILE,
                "--layer-plan",
                str(root / "plans" / f"{group}.json"),
                "--structured-k8-decoder",
                "arithmetic",
                "--native-decoder-package",
                protocol["native"],
                "--context-length",
                "1024",
                "--sequence-lengths",
                "1",
                "128",
                "--parts",
                str(number),
            ],
            root / "logs" / f"build_{group}_part{number}.log",
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(part, (2, 3, 4)))
    assemble_group(root, group)


def assemble_group(root: Path, group: str) -> None:
    """Assemble completed parts without recompiling or overwriting any artifact."""
    target = root / group
    expected = read(root / "plans" / f"{group}.json")["config_hash"]
    for number in (2, 3, 4):
        source = root / "builds" / group / f"part{number}"
        report = read(source / "convert_report.json")
        assert report["config_hash"] == expected
        assert report["structured_k8_decoder"] == "arithmetic"
        assert (source / f"part{number}_of_4.bin").is_file()
    target.mkdir()
    combined = copy.deepcopy(
        read(root / "builds" / group / "part2/convert_report.json")
    )
    combined["parts"] = {}
    for number in (1, 2, 3, 4):
        source = (
            root / "uniform_odd"
            if number == 1
            else root / "builds" / group / f"part{number}"
        )
        report = read(source / "convert_report.json")
        name = f"part{number}_of_4"
        combined["parts"][name] = report["parts"][name]
        symlink(source / f"{name}.bin", target / f"{name}.bin")
        for graph in report["parts"][name]["graphs"]:
            for path in source.glob(graph + ".*"):
                if path.is_file():
                    symlink(path, target / path.name)
        for path in source.glob("*.data"):
            symlink(path, target / path.name)
    save_new(target / "convert_report.json", combined)
    execute(
        "verify_rotated_attention.py",
        [
            "--bundle",
            str(target),
            "--report",
            str(root / "reports" / f"audit_{group}.json"),
        ],
        root / "logs" / f"audit_{group}.log",
    )


def push(root: Path, group: str, protocol: dict) -> None:
    execute(
        "run_device_llm.py",
        [
            "push",
            "--bundle-dir",
            str(root / group),
            "--runner",
            protocol["runner"],
            "--name",
            root.name + "_" + group,
        ],
        root / "logs" / f"push_{group}.log",
    )


def run_once(
    root: Path, group: str, label: str, options: list[str], protocol: dict
) -> dict:
    report = root / "reports" / f"{group}_{label}.json"
    if report.exists():
        raise FileExistsError("Completed measurement already exists: " + str(report))
    execute(
        "run_device_llm.py",
        [
            "run",
            "--name",
            root.name + "_" + group,
            "--assets",
            protocol["assets"],
            "--report",
            str(report),
            "--remote-report-tag",
            root.name + "_" + group + "_" + label,
            *options,
        ],
        report.with_suffix(".stdout.log"),
    )
    value = read(report)
    expected = read(root / group / "convert_report.json")
    assert (
        value["config_hash"] == expected["config_hash"] and value["quantize_current_kv"]
    )
    assert value["kv_store_bytes"] == 30277632 and not value.get("op_profiles")
    assert (
        value["native_decoder"]["sha256"]
        == expected["native_decoder"]["libraries"]["hexagon-v81"]["sha256"]
    )
    assert (
        value["assets"]["tokens_sha256"]
        == read(root / "assets/assets.json")["sha256"][value["assets"]["tokens_file"]]
    )
    return value


def score(root: Path, group: str, split: str, protocol: dict) -> None:
    prefix = "calibration" if split == "calibration" else "wikitext_w"
    for i in range(4):
        name = f"{prefix}_{i}.bin" if split == "calibration" else f"{prefix}{i}.bin"
        run_once(
            root,
            group,
            f"{split}_{i}",
            ["--mode", "score", "--tokens", name, "--dump-logits"],
            protocol,
        )


def audit_allocation(root: Path, group: str) -> None:
    report = root / "reports" / f"allocation_audit_{group}.json"
    if report.exists():
        assert read(report)["passed"]
        return
    config = get_profile(
        STRUCTURED_ALLOCATION_PROFILE, layer_plan=read(root / "plans" / f"{group}.json")
    )
    rows = {}
    for part in (2, 3, 4):
        for ar in (1, 128):
            stem = f"{'token' if ar == 1 else 'prompt'}_ar{ar}_cl1024_{part}_of_4"
            graphs = [
                onnx.load(root / g / (stem + ".onnx"), load_external_data=False).graph
                for g in ("uniform_odd", group)
            ]
            stable = [
                {
                    node.name: node.SerializeToString()
                    for node in graph.node
                    if not re.match(r"tq_key_\d+_", node.name)
                }
                for graph in graphs
            ]
            constants = [
                {
                    t.name: t.SerializeToString()
                    for t in graph.initializer
                    if not re.match(r"tq_key_\d+_", t.name)
                }
                for graph in graphs
            ]
            shared = constants[0].keys() & constants[1].keys()
            enc = [
                read(root / g / (stem + ".encodings")) for g in ("uniform_odd", group)
            ]
            new_nodes = {node.name: node for node in graphs[1].node}
            expected_ops = []
            for layer in range(28):
                name = f"tq_key_{layer}_current_native_fp16"
                if name in new_nodes:
                    expected_ops.append(
                        new_nodes[name].op_type
                        == (
                            "Decode8Structured"
                            if config.codec_for("key", layer).bits == 8
                            else "Decode4"
                        )
                    )
            rows[stem] = {
                "non_key_nodes_identical": stable[0] == stable[1],
                "shared_non_key_constants_identical": all(
                    constants[0][n] == constants[1][n] for n in shared
                ),
                "weight_encodings_identical": enc[0]["param_encodings"]
                == enc[1]["param_encodings"],
                "expected_current_decoder_types": bool(expected_ops)
                and all(expected_ops),
            }
    passed = all(all(row.values()) for row in rows.values())
    save_new(
        report,
        {
            "group": group,
            "config_hash": config.config_hash(),
            "graphs": rows,
            "passed": passed,
        },
    )
    if not passed:
        raise ValueError("Allocation changed operations outside the key codec")


def finalize(root: Path) -> None:
    candidate = read(root / "plans/nll_candidate.json")
    match = next(
        (name for name, layers in STATIC.items() if candidate["k8_layers"] == layers),
        "nll_candidate",
    )
    means = {}
    for group in dict.fromkeys(("uniform_odd", match)):
        quality = [
            read(root / "reports" / f"{group}_calibration_{i}.json") for i in range(4)
        ]
        assert all(row["scored_tokens"] == 1023 for row in quality)
        means[group] = sum(row["nll_sum"] for row in quality) / 4092
    accepted = means[match] < means["uniform_odd"] - 1e-6
    selected = match if accepted else "uniform_odd"
    selected_plan = read(root / "plans" / f"{selected}.json")
    save_new(root / "plans/nll_selected.json", selected_plan)
    save_new(
        root / "selection.json",
        {
            "candidate_group": match,
            "selected_group": selected,
            "accepted_on_HTP_calibration": accepted,
            "calibration_mean_nll": means,
            "candidate_delta_nll": means[match] - means["uniform_odd"],
            "candidate_layers": candidate["k8_layers"],
            "selected_layers": selected_plan["k8_layers"],
            "selected_config_hash": selected_plan["config_hash"],
            "offline_search_sha256": digest(root / "offline/search.json"),
            "test_used_for_selection": False,
            "rule": "Pre-registered HTP calibration acceptance gate, minimum NLL decrease 1e-6; keep initial Uniform-Odd if transfer fails. No re-search based on test.",
        },
    )


def evaluate(root: Path, group: str, protocol: dict) -> None:
    if not (root / "selection.json").exists():
        raise ValueError("Freeze final selection before test evaluation")
    audit_allocation(root, group)
    run_once(
        root,
        group,
        "perf",
        [
            "--mode",
            "generate",
            "--tokens",
            "boundary_prompt_cl1024.bin",
            "--n-gen",
            "128",
            "--sessions",
            "1",
        ],
        protocol,
    )
    score(root, group, "test", protocol)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=[
            "prepare",
            "build",
            "assemble",
            "push",
            "calibration",
            "evaluate",
            "audit",
            "finalize",
        ],
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--groups", nargs="+", default=list(STATIC)[1:])
    args = p.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        prepare(root)
        return
    protocol = validate(root)
    if args.stage == "finalize":
        finalize(root)
        return
    for group in args.groups:
        if args.stage == "build":
            build_group(root, group, protocol)
        elif args.stage == "assemble":
            assemble_group(root, group)
        elif args.stage == "push":
            push(root, group, protocol)
        elif args.stage == "calibration":
            score(root, group, "calibration", protocol)
        elif args.stage == "audit":
            audit_allocation(root, group)
        else:
            evaluate(root, group, protocol)


if __name__ == "__main__":
    main()
