# SPDX-License-Identifier: BSD-3-Clause
"""Immutable Uniform/Structured K8 experiment; three crossed performance runs."""

from __future__ import annotations

import argparse
import copy
import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark_asymmetric_once import ARTIFACTS, digest, execute, read, save_new
from benchmark_lm8_convergence import STRUCTURED_SOURCE, freeze_sources, heldout
from benchmark_mixed_once import measure
from benchmark_nll_allocation import symlink

from qai_hub_models.models.templates.llm.turboquant.config import get_profile
from qai_hub_models.models.templates.llm.turboquant.uniform_k8 import CODEBOOK_SHA256

GROUPS = {
    "uniform": "mixed_k6_v2_uniform_uniform_k8",
    "structured": "mixed_k6_v2_uniform_structured_k8",
}
ORDER = ["uniform", "structured", "structured", "uniform", "uniform", "structured"]


def prepare(root: Path, data_manifest: Path | None) -> None:
    for name in ("logs", "reports", "assets", "builds", "tests"):
        (root / name).mkdir(parents=True, exist_ok=False)
    assert (
        read(root / "codebook/optimization.json")["centroid_sha256"] == CODEBOOK_SHA256
    )
    assets = copy.deepcopy(read(ARTIFACTS / "device_assets_cl1024/assets.json"))
    for name, sha in assets["sha256"].items():
        source = ARTIFACTS / "device_assets_cl1024" / name
        assert digest(source) == sha
        symlink(source, root / "assets" / name)
    if data_manifest:
        data = read(data_manifest)
        assert len(data["heldout"]) == len({d["file"] for d in data["heldout"]}) == 4
        assert data["legacy"] == {
            n: assets["sha256"][n] for n in assets["wikitext_windows"]
        }
        for d in data["heldout"]:
            source = data_manifest.parent / "assets" / d["file"]
            assert (
                digest(source) == d["tokens_sha256"] and source.stat().st_size == 4096
            )
            symlink(source, root / "assets" / d["file"])
            assets["sha256"][d["file"]] = d["tokens_sha256"]
    else:
        data = heldout(root, assets, selection_salt="UniformK8-heldout-20261020:")
    save_new(root / "data_manifest.json", data)
    save_new(root / "assets/assets.json", assets)
    native = read(root / "native/manifest.json")
    assert {"Decode8Structured", "Decode8Uniform", "Decode4", "Decode2"}.issubset(
        native["operations"]
    )
    runner = ARTIFACTS / "structured_k8_nll_allocation_20261013/runner/qnn-llm-runner"
    frozen = [
        root / "codebook/optimization.json",
        root / "data_manifest.json",
        root / "native/manifest.json",
        runner,
        STRUCTURED_SOURCE / "convert_report.json",
    ]
    frozen += [Path(v["path"]) for v in native["libraries"].values()]
    frozen += [root / "assets" / n for n in assets["sha256"]]
    # Only immutable source graph inputs; never inherit runtime manifests/logs.
    target = root / "structured"
    target.mkdir()
    for path in STRUCTURED_SOURCE.iterdir():
        if path.is_file() and (
            path.suffix in (".onnx", ".dlc", ".encodings", ".data")
            or path.name.endswith((".kv_edits.json", ".dlcinfo.txt"))
        ):
            symlink(path, target / path.name)
            frozen.append(path)
    frozen += list(STRUCTURED_SOURCE.glob("part*_of_4.bin"))
    previous = copy.deepcopy(read(STRUCTURED_SOURCE / "convert_report.json"))
    assert previous["config_hash"] == get_profile(GROUPS["structured"]).config_hash()
    assert previous["structured_k8_decoder"] == "arithmetic"
    previous["native_decoder"] = native
    previous["uniform_experiment_source"] = str(STRUCTURED_SOURCE)
    save_new(target / "convert_report.json", previous)
    save_new(
        root / "protocol.json",
        {
            "branch": "exp/uniform-k8-baseline",
            "base_branch": "exp/lloyd-max-k8-convergence",
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "model": "Qwen3-1.7B W4A16",
            "context_length": 1024,
            "profiles": {
                g: {
                    "profile": p,
                    "config": get_profile(p).to_dict(),
                    "config_hash": get_profile(p).config_hash(),
                }
                for g, p in GROUPS.items()
            },
            "assets": str(root / "assets"),
            "assets_manifest": assets,
            "runner": str(runner),
            "native": str(root / "native"),
            "split": str(ARTIFACTS / "qwen3_1_7b_w4a16_split"),
            "frozen_files": {str(p): digest(p) for p in frozen},
            "performance": {
                "sessions_per_group": 3,
                "order": ORDER,
                "prompt": 897,
                "generated": 128,
                "profiled": False,
                "statistics": "mean, sample standard deviation, min, max; no significance claim",
            },
            "quality": {
                "legacy_windows": 4,
                "new_heldout_windows": 4,
                "targets_per_window": 1023,
                "runs_per_window": 1,
            },
            "fixed": "Odd14 K8/even14 K4, V2 all; Dense K42/V542, current/past compressed, FP16 scales/norm correction, tile256, weights/calibration; 30277632 cache bytes",
            "selection": "One spacing selected only by analytic Gaussian scalar MSE, rounded once to FP16; no NLL/real-KV selection",
            "probe": "Encoder AR1/AR128, decoder 1/3/256/1023; separate instrumented costs, no conversion from busy cycles to TTFT percentages",
            "limits": "Existing AR128 encoder tolerance failures tracked separately; new heldout is unused in recorded experiments, not guaranteed absent from model pretraining",
        },
    )
    freeze_sources(root, ORDER)
    print("HELDOUT", [d["title"] for d in data["heldout"]], flush=True)


def validate(root: Path) -> dict:
    protocol = read(root / "protocol.json")
    for name, sha in protocol["frozen_files"].items():
        assert digest(Path(name)) == sha, name
    for g, p in GROUPS.items():
        assert get_profile(p).config_hash() == protocol["profiles"][g]["config_hash"]
    return protocol


def probes(root: Path, stage: str, components: list[str]) -> None:
    for group in GROUPS:
        if "encoder" in components:
            work = root / f"encoder_{group}"
            if stage == "build" and work.exists():
                raise FileExistsError(work)
            options = [
                "--work-dir",
                str(work),
                "--snapshot",
                str(ARTIFACTS / "qwen3_1_7b_kv_snapshot.npz"),
                "--profile",
                group + "_k8_v4_scaled",
                "--range-scale",
                "0.5",
                "--tokens",
                "1",
                "128",
                "--head-major",
                "--operations",
                "encode",
                "--device-dir",
                f"/data/local/tmp/qaihm_turboquant/{root.name}_{group}_encoder",
            ]
            execute(
                "htp_codec_validation.py",
                [stage, *options],
                root / "logs" / f"encoder_{group}_{stage}.log",
            )
            if stage == "run":
                execute(
                    "htp_codec_validation.py",
                    ["compare", *options],
                    root / "logs" / f"encoder_{group}_compare.log",
                )
        if "decoder" in components:
            work = root / f"decoder_{group}"
            if stage == "build" and work.exists():
                raise FileExistsError(work)
            options = [
                "--work-dir",
                str(work),
                "--package",
                str(root / "native"),
                "--bits",
                "8",
                "--lut",
                group + "_k8",
                "--" + group + "-arithmetic",
                "--tokens",
                "1",
                "3",
                "256",
                "1023",
                "--scale-edges",
                "--strict-bit-exact",
                "--optrace",
                "--device-dir",
                f"/data/local/tmp/qaihm_turboquant/{root.name}_{group}_decoder",
            ]
            execute(
                "validate_native_decoder.py",
                [stage, *options],
                root / "logs" / f"decoder_{group}_{stage}.log",
            )
            if stage == "run":
                execute(
                    "validate_native_decoder.py",
                    ["compare", *options],
                    root / "logs" / f"decoder_{group}_compare.log",
                )


def build(root: Path, protocol: dict) -> None:
    suites = ET.parse(root / "tests/regression_final.xml").getroot().iter("testsuite")
    assert all(
        int(s.get("failures", "0")) == int(s.get("errors", "0")) == 0 for s in suites
    )
    for group in GROUPS:
        enc = read(root / f"encoder_{group}/p2_report.json")
        assert all(enc["graphs"][f"tq_encode_key_t{t}"]["passed"] for t in (1, 128))
        checks = read(root / f"decoder_{group}/correctness.json")
        assert len(checks) == 4 and all(
            c["passed"] and c["bit_exact_fraction"] == 1 for c in checks.values()
        )

    def job(part: int) -> None:
        group = "structured" if part == 0 else "uniform"
        target = root / group if part == 0 else root / "builds" / f"part{part}"
        options = [
            "--split-dir",
            protocol["split"],
            "--out",
            str(target),
            "--profile",
            GROUPS[group],
            "--context-length",
            "1024",
            "--sequence-lengths",
            "1",
            "128",
            "--native-decoder-package",
            protocol["native"],
        ]
        options += (
            ["--context-only", "--structured-k8-decoder", "arithmetic"]
            if part == 0
            else ["--parts", str(part)]
        )
        execute("convert_parts.py", options, root / "logs" / f"build_{part}.log")

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(job, (2, 3, 4, 0)))
    target = root / "uniform"
    target.mkdir(exist_ok=False)
    combined = copy.deepcopy(read(root / "builds/part2/convert_report.json"))
    combined["parts"] = {}
    provenance = {}
    for part in (1, 2, 3, 4):
        source = root / "structured" if part == 1 else root / "builds" / f"part{part}"
        report = read(source / "convert_report.json")
        name = f"part{part}_of_4"
        entry = report["parts"][name]
        assert "context_s" in entry and report["native_decoder"] == read(
            root / "native/manifest.json"
        )
        if part == 1:
            assert all(
                not g.get("surgery", {}).get("codec_io")
                for g in entry["graphs"].values()
            )
        else:
            assert report["config_hash"] == combined["config_hash"]
        combined["parts"][name] = entry
        symlink(source / f"{name}.bin", target / f"{name}.bin")
        for graph in entry["graphs"]:
            for path in source.glob(graph + ".*"):
                if path.is_file():
                    symlink(path, target / path.name)
        for path in source.glob("*.data"):
            symlink(path, target / path.name)
        provenance[name] = {
            "source": str(source),
            "binary_sha256": digest(source / f"{name}.bin"),
            "report_sha256": digest(source / "convert_report.json"),
        }
    combined["uniform_parts"] = provenance
    save_new(target / "convert_report.json", combined)


def evaluate(root: Path, protocol: dict) -> None:
    assert read(root / "reports/graph_audit.json")["passed"], (
        "Complete the paired graph audit before device evaluation"
    )
    for group in GROUPS:
        assert not (root / group / "runtime_manifest.json").is_symlink()
        execute(
            "verify_rotated_attention.py",
            [
                "--bundle",
                str(root / group),
                "--report",
                str(root / "reports" / f"audit_{group}.json"),
            ],
            root / "logs" / f"audit_{group}.log",
        )
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
        reset = measure(
            root,
            group,
            "reset",
            ["--mode", "generate", "--n-gen", "8", "--sessions", "2"],
            protocol,
        )
        assert reset["sessions"][0]["generated"] == reset["sessions"][1]["generated"]
    counts = dict.fromkeys(GROUPS, 0)
    for group in protocol["performance"]["order"]:
        label = f"perf_r{counts[group]}"
        counts[group] += 1
        measure(
            root,
            group,
            label,
            [
                "--mode",
                "generate",
                "--n-gen",
                "128",
                "--sessions",
                "1",
                "--tokens",
                "boundary_prompt_cl1024.bin",
            ],
            protocol,
        )
    for group in GROUPS:
        for split, prefix in (("legacy", "wikitext_w"), ("heldout", "heldout_")):
            for i in range(4):
                report = measure(
                    root,
                    group,
                    f"score_{split}_{i}",
                    ["--mode", "score", "--tokens", f"{prefix}{i}.bin"],
                    protocol,
                )
                assert report["scored_tokens"] == 1023


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("prepare", "probe-build", "probe-run", "build", "evaluate")
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-manifest", type=Path)
    p.add_argument(
        "--components",
        nargs="+",
        choices=("encoder", "decoder"),
        default=["encoder", "decoder"],
    )
    a = p.parse_args()
    root = a.root.resolve()
    if a.stage == "prepare":
        prepare(root, a.data_manifest)
    elif a.stage.startswith("probe-"):
        probes(root, a.stage.removeprefix("probe-"), a.components)
    else:
        protocol = validate(root)
        if a.stage == "build":
            build(root, protocol)
        else:
            evaluate(root, protocol)


if __name__ == "__main__":
    main()
