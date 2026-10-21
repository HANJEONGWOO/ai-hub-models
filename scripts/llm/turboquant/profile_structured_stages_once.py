# SPDX-License-Identifier: BSD-3-Clause
"""Profile only the frozen selected Structured K8 allocation, exactly once.

Historical K4/V4, K5/V3 and K6/V2 measurements are read-only controls. Reuse
the existing context builder, runner, renderer and physical-stage classifier.
No scoring, throughput benchmark, quantizer changes or allocation search.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from profile_stages_once import DEVICE_ROOT, SCRIPTS, digest, read, save_new

ARTIFACTS = Path("/mnt/d/ai-hub-models/binaries/turboquant")
GROUP = "structured_k8"
CONTROLS = ("k4_v4", "k5_v3", "k6_v2")


def validate_config(config: dict, selected: list[int]) -> None:
    bits = config["layer_plan"]["bits"]
    if len(bits) != 28 or bits.count([8, 2]) != 14 or bits.count([4, 2]) != 14:
        raise ValueError("Expected exactly 14 K8, 14 K4 and 28 V2 layers")
    if [i for i, pair in enumerate(bits) if pair[0] == 8] != selected:
        raise ValueError("The previously selected layer mask must not change")
    if (
        config["rotation"] != "dense_qr"
        or config["qjl"]
        or not config["norm_correction"]
        or config["norm_representation"] != "effective_scale"
        or config["norm_dtype"] != "float16"
        or "structured_k8" not in config
    ):
        raise ValueError("Unexpected numerical configuration")


def validate(root: Path) -> dict:
    protocol = read(root / "protocol.json")
    for name, expected in protocol["preserved_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError("Frozen input changed: " + name)
    return protocol


def freeze(root: Path, experiment: Path, historical: Path) -> None:
    if root.exists():
        raise FileExistsError("Use a new artifact root; never overwrite a run")
    selection = read(experiment / "selection.json")
    source_group = selection["selected_group"]
    bundle = experiment / source_group
    source = read(bundle / "convert_report.json")
    validate_config(source["config"], selection["selected_layers"])
    if source["attention_tile"] != 256 or not source["quantize_current_kv"]:
        raise ValueError("Tile256 and current-KV compression must be preserved")
    if source["config_hash"] != selection["selected_config_hash"]:
        raise ValueError("Selection and compiled configuration differ")
    reference = experiment / "reports" / f"{source_group}_perf_r0.json"
    previous = read(reference)
    if previous["config_hash"] != source["config_hash"]:
        raise ValueError("Historical uninstrumented reference differs")
    old_protocol = read(experiment / "protocol.json")
    runner = Path(old_protocol["runner"])
    assets = ARTIFACTS / "device_assets_cl1024"
    asset_manifest = read(assets / "assets.json")
    if digest(assets / "boundary_prompt_cl1024.bin") != digest(
        Path(old_protocol["assets"]) / "boundary_prompt_cl1024.bin"
    ):
        raise ValueError("Previous selected run used a different prompt")
    old_controls = read(historical / "reference/experiment.json")
    if old_controls["assets_sha256"] != asset_manifest["sha256"]:
        raise ValueError("Historical controls used different assets")
    paths = [
        experiment / "selection.json",
        experiment / "plans/final_selected.json",
        experiment / "protocol.json",
        experiment / "final_verification.json",
        bundle / "convert_report.json",
        reference,
        runner,
        assets / "assets.json",
        historical / "reference/experiment.json",
        historical / "comparison.json",
        historical / "stage_comparison.csv",
    ]
    paths += [assets / name for name in asset_manifest["sha256"]]
    for part, item in source["parts"].items():
        paths.append(bundle / f"{part}.bin")
        for graph in item["graphs"]:
            for suffix in (
                ".dlc",
                ".dlcinfo.txt",
                ".onnx",
                ".encodings",
                ".kv_edits.json",
            ):
                path = bundle / (graph + suffix)
                if path.exists():
                    paths.append(path)
    paths += [
        Path(lib["path"]) for lib in source["native_decoder"]["libraries"].values()
    ]
    for group in CONTROLS:
        for suffix in (
            "_optrace_summary.json",
            "_optrace_summary.csv",
            "_profile_once.json",
            "_profile_once.attempt.json",
            "_profile_once.complete.json",
        ):
            paths.append(historical / "reports" / (group + suffix))
    # Freeze numerical implementation/defaults without editing their sources.
    model_dir = Path("src/qai_hub_models/models/templates/llm/turboquant").resolve()
    paths += [
        p for p in model_dir.rglob("*") if p.suffix in (".py", ".cpp", ".h", ".xml")
    ]
    paths += [
        SCRIPTS / name
        for name in (
            "build_optrace_contexts.py",
            "profile_optrace_once.py",
            "render_optrace.py",
            "summarize_optrace.py",
            "summarize_stage_profiles.py",
            "run_device_llm.py",
        )
    ]
    preserved = {str(p): digest(p) for p in sorted(set(paths))}
    root.mkdir(parents=True)
    (root / "reference").mkdir()
    (root / "runner").mkdir()
    (root / "runner/qnn-llm-runner").symlink_to(runner)
    save_new(
        root / "reference/experiment.json",
        {
            "groups": {
                GROUP: {
                    "bundle": str(bundle),
                    "conversion_sha256": digest(bundle / "convert_report.json"),
                    "reference_report": str(reference),
                    "reference_report_sha256": digest(reference),
                }
            },
            "assets": str(assets),
            "assets_sha256": asset_manifest["sha256"],
        },
    )
    save_new(
        root / "protocol.json",
        {
            "source_experiment": str(experiment),
            "source_group": source_group,
            "source_bundle": str(bundle),
            "historical_controls": str(historical),
            "selected_layers": selection["selected_layers"],
            "config_hash": source["config_hash"],
            "assets": str(assets),
            "sessions": {GROUP: 1},
            "historical_groups_remeasured": [],
            "prompt_tokens": 897,
            "generated_tokens": 128,
            "prefill": "all eight AR128 chunks (1 valid token then seven full chunks)",
            "decode_steps": [0, 63, 126],
            "primary_decode_step": "decode_126",
            "latency_units": "SDK accelerator us and instrumented host seconds only",
            "stage_units": "physical HVX/HMX busy cycles; NOT wall time",
            "stage_time_unavailable": "No calibrated per-stage clock; time fields remain null, not invented",
            "no_ppl_or_uninstrumented_benchmark": True,
            "native_runtime_sha256": source["native_decoder"]["libraries"][
                "hexagon-v81"
            ]["sha256"],
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "preserved_sha256": preserved,
        },
    )
    print("FROZEN", GROUP, selection["selected_layers"], flush=True)


def device_space_required(root: Path) -> int:
    # Context binaries plus trace/log allowance and a safety margin. Never
    # delete historical phone files automatically to satisfy this guard.
    return (
        sum(
            (root / GROUP / f"part{p}_of_4/part{p}_of_4.bin").stat().st_size
            for p in range(1, 5)
        )
        + 512 * 1024**2
    )


def release_control_device_copy(root: Path, adb: Path, protocol: dict) -> None:
    """Explicitly authorized cleanup of four hash-verified phone copies only."""
    if (root / "device_cleanup_authorized.json").exists():
        raise FileExistsError("Cleanup already attempted; inspect its record")
    historical = Path(protocol["historical_controls"])
    if historical.name != "asymmetric_optrace_20261010":
        raise ValueError("Cleanup is restricted to the explicitly inspected old run")
    remote = DEVICE_ROOT + "/bundles/asymmetric_optrace_20261010_k4_v4"
    records = []
    for part in range(1, 5):
        name = f"part{part}_of_4.bin"
        local = historical / "k4_v4" / f"part{part}_of_4" / name
        expected = read(local.parent / "complete.json")["context_sha256"]
        if digest(local) != expected:
            raise ValueError("Local backup differs: " + str(local))
        target = remote + "/" + name
        subprocess.run(
            [str(adb), "shell", f"test -f {target} && test ! -L {target}"], check=True
        )
        actual = subprocess.check_output(
            [str(adb), "shell", "sha256sum", target], text=True
        ).split()[0]
        if actual != expected:
            raise ValueError("Phone and local backup differ: " + target)
        records.append(
            {
                "remote": target,
                "local_backup": str(local),
                "sha256": expected,
                "bytes": local.stat().st_size,
            }
        )
    save_new(
        root / "device_cleanup_authorized.json",
        {
            "authorization": "User explicitly allowed removal of previous phone experiment copies after PC backup/hash verification",
            "targets": records,
            "not_removed": "PC originals, phone reports/logs/manifests/native libraries, other bundles",
            "restore": "adb push each local_backup to its recorded remote path after wslpath -w conversion",
        },
    )
    # All four exact targets have just been checked. No glob, recursive delete,
    # broad directory or unresolved variable is passed to the device shell.
    subprocess.run(
        [str(adb), "shell", "rm", "--", *[r["remote"] for r in records]], check=True
    )
    for record in records:
        subprocess.run(
            [str(adb), "shell", "test", "!", "-e", record["remote"]], check=True
        )
        if digest(Path(record["local_backup"])) != record["sha256"]:
            raise ValueError("Local backup changed during cleanup")
    save_new(
        root / "device_cleanup_complete.json",
        {
            "freed_file_bytes": sum(r["bytes"] for r in records),
            "restorable_from_verified_pc_copies": True,
            "device_df": subprocess.check_output(
                [str(adb), "shell", "df", "-k", DEVICE_ROOT], text=True
            ),
        },
    )
    print(
        "Removed four backed-up old phone contexts; PC originals preserved", flush=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "freeze",
            "build",
            "measure",
            "render",
            "summarize",
            "verify",
            "release-control-device-copy",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=ARTIFACTS / "structured_k8_layer_search_v2_20261020",
    )
    parser.add_argument(
        "--historical", type=Path, default=ARTIFACTS / "asymmetric_optrace_20261010"
    )
    parser.add_argument("--adb", type=Path, default=Path("/mnt/c/adb/adb.exe"))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "freeze":
        freeze(root, args.experiment.resolve(), args.historical.resolve())
        return
    protocol = validate(root)

    def run(script: str, *options: str, python: str = sys.executable) -> None:
        subprocess.run([python, str(SCRIPTS / script), *options], check=True)

    if args.action == "release-control-device-copy":
        release_control_device_copy(root, args.adb, protocol)
    elif args.action == "build":

        def build(part: int) -> None:
            run(
                "build_optrace_contexts.py",
                "--source",
                protocol["source_bundle"],
                "--out",
                str(root / GROUP),
                "--parts",
                str(part),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(build, range(1, 5)))
        validate(root)
    elif args.action == "measure":
        df = subprocess.check_output(
            [str(args.adb), "shell", "df", "-k", DEVICE_ROOT], text=True
        )
        available = int(df.strip().splitlines()[-1].split()[3]) * 1024
        required = device_space_required(root)
        if available < required:
            raise RuntimeError(
                f"Device space {available / 1024**3:.3f} GiB < required {required / 1024**3:.3f} GiB. No files deleted; no inference attempted."
            )
        run(
            "profile_optrace_once.py",
            "measure",
            "--root",
            str(root),
            "--reference-reports",
            str(root / "reference"),
            "--assets",
            protocol["assets"],
            "--groups",
            GROUP,
            "--adb",
            str(args.adb),
        )
    elif args.action == "render":
        run(
            "render_optrace.py",
            "--logs",
            str(root / "traces" / GROUP),
            "--contexts",
            str(root / GROUP),
            "--out",
            str(root / "rendered" / GROUP),
            "--jobs",
            "4",
            python="/home/hjw/qnn-venv/bin/python",
        )
    elif args.action == "summarize":
        summary = root / "reports" / f"{GROUP}_optrace_summary.json"
        if not summary.exists():
            run("summarize_optrace.py", "--root", str(root), "--groups", GROUP)
        run("summarize_structured_stages.py", "--root", str(root))
    else:
        report_path = root / "reports" / f"{GROUP}_profile_once.json"
        report = read(report_path)
        complete = read(root / "reports" / f"{GROUP}_profile_once.complete.json")
        if digest(report_path) != complete["report_sha256"]:
            raise ValueError("Completed raw report changed")
        for filename, expected in complete["logs_sha256"].items():
            if digest(root / "traces" / GROUP / filename) != expected:
                raise ValueError("Raw trace changed: " + filename)
        if not complete["generated_ids_match_reference"]:
            raise ValueError("Profiled continuation differs from unchanged reference")
        if len(report["op_profiles"]) != 11 or len(complete["logs_sha256"]) != 44:
            raise ValueError("Expected exactly 11 samples / 44 graph traces")
        comparison = read(root / "comparison.json")
        if len(comparison["seven_stages"]) != 28:
            raise ValueError("Expected seven stages times four groups")
        if report["kv_store_bytes"] != 30277632:
            raise ValueError("Unexpected KV memory budget")
        if report["native_decoder"]["sha256"] != protocol["native_runtime_sha256"]:
            raise ValueError("Unexpected device Native runtime")
        cleanup = root / "device_cleanup_authorized.json"
        if cleanup.exists():
            for record in read(cleanup)["targets"]:
                if digest(Path(record["local_backup"])) != record["sha256"]:
                    raise ValueError("Recovery copy changed: " + record["local_backup"])
        suites = ET.parse(root / "regression_all.xml").getroot().findall("testsuite")
        test_count = sum(int(s.attrib["tests"]) for s in suites)
        if not test_count or any(
            int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped")
        ):
            raise ValueError("Regression tests did not fully pass")
        if (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != protocol["head"]
        ):
            raise ValueError("Unexpected commit during experiment")
        save_new(
            root / "final_verification.json",
            {
                "passed": True,
                "generated_ids_match_reference": True,
                "new_sessions": {GROUP: 1},
                "control_sessions": 0,
                "preserved_files": len(protocol["preserved_sha256"]),
                "regression_tests": test_count,
                "artifacts_sha256": {
                    str(p.relative_to(root)): digest(p)
                    for p in sorted(root.rglob("*"))
                    if p.is_file()
                },
                "scripts_sha256": {
                    name: digest(SCRIPTS / name)
                    for name in (Path(__file__).name, "summarize_structured_stages.py")
                },
                "documentation_sha256": digest(SCRIPTS / "README.md"),
                "test_source_sha256": digest(
                    SCRIPTS.parents[2]
                    / "src/qai_hub_models/test/test_models/test_turboquant_structured_stage_profile.py"
                ),
                "automatic_commit": False,
            },
        )
        print(
            "VERIFIED one Structured session; historical inputs/defaults unchanged",
            flush=True,
        )


if __name__ == "__main__":
    main()
