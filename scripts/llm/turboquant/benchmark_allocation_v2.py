# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""HTP builds, validation-only selection, then frozen test and crossed timings."""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark_asymmetric_once import ARTIFACTS, digest, execute, read, save_new
from benchmark_nll_allocation import (
    STATIC,
    audit_allocation,
    build_group,
    push,
    run_once,
    symlink,
)
from prepare_allocation_v2 import LATEST, OLD

from qai_hub_models.models.templates.llm.turboquant.config import (
    STRUCTURED_ALLOCATION_PROFILE,
    TurboQuantConfig,
    get_profile,
    structured_allocation_plan,
)

STATIC_SOURCES = {
    "uniform_odd": LATEST / "structured",
    "early": OLD / "early",
    "boundary": OLD / "boundary",
    "k6_v2": ARTIFACTS / "asymmetric_hvx_20261010/k6_v2",
}


def config(root: Path, group: str) -> TurboQuantConfig:
    return (
        get_profile("k6_v2_scaled")
        if group == "k6_v2"
        else get_profile(
            STRUCTURED_ALLOCATION_PROFILE,
            layer_plan=read(root / "plans" / f"{group}.json"),
        )
    )


def physical_identity(value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("profile")
    if "layer_plan" in value:
        value["layer_plan"].pop("policy")
    return value


def validate(root: Path) -> dict:
    protocol = read(root / "protocol.json")
    for p, sha in protocol["frozen_files"].items():
        assert digest(Path(p)) == sha, p
    for p, sha in read(root / "source_identity.json")["files_sha256"].items():
        assert digest(Path(p)) == sha, p
    return protocol


def clone_source(root: Path, group: str, source: Path, protocol: dict) -> bool:
    """Reuse immutable inputs only. Never symlink runtime reports or rebuilt bins."""
    target = root / group
    target.mkdir(exist_ok=False)
    old = read(source / "convert_report.json")
    c = config(root, group)
    assert physical_identity(old["config"]) == physical_identity(c.to_dict())
    native = read(Path(protocol["native"]) / "manifest.json")
    reuse_bins = old["native_decoder"] == native
    files = [source / "convert_report.json"]
    for p in sorted(source.iterdir()):
        immutable = p.suffix in (
            ".onnx",
            ".encodings",
            ".dlc",
            ".data",
        ) or p.name.endswith((".kv_edits.json", ".dlcinfo.txt"))
        binary = p.name.startswith("part") and p.suffix == ".bin"
        if p.is_file() and (immutable or binary):
            files.append(p)
            if immutable or reuse_bins:
                symlink(p, target / p.name)
    save_new(
        root / "reports" / f"reuse_{group}.json",
        {
            "source": str(source),
            "source_files_sha256": {str(p): digest(p) for p in files},
            "reused_contexts": reuse_bins,
            "reason": "Same physical config/weights. Different Native package requires private context rebuild; profile/policy are cache identity only.",
        },
    )
    report = copy.deepcopy(old)
    report.update(
        profile=c.profile,
        config=c.to_dict(),
        config_hash=c.config_hash(),
        native_decoder=native,
        allocation_v2_source=str(source),
    )
    if group != "k6_v2":
        report["layer_plan"] = str(root / "plans" / f"{group}.json")
    save_new(target / "convert_report.json", report)
    hydrate_graph_sources(root, group, source)
    return reuse_bins


def hydrate_graph_sources(root: Path, group: str, source: Path) -> None:
    """Some historical bundles contain bins only; graph files live by each bin."""
    record = root / "reports" / f"graph_sources_{group}.json"
    if record.exists():
        for p, sha in read(record)["files_sha256"].items():
            assert digest(Path(p)) == sha, p
        return
    report = read(source / "convert_report.json")
    added = {}
    for part, info in report["parts"].items():
        directory = (source / (part + ".bin")).resolve(strict=True).parent
        inputs = list(directory.glob("*.data"))
        for graph in info["graphs"]:
            for suffix in (
                ".onnx",
                ".encodings",
                ".dlc",
                ".kv_edits.json",
                ".dlcinfo.txt",
            ):
                destination = root / group / (graph + suffix)
                if not destination.exists():
                    candidate = directory / destination.name
                    if candidate.is_file():
                        inputs.append(candidate)
        for p in inputs:
            destination = root / group / p.name
            if destination.exists():
                assert destination.resolve() == p.resolve() or digest(
                    destination
                ) == digest(p)
            else:
                symlink(p, destination)
                added[str(p)] = digest(p)
        for graph in info["graphs"]:
            assert (root / group / (graph + ".dlc")).is_file(), graph
    save_new(record, {"source": str(source), "files_sha256": added})


def audit(root: Path, group: str) -> None:
    if not (root / "reports" / f"audit_{group}.json").exists():
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
    assert read(root / "reports" / f"audit_{group}.json")["passed"]
    if group != "k6_v2":
        audit_allocation(root, group)


def rebuild_context(root: Path, group: str, protocol: dict) -> None:
    c = config(root, group)
    missing = [
        n for n in range(1, 5) if not (root / group / f"part{n}_of_4.bin").exists()
    ]
    if not missing:
        return
    options = [
        "--split-dir",
        protocol["split"],
        "--out",
        str(root / group),
        "--profile",
        c.profile,
        "--context-length",
        "1024",
        "--sequence-lengths",
        "1",
        "128",
        "--native-decoder-package",
        protocol["native"],
        "--context-only",
        "--parts",
        *map(str, missing),
    ]
    if group != "k6_v2":
        options += [
            "--layer-plan",
            str(root / "plans" / f"{group}.json"),
            "--structured-k8-decoder",
            "arithmetic",
        ]
    log = root / "logs" / f"context_{group}.log"
    attempt = 1
    while log.exists():
        attempt += 1
        log = root / "logs" / f"context_{group}_attempt{attempt}.log"
    execute("convert_parts.py", options, log)


def ready(root: Path, group: str) -> None:
    audit(root, group)
    bins = {p.name: digest(p) for p in sorted((root / group).glob("part*_of_4.bin"))}
    assert len(bins) == 4
    save_new(
        root / "reports" / f"ready_{group}.json",
        {"config_hash": config(root, group).config_hash(), "binaries_sha256": bins},
    )


def static(root: Path, protocol: dict) -> None:
    jobs = []
    for group, source in STATIC_SOURCES.items():
        if (root / "reports" / f"ready_{group}.json").exists():
            continue
        if (root / group / "convert_report.json").exists():
            report = read(root / group / "convert_report.json")
            assert report["config_hash"] == config(root, group).config_hash()
            assert report["native_decoder"] == read(
                Path(protocol["native"]) / "manifest.json"
            )
            hydrate_graph_sources(root, group, source)
            reuse = all(
                (root / group / f"part{n}_of_4.bin").is_file() for n in range(1, 5)
            )
        else:
            reuse = clone_source(root, group, source, protocol)
        if reuse:
            ready(root, group)
        else:
            jobs.append(group)

    def job(group: str) -> None:
        rebuild_context(root, group, protocol)
        ready(root, group)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(job, jobs))


def ensure_built(root: Path, name: str, protocol: dict) -> None:
    c = config(root, name)
    if (root / "reports" / f"ready_{name}.json").exists():
        previous = read(root / "reports" / f"ready_{name}.json")
        assert previous["config_hash"] == c.config_hash()
        assert len(previous["binaries_sha256"]) == 4
        for filename, sha in previous["binaries_sha256"].items():
            assert digest(root / name / filename) == sha
        return
    # Reuse an old exact allocation if it exists, never substitute a nearby mask.
    source = next(
        (
            OLD / n
            for n in [*STATIC, "nll_candidate"]
            if read(OLD / n / "convert_report.json")["config_hash"] == c.config_hash()
        ),
        None,
    )
    if source:
        if not clone_source(root, name, source, protocol):
            rebuild_context(root, name, protocol)
    else:
        build_group(root, name, protocol)
    ready(root, name)


def prebuild(root: Path, method: str, protocol: dict) -> None:
    """Build only an already-fixed method nominee, never a provisional winner."""
    assert method in ("legacy_greedy", "multi_greedy")
    starts = (
        ["uniform_odd"] if method == "legacy_greedy" else protocol["search"]["starts"]
    )
    paths = [root / "offline" / f"greedy_{n}.json" for n in starts]
    results = [read(p) for p in paths]  # Requires every applicable start completed.
    identity = digest(root / "offline/scorer_identity.json")
    assert all(
        r["run"] == "greedy_" + n
        and r["scorer_identity"] == identity
        and r["cost"]["incomplete_attempts"] == 0
        for n, r in zip(starts, results, strict=True)
    )
    winner = min(
        results,
        key=lambda r: (r["final_nll"], sum(1 << i for i in r["final_layers"])),
    )
    plan = structured_allocation_plan(winner["final_layers"])
    path = root / "plans" / f"{method}.json"
    if path.exists():
        assert read(path) == plan
    else:
        save_new(path, plan)
    record = {
        "completed_run_files_sha256": {str(p): digest(p) for p in paths},
        "source_run": winner["run"],
        "config_hash": plan["config_hash"],
        "reason": "This method's nominee is fixed by completed search only. No pending beam result or validation/test score is used.",
    }
    prior = root / "reports" / f"prebuild_{method}.json"
    if prior.exists():
        assert read(prior) == record
    else:
        save_new(prior, record)
    for marker in (root / "reports").glob("ready_*.json"):
        if read(marker)["config_hash"] == plan["config_hash"]:
            print("Already built identical config:", marker.stem, flush=True)
            return
    ensure_built(root, method, protocol)


def candidates(root: Path, protocol: dict) -> None:
    nominated = read(root / "nominations.json")["nominees"]
    names = [
        *protocol["evaluation"]["groups"],
        *[n for n in nominated if n.startswith("extra_")],
    ]
    aliases, seen = {}, {}
    for name in names:
        c = config(root, name)
        if c.config_hash() in seen:
            aliases[name] = seen[c.config_hash()]
            continue
        aliases[name] = name
        seen[c.config_hash()] = name
        ensure_built(root, name, protocol)
    save_new(
        root / "physical_groups.json",
        {
            "aliases": aliases,
            "groups": list(seen.values()),
            "htp_nominees": list(dict.fromkeys(aliases[n] for n in nominated)),
            "nominations_sha256": digest(root / "nominations.json"),
        },
    )


def ensure_push(root: Path, group: str, protocol: dict) -> None:
    path = root / "reports" / f"pushed_{group}.json"
    if path.exists():
        assert read(path)["ready_sha256"] == digest(
            root / "reports" / f"ready_{group}.json"
        )
        return
    stage_shared_part(root, group)
    push(root, group, protocol)
    save_new(path, {"ready_sha256": digest(root / "reports" / f"ready_{group}.json")})


def stage_shared_part(root: Path, group: str) -> None:
    """Share only this experiment's byte-identical embedding part on the device."""
    name = root.name + "_" + group
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Unsafe device bundle name")
    remote = "/data/local/tmp/qaihm_turboquant/llm/bundles/" + name

    def shell(*args: str, check: bool = True) -> str:
        return subprocess.run(
            ["/mnt/c/adb/adb.exe", "shell", *args],
            check=check,
            capture_output=True,
            text=True,
        ).stdout.strip()

    shell("mkdir", "-p", remote)
    ready = read(root / "reports" / f"ready_{group}.json")
    shared = False
    if group != "uniform_odd":
        anchor = (
            "/data/local/tmp/qaihm_turboquant/llm/bundles/"
            + root.name
            + "_uniform_odd/part1_of_4.bin"
        )
        expected = ready["binaries_sha256"]["part1_of_4.bin"]
        assert (
            read(root / "reports/ready_uniform_odd.json")["binaries_sha256"][
                "part1_of_4.bin"
            ]
            == expected
        )
        observed = shell("sha256sum", anchor).split()
        assert observed and observed[0] == expected
        target = remote + "/part1_of_4.bin"
        existing = shell("sha256sum", target, check=False).split()
        if existing:
            assert existing[0] == expected, "Never overwrite a shared device binary"
        else:
            # Android denies hard links for shell_data_file on this device.
            # Read-only shared inputs can use ordinary symlinks without changing
            # permissions, SELinux policy, or any pre-existing experiment file.
            shell("ln", "-s", anchor, target)
        shared = True
    missing_bytes = 0
    for filename, sha in ready["binaries_sha256"].items():
        observed = shell("sha256sum", remote + "/" + filename, check=False).split()
        if observed:
            assert observed[0] == sha, "Refuse changing an existing device binary"
        else:
            missing_bytes += (root / group / filename).stat().st_size
    free_bytes = int(shell("df", "-k", remote).splitlines()[-1].split()[3]) * 1024
    if free_bytes < missing_bytes + 256 * 2**20:
        raise RuntimeError("Insufficient device space; preserve all previous artifacts")
    report = root / "reports" / f"staging_{group}.json"
    if not report.exists():
        save_new(
            report,
            {
                "remote_bundle": remote,
                "shared_part1_with_this_experiment_only": shared,
                "free_bytes_before_push": free_bytes,
                "missing_binary_bytes": missing_bytes,
                "previous_device_artifacts_deleted": False,
            },
        )


def score_set(root: Path, group: str, role: str, protocol: dict) -> list[dict]:
    if role in ("test", "legacy"):
        selection = read(root / "selection.json")
        if selection["test_used"]:
            raise ValueError("Final selection must not use independent test scores")
    elif role not in ("search", "validation"):
        raise ValueError("Unknown evaluation split")
    data = read(root / "data_manifest.json")
    rows = (
        [{"file": n} for n in protocol["assets_manifest"]["wikitext_windows"]]
        if role == "legacy"
        else data["windows"][role]
    )
    result = []
    for i, row in enumerate(rows):
        label = f"{role}_{i:02d}"
        path = root / "reports" / f"{group}_{label}.json"
        if path.exists():
            report = read(path)
            assert report["config_hash"] == config(root, group).config_hash()
            assert (
                report["assets"]["tokens_sha256"]
                == protocol["assets_manifest"]["sha256"][row["file"]]
            )
        else:
            report = run_once(
                root,
                group,
                label,
                ["--mode", "score", "--tokens", row["file"]],
                protocol,
            )
        assert report["scored_tokens"] == 1023 and report["kv_store_bytes"] == 30277632
        result.append(report)
    return result


def calibrate(root: Path, protocol: dict) -> None:
    if (root / "selection.json").exists():
        raise FileExistsError(root / "selection.json")
    groups = read(root / "physical_groups.json")
    tested = list(dict.fromkeys(["uniform_odd", *groups["htp_nominees"]]))
    scores = {}
    for group in tested:
        ensure_push(root, group, protocol)
        by_role = {}
        for role in ("search", "validation"):
            rows = score_set(root, group, role, protocol)
            by_role[role] = sum(r["nll_sum"] for r in rows) / sum(
                r["scored_tokens"] for r in rows
            )
        scores[group] = by_role
    epsilon = protocol["search"]["epsilon"]
    minimum = min(x["validation"] for x in scores.values())
    tied = [g for g, n in scores.items() if n["validation"] <= minimum + epsilon]
    selected = (
        "uniform_odd"
        if "uniform_odd" in tied
        else min(
            tied,
            key=lambda g: sum(
                1 << i for i in read(root / "plans" / f"{g}.json")["k8_layers"]
            ),
        )
    )
    plan = read(root / "plans" / f"{selected}.json")
    save_new(root / "plans/final_selected.json", plan)
    save_new(
        root / "selection.json",
        {
            "selected_group": selected,
            "selected_layers": plan["k8_layers"],
            "selected_config_hash": plan["config_hash"],
            "htp_mean_nll": scores,
            "nominations_sha256": digest(root / "nominations.json"),
            "physical_groups_sha256": digest(root / "physical_groups.json"),
            "criterion": "HTP validation-4 mean token NLL; search-12 is diagnostic only",
            "test_used": False,
        },
    )
    print("SELECTED", selected, plan["k8_layers"], scores[selected], flush=True)


def check_reset(root: Path, group: str, protocol: dict) -> None:
    ensure_push(root, group, protocol)
    path = root / "reports" / f"{group}_reset.json"
    result = (
        read(path)
        if path.exists()
        else run_once(
            root,
            group,
            "reset",
            ["--mode", "generate", "--n-gen", "8", "--sessions", "2"],
            protocol,
        )
    )
    assert result["config_hash"] == config(root, group).config_hash()
    assert result["kv_store_bytes"] == 30277632 and result["quantize_current_kv"]
    assert result["sessions"][0]["generated"] == result["sessions"][1]["generated"]


def probe(root: Path, protocol: dict) -> None:
    # No new calibration/test document is consumed and no quality selection.
    for group in STATIC_SOURCES:
        assert (root / "reports" / f"ready_{group}.json").exists()
        check_reset(root, group, protocol)


def evaluate(root: Path, protocol: dict) -> None:
    selection = read(root / "selection.json")
    assert not selection["test_used"]
    groups = read(root / "physical_groups.json")["groups"]
    rounds = [groups, list(reversed(groups)), groups[3:] + groups[:3]]
    order = [g for round_ in rounds for g in round_]
    order_path = root / "device_order.json"
    recorded = {
        "rounds": rounds,
        "order": order,
        "selection_sha256": digest(root / "selection.json"),
    }
    if order_path.exists():
        assert read(order_path) == recorded
    else:
        save_new(order_path, recorded)
    for group in groups:
        check_reset(root, group, protocol)
    counts = dict.fromkeys(groups, 0)
    for group in order:
        label = f"perf_r{counts[group]}"
        counts[group] += 1
        if not (root / "reports" / f"{group}_{label}.json").exists():
            run_once(
                root,
                group,
                label,
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
    for group in groups:
        score_set(root, group, "test", protocol)
        score_set(root, group, "legacy", protocol)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=(
            "static",
            "probe",
            "prebuild-legacy",
            "prebuild-greedy",
            "candidates",
            "calibrate",
            "evaluate",
        ),
    )
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    protocol = validate(root)
    {
        "static": static,
        "probe": probe,
        "prebuild-legacy": lambda r, p: prebuild(r, "legacy_greedy", p),
        "prebuild-greedy": lambda r, p: prebuild(r, "multi_greedy", p),
        "candidates": candidates,
        "calibrate": calibrate,
        "evaluate": evaluate,
    }[args.stage](root, protocol)


if __name__ == "__main__":
    main()
