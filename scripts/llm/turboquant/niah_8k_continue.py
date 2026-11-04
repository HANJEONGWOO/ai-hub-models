# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate the approved 1K/2K/4K/8K scope without further 16K work.

The original 16K protocol, inputs, completed probes and failed build artifacts
remain unchanged. A separate immutable scope amendment selects the existing
inputs. An isolated 2K supplement adds 50 inputs; the unchanged runner evaluates
800 main cases once. Only 2K is built, serially, before any main evaluation.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import niah_continue as original
import niah_oom_resume as recovery
import niah_resume_build as resume

CAPACITIES = (1024, 2048, 4096, 8192)
PRESERVED_CAPACITIES = (1024, 4096, 8192)
GROUPS = original.GROUPS
SCOPE_NAME = "scope_8k_20261103"
SUPPLEMENT_NAME = "niah_supplement_cl2048_20261103"
EXPECTED_CASES = {"test": 200, "pilot": 4, "smoke": 4}
MAIN_RUNS = 800
CONTROLLERS = {
    "niah_continue.py",
    "niah_oom_resume.py",
    "niah_swap_resume.py",
    "niah_8k_continue.py",
    "benchmark_niah.py",
    "niah_device_copies.py",
    "niah_2k_supplement.py",
    "niah_2k_device_copies.py",
}


def selected_manifest(data: dict, supplement: dict, source_hashes: dict[str, str]) -> dict:
    """Select unchanged case objects; never regenerate or retokenize inputs."""
    cases = [c for c in data["cases"] if c["capacity"] in PRESERVED_CAPACITIES]
    added = supplement["cases"]
    if any(c["capacity"] != 2048 for c in added):
        raise ValueError("Supplement contains a non-2K case")
    cases += added
    counts = dict(collections.Counter(c["split"] for c in cases))
    if counts != EXPECTED_CASES:
        raise ValueError(f"Unexpected selected case counts: {counts}")
    if len({c["case_id"] for c in cases}) != sum(EXPECTED_CASES.values()):
        raise ValueError("Selected input IDs are not unique")
    for capacity in CAPACITIES:
        if sum(c["split"] == "test" and c["capacity"] == capacity for c in cases) != 50:
            raise ValueError("Each selected capacity must retain all 50 main cases")
    return {
        **data,
        "protocol": {**data["protocol"], "capacities": list(CAPACITIES)},
        "cases": cases,
        "case_counts": counts,
        "scope_amendment": {
            "source_manifests_sha256": source_hashes,
            "original_capacities": list(original.CAPACITIES),
            "excluded_capacity": 16384,
            "reason": "User stopped 16K, approved evaluation up to 8K, then added 2K",
            "existing_inputs_changed": False,
            "new_capacity": 2048,
            "new_pilots": "Two separate 2K operational pilots from the frozen pilot bases",
            "main_runs": MAIN_RUNS,
        },
    }


def ensure_idle(root: Path, supplement_root: Path) -> None:
    observed = resume.processes()
    for target_root in (root, supplement_root):
        if resume.build_processes(target_root, observed):
            raise RuntimeError("Host converter/QNN writers remain active")
    for process in observed:
        script = Path(process.argv[1]).name if len(process.argv) > 1 else ""
        target = resume.option(process.argv, "--root")
        if script not in CONTROLLERS or not target or Path(target).resolve() not in (root, supplement_root):
            continue
        if process.pid == os.getppid() and "--detach" in process.argv:
            continue
        raise RuntimeError(f"NIAH controller/runtime remains active: PID {process.pid}")


def reject_excluded_results(root: Path, capacities: tuple[int, ...]) -> None:
    for path in sorted((root / "results").glob("*.json")):
        value = original.read(path)
        records = value if isinstance(value, list) else value.get("records", [value])
        for result in records:
            if result.get("split", "test") == "test" and result.get("capacity") not in capacities:
                raise ValueError(f"Main result outside the approved 8K subset: {path}")


def preserved_16k_metadata(root: Path) -> list[dict]:
    """Record prior recovery ledgers only; never move/hash large partial outputs."""
    records = []
    for directory in sorted(root.glob("recovery_*")):
        if not directory.is_dir():
            continue
        files = [p for p in directory.iterdir() if p.is_file() and p.suffix == ".json"]
        records.append({
            "directory": str(directory.relative_to(root)),
            "records_sha256": {p.name: original.digest(p) for p in sorted(files)},
            "policy": "Preserved unchanged; no further 16K build or archive operation",
        })
    return records


def source_hashes(root: Path, supplement_root: Path) -> dict[str, str]:
    files = [
        Path(__file__).resolve(),
        *[original.HERE / name for name in (
            "niah_continue.py", "niah_oom_resume.py", "niah_resume_build.py",
            "niah_device_copies.py", "benchmark_niah.py", "niah_report.py",
            "niah_2k_supplement.py", "niah_2k_device_copies.py",
        )],
        supplement_root / "protocol.json",
        supplement_root / "data/data_manifest.json",
        *[root / name for name in (
            "protocol.json", "data/data_manifest.json", "execution_freeze.json",
            "continuation_contract.json",
        )],
    ]
    return {str(p.resolve()): original.digest(p) for p in files}


def merged_manifest(root: Path, supplement_root: Path) -> dict:
    sources = [p / "data/data_manifest.json" for p in (root, supplement_root)]
    return selected_manifest(original.read(sources[0]), original.read(sources[1]),
                             {str(p): original.digest(p) for p in sources})


def verify_supplement(root: Path, supplement_root: Path) -> None:
    import niah_2k_supplement as supplement

    if supplement_root != root / SUPPLEMENT_NAME or supplement_root.is_symlink():
        raise ValueError("Unexpected isolated 2K supplement root")
    supplement.verify_supplement(supplement_root)


def create_scope(root: Path, supplement_root: Path, python: Path, authorization: str) -> Path:
    if not authorization.strip():
        raise ValueError("Explicit user authorization is required")
    recovery.validate_original(root, python)
    verify_supplement(root, supplement_root)
    reject_excluded_results(root, PRESERVED_CAPACITIES)
    reject_excluded_results(supplement_root, (2048,))
    selected = merged_manifest(root, supplement_root)
    scope = recovery.safe_path(root, SCOPE_NAME, exists=False)
    scope.mkdir(exist_ok=False)
    manifest = scope / "selected_manifest.json"
    original.save(manifest, selected)
    original.save(scope / "amendment.json", {
        "utc": original.now(),
        "authorization": authorization,
        "capacities": list(CAPACITIES),
        "groups_in_fixed_order": list(GROUPS),
        "main_runs": MAIN_RUNS,
        "supplement_root": str(supplement_root),
        "python": str(python.resolve(strict=True)),
        "files_sha256": source_hashes(root, supplement_root),
        "selected_manifest_sha256": original.digest(manifest),
        "preserved_16k_recovery_metadata": preserved_16k_metadata(root),
        "policy": "Only serial 2K build; preserve 12 old probes/pilot; 2K pilot + 4 probes; all 16 active probes before 800 main runs; no retries or 16K work",
    })
    return scope


def verify_scope(root: Path, supplement_root: Path, python: Path, scope: Path) -> dict:
    if scope != recovery.safe_path(root, SCOPE_NAME):
        raise ValueError("Unexpected 8K scope directory")
    amendment = original.read(scope / "amendment.json")
    if (
        not amendment.get("authorization", "").strip()
        or amendment.get("capacities") != list(CAPACITIES)
        or amendment.get("groups_in_fixed_order") != list(GROUPS)
        or amendment.get("main_runs") != MAIN_RUNS
        or amendment.get("supplement_root") != str(supplement_root)
        or amendment.get("python") != str(python.resolve(strict=True))
        or amendment.get("files_sha256") != source_hashes(root, supplement_root)
        or amendment.get("selected_manifest_sha256") != original.digest(scope / "selected_manifest.json")
    ):
        raise ValueError("8K scope/source hash mismatch")
    if original.read(scope / "selected_manifest.json") != merged_manifest(root, supplement_root):
        raise ValueError("Selected manifest is not the unchanged frozen subset")
    recovery.validate_original(root, python)
    verify_supplement(root, supplement_root)
    reject_excluded_results(root, PRESERVED_CAPACITIES)
    reject_excluded_results(supplement_root, (2048,))
    return amendment


class Controller(original.Controller):
    def __init__(self, root: Path, supplement_root: Path, python: Path, scope: Path) -> None:
        super().__init__(root, python, workers=1)
        self.scope, self.supplement_root = scope, supplement_root
        self.supplement_data: dict = {}

    def target_root(self, capacity: int) -> Path:
        if capacity not in CAPACITIES:
            raise ValueError("Capacity exceeds the approved scope")
        return self.supplement_root if capacity == 2048 else self.root

    def target_data(self, capacity: int) -> dict:
        return self.supplement_data if capacity == 2048 else self.data

    def args(self, capacity: int, group: str | None = None) -> list[str]:
        result = ["--root", str(self.target_root(capacity)), "--capacity", str(capacity)]
        return result + (["--group", group] if group is not None else [])

    def contract(self) -> None:
        amendment = verify_scope(self.root, self.supplement_root, self.python, self.scope)
        original.save(self.run_dir / "scope_amendment.json", {
            "scope_directory": str(self.scope),
            "amendment_sha256": original.digest(self.scope / "amendment.json"),
            "selected_manifest_sha256": amendment["selected_manifest_sha256"],
            "main_runs": MAIN_RUNS,
        })

    def command(self, label: str, script: str, *options: str) -> None:
        if script not in ("benchmark_niah.py", "niah_device_copies.py",
                          "niah_2k_supplement.py", "niah_2k_device_copies.py"):
            raise ValueError("Only scoped 2K build and unchanged evaluation commands are allowed")
        if "--root" not in options or "--capacity" not in options:
            raise ValueError("Every command must declare its root and capacity")
        capacity = int(options[options.index("--capacity") + 1])
        root = Path(options[options.index("--root") + 1])
        if root != self.target_root(capacity):
            raise ValueError("Command root/capacity mismatch")
        if script in ("niah_2k_supplement.py", "niah_2k_device_copies.py") and capacity != 2048:
            raise ValueError("2K adapter cannot operate on preserved capacities")
        if script == "niah_device_copies.py" and capacity == 2048:
            raise ValueError("2K requires the scoped device-copy adapter")
        if script == "niah_2k_supplement.py":
            if options[0] not in ("build", "audit"):
                raise ValueError("Supplement must already be prepared")
            if options[0] == "build" and ("--workers" not in options or options[options.index("--workers") + 1] != "1"):
                raise ValueError("Only serial 2K builds are authorized")
        if script == "benchmark_niah.py" and capacity != 2048 and options[0] not in ("measure", "deploy"):
            raise ValueError("Existing pilot/probes/assets must not be rerun")
        super().command(label, script, *options)

    def preserved_probes(self) -> None:
        self.current = "preserved_12_probes_gate"
        original.verify_audit(self.root, 1024, "fp16")
        if not original.stage_complete(self.root, self.data, "pilot", 1024, "fp16"):
            raise ValueError("Existing FP16 pilot is required; no automatic rerun")
        for capacity in PRESERVED_CAPACITIES:
            for group in GROUPS:
                original.verify_audit(self.root, capacity, group)
                if not original.stage_complete(self.root, self.data, "probe", capacity, group):
                    raise ValueError(f"All 12 preserved probes required: CL{capacity} {group}")

    def all_probes(self) -> None:
        self.current = "all_16_active_probes_gate"
        self.preserved_probes()
        if not original.stage_complete(self.supplement_root, self.supplement_data, "pilot", 2048, "fp16"):
            raise ValueError("The 2K FP16 pilot gate is missing")
        for group in GROUPS:
            original.verify_audit(self.supplement_root, 2048, group)
            if not original.stage_complete(self.supplement_root, self.supplement_data, "probe", 2048, group):
                raise ValueError("All 16 active probes must pass before main evaluation")
        original.save(self.run_dir / "verified_gates.json", {
            "utc": original.now(), "pilots_verified_at_capacities": [1024, 2048],
            "probes": [{"capacity": c, "group": g} for c in CAPACITIES for g in GROUPS],
            "main_runs": MAIN_RUNS,
        })

    def deploy(self, capacity: int, group: str) -> None:
        root = self.target_root(capacity)
        existing = root / "deployments" / f"cl{capacity}_{group}.json"
        if existing.exists():
            script = "niah_2k_device_copies.py" if capacity == 2048 else "niah_device_copies.py"
            self.command(f"restore_cl{capacity}_{group}", script, "restore", *self.args(capacity, group))
        else:
            self.command(f"deploy_cl{capacity}_{group}", "benchmark_niah.py", "deploy", *self.args(capacity, group))

    def release(self, capacity: int, group: str) -> None:
        script = "niah_2k_device_copies.py" if capacity == 2048 else "niah_device_copies.py"
        self.command(f"release_cl{capacity}_{group}", script, "release", *self.args(capacity, group), "--if-present")

    def block(self, stage: str, capacity: int, group: str) -> None:
        root, data = self.target_root(capacity), self.target_data(capacity)
        if not original.stage_complete(root, data, stage, capacity, group):
            self.deploy(capacity, group)
            self.command(f"{stage}_cl{capacity}_{group}", "benchmark_niah.py", stage, *self.args(capacity, group))
            if not original.stage_complete(root, data, stage, capacity, group):
                raise ValueError(f"Successful command omitted verified {stage} completion")
        else:
            print(f"{original.now()} VERIFIED {stage} CL{capacity} {group}; no repeat", flush=True)
        if stage != "pilot":
            self.release(capacity, group)

    def report(self) -> None:
        import niah_report as reporting

        reject_excluded_results(self.root, PRESERVED_CAPACITIES)
        reject_excluded_results(self.supplement_root, (2048,))
        output = self.root / "report_8k"
        marker = self.root / "continuation_report_8k.json"
        if output.exists() or marker.exists():
            raise FileExistsError("Existing 8K report preserved; no overwrite/retry")
        records, inputs = [], {}
        for root in (self.root, self.supplement_root):
            for path in sorted((root / "results").glob("*.json")):
                sha = original.digest(path)
                inputs[str(path)] = sha
                value = original.read(path)
                values = value if isinstance(value, list) else value.get("records", [value])
                records.extend({**record, "source_root": str(root),
                                "source_result_file": str(path), "source_result_sha256": sha}
                               for record in values)
        self.current = "report_8k"
        self.status("running")
        manifest = original.read(self.scope / "selected_manifest.json")
        data = reporting.aggregate_records(records, manifest, groups=GROUPS, capacities=CAPACITIES)
        reporting.write_report(data, output)
        if data.get("status") != "complete":
            raise ValueError("8K report has incomplete coverage")
        original.save(marker, {
            "utc": original.now(), "inputs_sha256": inputs,
            "manifest_sha256": original.digest(self.scope / "selected_manifest.json"),
            "amendment_sha256": original.digest(self.scope / "amendment.json"),
            "roots": [str(self.root), str(self.supplement_root)],
            "files_sha256": {p.name: original.digest(p) for p in output.iterdir() if p.is_file()},
        })

    def run(self) -> None:
        try:
            self.status("running")
            self.data = original.verify_frozen(self.root)
            self.contract()
            self.supplement_data = original.read(self.supplement_root / "data/data_manifest.json")
            self.preserved_probes()
            self.command("build_cl2048", "niah_2k_supplement.py", "build", *self.args(2048),
                         "--group", *GROUPS, "--workers", "1")
            self.command("audit_cl2048", "niah_2k_supplement.py", "audit", *self.args(2048), "--group", *GROUPS)
            for group in GROUPS:
                original.verify_audit(self.supplement_root, 2048, group)
            self.command("assets_cl2048", "benchmark_niah.py", "assets", *self.args(2048))
            self.block("pilot", 2048, "fp16")
            for group in GROUPS:
                self.block("probe", 2048, group)
            self.all_probes()
            for capacity in CAPACITIES:
                for group in GROUPS:
                    self.block("measure", capacity, group)
            self.report()
            self.current = "complete"
            self.status("complete", main_runs=MAIN_RUNS, capacities=list(CAPACITIES))
        except BaseException as error:
            self.status("failed", error=repr(error), policy="Stopped without retry; 16K remains excluded")
            raise


def run(root: Path, supplement_root: Path, python: Path, scope: Path) -> None:
    with original.exclusive(root):
        with recovery.build_lock(root):
            ensure_idle(root, supplement_root)
            verify_scope(root, supplement_root, python, scope)
            original.save(scope / "run_consumed.json", {"utc": original.now(), "pid": os.getpid()})
        controller = Controller(root, supplement_root, python, scope)
        original.save(scope / "controller.json", {"run_directory": str(controller.run_dir), "pid": os.getpid()})
        controller.run()


def launch(root: Path, supplement_root: Path, python: Path, authorization: str) -> dict:
    with recovery.locks(root):
        ensure_idle(root, supplement_root)
        scope = create_scope(root, supplement_root, python, authorization)
        original.save(scope / "launch_claim.json", {"utc": original.now(), "pid": os.getpid()})
    # Do not let a child race the launcher's still-held controller/build locks.
    argv = [str(python), str(Path(__file__).resolve()), "--root", str(root),
            "--supplement-root", str(supplement_root), "--python", str(python), "--scope", str(scope)]
    with (scope / "controller.log").open("x") as log:
        child = subprocess.Popen(
            argv, cwd=original.REPO,
            env={**os.environ, "PYTHONPATH": str(original.REPO / "src"), "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {"utc": original.now(), "pid": child.pid, "argv": argv,
              "log": str(scope / "controller.log"), "scope_directory": str(scope),
              "status": "launched_not_verified_complete"}
    original.save(scope / "launch.json", record)
    print(json.dumps(record, indent=2), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--authorization")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    supplement_root = args.supplement_root.expanduser().resolve(strict=True)
    python = args.python.expanduser().absolute()
    if args.detach:
        if args.scope or not args.authorization:
            parser.error("Initial launch requires --authorization, without --scope")
        launch(root, supplement_root, python, args.authorization)
    else:
        if args.authorization or not args.scope:
            parser.error("A child requires its existing --scope, without --authorization")
        run(root, supplement_root, python, args.scope.resolve(strict=True))


if __name__ == "__main__":
    main()
