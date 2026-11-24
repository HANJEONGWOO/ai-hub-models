# SPDX-License-Identifier: BSD-3-Clause
"""One explicitly approved NIAH continuation after a pre-runner ADB failure.

Only the known failed result and attempt are archived, never deleted. Completed
inferences, frozen scientific contracts and all build outputs remain unchanged.
The original runtime skips verified successes and still stops on any new error.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import niah_8k_continue as scoped

original = scoped.original
recovery = scoped.recovery
PRIOR_RUN = "20261103T085201_312970Z_11915"
PRIOR_PID = 11915
CASE_ID = "test_08_cl2048_d025"
LABEL = f"fp16_{CASE_ID}"
TARGETS = (
    f"{scoped.SUPPLEMENT_NAME}/results/{LABEL}.json",
    f"{scoped.SUPPLEMENT_NAME}/attempts/{LABEL}",
)
EXPECTED_SUCCESSES = {(1024, group): 50 for group in scoped.GROUPS} | {(2048, "fp16"): 41}


def ensure_idle(root: Path) -> None:
    scoped.ensure_idle(root, root / scoped.SUPPLEMENT_NAME)
    for process in scoped.resume.processes():
        script = Path(process.argv[1]).name if len(process.argv) > 1 else ""
        target = scoped.resume.option(process.argv, "--root")
        if script != Path(__file__).name or not target or Path(target).resolve() != root:
            continue
        if process.pid == os.getppid() and "--detach" in process.argv:
            continue
        raise RuntimeError(f"ADB recovery remains active: PID {process.pid}")


def prior_provenance(root: Path, python: Path) -> dict:
    scoped.verify_scope(root, root / scoped.SUPPLEMENT_NAME, python, root / scoped.SCOPE_NAME)
    run_dir = recovery.safe_path(root, f"continuation_runs/{PRIOR_RUN}")
    status = original.read(run_dir / "status.json")
    scope = root / scoped.SCOPE_NAME
    if (
        status.get("status") != "failed"
        or status.get("stage") != "measure_cl2048_fp16"
        or status.get("pid") != PRIOR_PID
        or status.get("utc") != "2026-11-03T13:46:44.422518+00:00"
        or original.read(scope / "controller.json") != {"run_directory": str(run_dir), "pid": PRIOR_PID}
        or original.read(scope / "run_consumed.json").get("pid") != PRIOR_PID
        or original.read(scope / "launch.json").get("pid") != PRIOR_PID
    ):
        raise ValueError("Unexpected previous 8K-scope failure")
    files = [p for directory in (scope, run_dir) for p in sorted(directory.iterdir()) if p.is_file()]
    return {str(p.relative_to(root)): original.digest(p) for p in files}


def validate_pre_runner_failure(result: dict, attempt: Path, supplement_root: Path) -> None:
    """Reject inference failures: the known failed command was bundle ls only."""
    commands = [json.loads(line) for line in (attempt / "commands.jsonl").read_text().splitlines() if line.strip()]
    expected = ["shell", f"ls /data/local/tmp/qaihm_turboquant/llm/bundles/{supplement_root.name}_cl2048_fp16"]
    start = original.read(attempt / "start.json")
    if (
        result.get("status") != "error" or result.get("case_id") != CASE_ID
        or result.get("group") != "fp16" or result.get("capacity") != 2048
        or result.get("split") != "test"
        or result.get("failure_class") != "infrastructure_or_contract_failure_not_model_wrong_answer"
        or "adb.exe: no devices/emulators found" not in result.get("error", "")
        or len(commands) != 1 or commands[0].get("command") != expected
        or "adb.exe: no devices/emulators found" not in commands[0].get("error", "")
        or set(p.name for p in attempt.iterdir()) != {"start.json", "commands.jsonl", "host_stdout.log"}
        or (attempt / "host_stdout.log").stat().st_size != 0
        or start.get("stage") != "measure" or start.get("case_id") != CASE_ID
        or start.get("group") != "fp16"
        or start.get("deployment_sha256") != result.get("deployment_sha256")
        or start.get("settings") != {"n_gen": 64, "sessions": 1, "stop_on_eos": True,
                                      "tokens": f"{CASE_ID}.bin",
                                      "bundle": f"{supplement_root.name}_cl2048_fp16"}
    ):
        raise ValueError("Failure is not the exact pre-runner ADB disconnect")


def successful_inventory(root: Path) -> dict:
    """Validate every prior result and hash successes, raw attempts and gates."""
    import benchmark_niah as runtime

    counts: collections.Counter = collections.Counter()
    preserved: dict = {}
    failures = []
    for source in (root, root / scoped.SUPPLEMENT_NAME):
        data = original.read(source / "data/data_manifest.json")
        cases = {case["case_id"]: case for case in data["cases"]}
        expected_attempts = set()
        for path in sorted((source / "results").glob("*.json")):
            result = original.read(path)
            case, group = cases[result["case_id"]], result["group"]
            if group not in scoped.GROUPS or case["capacity"] not in scoped.CAPACITIES:
                raise ValueError("Prior result outside the approved active scope")
            if runtime.result_path(source, group, case) != path:
                raise ValueError("Unexpected result filename")
            label = f"{group}_{case['case_id']}"
            attempt = recovery.safe_path(source, f"attempts/{label}")
            expected_attempts.add(label)
            deployment_file = runtime.deployment_path(source, case["capacity"], group)
            if result.get("status") != "ok":
                failures.append(str(path.relative_to(root)))
                if str(path.relative_to(root)) != TARGETS[0]:
                    raise ValueError("An additional failed result requires separate approval")
                if (result.get("tokens_sha256") != case["tokens_sha256"]
                        or result.get("deployment_sha256") != original.digest(deployment_file)):
                    raise ValueError("Failed attempt input/deployment differs from frozen data")
                validate_pre_runner_failure(result, attempt, source)
                continue
            checked = runtime.check_existing(source, group, case, original.digest(deployment_file))
            runtime.validate_report(original.read(source / checked["runner_report"]), case,
                                    original.read(deployment_file), runtime.name(source, case["capacity"], group))
            if case["split"] == "test":
                counts[(case["capacity"], group)] += 1
            for keep in (path, attempt):
                preserved[str(keep.relative_to(root))] = recovery.inventory(keep, {})
        if {p.name for p in (source / "attempts").iterdir()} != expected_attempts:
            raise ValueError("An unaccounted attempt requires separate review")
        for name in ("stage_completion", "audits", "deployments"):
            path = recovery.safe_path(source, name)
            preserved[str(path.relative_to(root))] = recovery.inventory(path, {})
    if dict(counts) != EXPECTED_SUCCESSES or failures != [TARGETS[0]]:
        raise ValueError("Expected exactly 241 completed main cases and one known failed attempt")
    return preserved


def verify_gates(root: Path) -> None:
    """Read-only validation; do not rerun pilots, probes, audits or builds."""
    for source, capacities in ((root, scoped.PRESERVED_CAPACITIES), (root / scoped.SUPPLEMENT_NAME, (2048,))):
        data = original.read(source / "data/data_manifest.json")
        pilot_capacity = 2048 if source != root else 1024
        if not original.stage_complete(source, data, "pilot", pilot_capacity, "fp16"):
            raise ValueError("Preserved pilot gate is missing")
        for capacity in capacities:
            for group in scoped.GROUPS:
                original.verify_audit(source, capacity, group)
                if not original.stage_complete(source, data, "probe", capacity, group):
                    raise ValueError("All 16 preserved probe gates are required")


def prepare(root: Path, python: Path, authorization: str) -> Path:
    if not authorization.strip():
        raise ValueError("Explicit user authorization is required")
    with recovery.locks(root):
        if list(root.glob("recovery_adb_*")):
            raise FileExistsError("ADB recovery already exists; no prepare retry")
        ensure_idle(root)
        provenance = prior_provenance(root, python)
        original.verify_frozen(root)
        verify_gates(root)
        preserved = successful_inventory(root)
        targets = {name: recovery.inventory(recovery.safe_path(root, name), {}) for name in TARGETS}
        ensure_idle(root)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        archive = recovery.safe_path(root, f"recovery_adb_{stamp}_{os.getpid()}", exists=False)
        archive.mkdir(exist_ok=False)
        original.save(archive / "intent.json", {
            "utc": original.now(), "authorization": authorization,
            "helper_sha256": original.digest(Path(__file__)), "python": str(python.resolve(strict=True)),
            "prior_provenance": provenance, "preserved": preserved, "targets": targets,
            "capacities": list(scoped.CAPACITIES), "groups": list(scoped.GROUPS),
            "successful_main_preserved": 241, "remaining_main": 559,
            "retry_case": LABEL, "failure_before_runner_launch": True,
            "policy": "One approved pre-runner infrastructure retry; no successful inference repeats, rebuilds, 16K work or scientific changes",
        })
        try:
            for index, name in enumerate(TARGETS):
                recovery.archive_one(root, archive, name, {}, targets[name])
                original.save(archive / f"move_{index}.json", {"source": name, "verified": True})
            original.save(archive / "prepared.json", {
                "utc": original.now(), "intent_sha256": original.digest(archive / "intent.json"),
                "targets": list(TARGETS),
            })
        except BaseException as error:
            original.save(archive / "prepare_failed.json", {"utc": original.now(), "error": repr(error),
                          "policy": "Partial archive preserved; no automatic rollback or retry"})
            raise
    print(json.dumps({"prepared_archive": str(archive), "remaining_main": 559}), flush=True)
    return archive


def verify_prepared(root: Path, archive: Path, python: Path) -> dict:
    if archive.parent != root or not archive.name.startswith("recovery_adb_"):
        raise ValueError("Unexpected ADB archive")
    recovery.safe_path(root, archive.name)
    prepared, intent = original.read(archive / "prepared.json"), original.read(archive / "intent.json")
    if (
        prepared["intent_sha256"] != original.digest(archive / "intent.json")
        or prepared["targets"] != list(TARGETS) or set(intent["targets"]) != set(TARGETS)
        or intent["helper_sha256"] != original.digest(Path(__file__))
        or intent["python"] != str(python.resolve(strict=True))
        or intent["capacities"] != list(scoped.CAPACITIES) or intent["groups"] != list(scoped.GROUPS)
        or intent["prior_provenance"] != prior_provenance(root, python)
        or not intent.get("authorization", "").strip()
        or intent.get("successful_main_preserved") != 241 or intent.get("remaining_main") != 559
        or intent.get("retry_case") != LABEL or intent.get("failure_before_runner_launch") is not True
    ):
        raise ValueError("ADB recovery provenance mismatch")
    for name, expected in intent["targets"].items():
        if recovery.inventory(recovery.safe_path(archive, f"preserved/{name}"), {}) != expected:
            raise ValueError(f"Archived failure changed: {name}")
    for name, expected in intent["preserved"].items():
        if recovery.inventory(recovery.safe_path(root, name), {}) != expected:
            raise ValueError(f"Preserved completed result/gate changed: {name}")
    return intent


class RecoveryController(scoped.Controller):
    def __init__(self, root: Path, python: Path, archive: Path) -> None:
        super().__init__(root, root / scoped.SUPPLEMENT_NAME, python, root / scoped.SCOPE_NAME)
        self.archive = archive

    def command(self, label: str, script: str, *options: str) -> None:
        allowed = {"benchmark_niah.py": {"measure", "deploy"},
                   "niah_device_copies.py": {"restore", "release"},
                   "niah_2k_device_copies.py": {"restore", "release"}}
        if not options or options[0] not in allowed.get(script, set()):
            raise ValueError("Recovery forbids rebuilds and pilot/probe/audit/assets reruns")
        super().command(label, script, *options)

    def report(self) -> None:
        super().report()
        intent = original.read(self.archive / "intent.json")
        failed = self.archive / "preserved" / TARGETS[0]
        replacement = self.root / TARGETS[0]
        annex = self.root / "report_8k/infrastructure_recovery.json"
        original.save(annex, {
            "authorization": intent["authorization"], "archive": str(self.archive),
            "original_failed_result": str(failed), "original_failed_result_sha256": original.digest(failed),
            "original_attempt": str(self.archive / "preserved" / TARGETS[1]),
            "approved_retry_result": str(replacement), "approved_retry_result_sha256": original.digest(replacement),
            "infrastructure_failures": 1, "approved_retry_cases": 1,
            "failure_before_model_inference": True, "successful_main_preserved": 241,
            "remaining_main_after_failure": 559,
            "policy": "No successful evaluations repeated or selected by quality; original pre-runner ADB failure is preserved separately from 800 unique successful cases",
        })
        original.save(self.root / "continuation_report_8k_adb_recovery.json", {
            "utc": original.now(), "parent_marker_sha256": original.digest(self.root / "continuation_report_8k.json"),
            "annex": str(annex), "annex_sha256": original.digest(annex),
            "archive_intent_sha256": original.digest(self.archive / "intent.json"),
        })

    def run(self) -> None:
        try:
            self.status("running", recovery_archive=str(self.archive))
            self.data = original.verify_frozen(self.root)
            self.contract()
            self.supplement_data = original.read(self.supplement_root / "data/data_manifest.json")
            self.all_probes()
            original.save(self.run_dir / "recovery_amendment.json", {
                "archive": str(self.archive), "intent_sha256": original.digest(self.archive / "intent.json"),
                "prepared_sha256": original.digest(self.archive / "prepared.json"),
                "successful_main_preserved": 241, "remaining_main": 559,
                "prior_failure_before_runner_launch": True,
            })
            for capacity in scoped.CAPACITIES:
                for group in scoped.GROUPS:
                    self.block("measure", capacity, group)
            self.report()
            original.save(self.archive / "evaluation_completion.json", {
                "utc": original.now(), "main_runs": scoped.MAIN_RUNS,
                "report_marker": str(self.root / "continuation_report_8k.json"),
                "report_marker_sha256": original.digest(self.root / "continuation_report_8k.json"),
                "prior_failed_attempt": "preserved/" + TARGETS[1],
                "policy": "All 800 unique successful cases; one separately preserved pre-runner ADB failure",
            })
            self.current = "complete"
            self.status("complete", main_runs=scoped.MAIN_RUNS, capacities=list(scoped.CAPACITIES),
                        recovery_archive=str(self.archive))
        except BaseException as error:
            self.status("failed", error=repr(error), recovery_archive=str(self.archive),
                        policy="Stopped without another retry; completed cases and all failures preserved")
            raise


def run(root: Path, python: Path, archive: Path) -> None:
    with original.exclusive(root):
        with recovery.build_lock(root):
            ensure_idle(root)
            verify_prepared(root, archive, python)
            if any(recovery.safe_path(root, name, exists=False).exists() for name in TARGETS):
                raise FileExistsError("Retry output already exists; second retry forbidden")
            original.save(archive / "run_consumed.json", {"utc": original.now(), "pid": os.getpid()})
        controller = RecoveryController(root, python, archive)
        original.save(archive / "controller.json", {"run_directory": str(controller.run_dir), "pid": os.getpid()})
        controller.run()


def launch(root: Path, python: Path, archive: Path) -> dict:
    with recovery.locks(root):
        ensure_idle(root)
        verify_prepared(root, archive, python)
        if (archive / "run_consumed.json").exists():
            raise FileExistsError("ADB recovery already consumed")
        original.save(archive / "launch_claim.json", {"utc": original.now(), "pid": os.getpid()})
    argv = [str(python), str(Path(__file__).resolve()), "run", "--root", str(root),
            "--python", str(python), "--archive", str(archive)]
    with (archive / "controller.log").open("x") as log:
        child = subprocess.Popen(argv, cwd=original.REPO,
                                 env={**os.environ, "PYTHONPATH": str(original.REPO / "src"), "PYTHONUNBUFFERED": "1"},
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    record = {"utc": original.now(), "pid": child.pid, "argv": argv,
              "log": str(archive / "controller.log"), "status": "launched_not_verified_complete"}
    original.save(archive / "launch.json", record)
    print(json.dumps(record, indent=2), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--authorization")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    root, python = args.root.absolute(), args.python.absolute()
    if root != root.resolve(strict=True):
        raise ValueError("Recovery root must not use symlink parents")
    if args.stage == "prepare":
        if args.archive or args.detach or not args.authorization:
            parser.error("prepare requires --authorization without --archive/--detach")
        prepare(root, python, args.authorization)
    else:
        if args.archive is None or args.authorization:
            parser.error("run requires --archive, without --authorization")
        (launch if args.detach else run)(root, python, args.archive.absolute())


if __name__ == "__main__":
    main()
