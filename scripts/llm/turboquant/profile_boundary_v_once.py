# SPDX-License-Identifier: BSD-3-Clause
"""Profile boundary-V HTP stages separately from the single unprofiled benchmark.

Reuses unchanged legacy optrace contexts, not their old measurements. Builds
new boundary-V contexts from frozen DLCs. No CPU fallback or second throughput run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark_asymmetric_once import ARTIFACTS, SCRIPTS, digest, read, save_new
from benchmark_boundary_v_once import GROUPS, runtime_override, validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("freeze", "build", "measure", "render", "summarize")
    )
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    args = parser.parse_args()
    root, experiment = args.root.resolve(), args.experiment.resolve()

    def run(script: str, *params: str, python: str = sys.executable) -> None:
        subprocess.run([python, str(SCRIPTS / script), *params], check=True)

    if args.action == "freeze":
        protocol = validate(experiment)
        root.mkdir(exist_ok=True)
        for group in args.groups:
            groups = {}
            bundle = experiment / group
            report = experiment / "reports" / f"{group}_perf_long.json"
            groups[group] = {
                "bundle": str(bundle),
                "conversion_sha256": digest(bundle / "convert_report.json"),
                "reference_report": str(report),
                "reference_report_sha256": digest(report),
            }
            runtime = runtime_override(group, protocol)
            if runtime:
                package = Path(runtime)
                groups[group].update(
                    native_runtime_package=str(package),
                    native_runtime_manifest_sha256=digest(package / "manifest.json"),
                )
            if group == "k4_v4":
                (root / group).symlink_to(
                    ARTIFACTS / "asymmetric_optrace_20261010" / group,
                    target_is_directory=True,
                )
            save_new(
                root / "reference" / group / "experiment.json",
                {
                    "groups": groups,
                    "assets": str(ARTIFACTS / "device_assets_cl1024"),
                    "assets_sha256": read(
                        ARTIFACTS / "device_assets_cl1024/assets.json"
                    )["sha256"],
                    "protocol": {
                        "sessions_per_group": 1,
                        "diagnostic_only": True,
                        "prefill": "8 AR128 chunks: first 1 valid token, then 7 full chunks",
                        "decode_steps": [0, 63, 126],
                        "limits": "physical busy cycles are work, not stage wall latency or TTFT savings",
                    },
                },
            )
        return
    if args.action == "build":
        if not (root / "runner/qnn-llm-runner").exists():
            subprocess.run(
                [
                    "bash",
                    str(SCRIPTS / "qnn_runner/build_android.sh"),
                    str(root / "runner"),
                ],
                check=True,
            )

        def build(group: str) -> None:
            if group == "k4_v4":
                return
            frozen = read(root / "reference" / group / "experiment.json")
            run(
                "build_optrace_contexts.py",
                "--source",
                frozen["groups"][group]["bundle"],
                "--out",
                str(root / group),
                "--parts",
                "1",
                "2",
                "3",
                "4",
            )

        with ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(build, args.groups))
    elif args.action == "measure":
        for group in args.groups:
            frozen = read(root / "reference" / group / "experiment.json")
            run(
                "profile_optrace_once.py",
                "measure",
                "--root",
                str(root),
                "--device-prefix",
                experiment.name + "_optrace",
                "--reference-reports",
                str(root / "reference" / group),
                "--assets",
                frozen["assets"],
                "--groups",
                group,
            )
    elif args.action == "render":
        for group in args.groups:
            run(
                "render_optrace.py",
                "--logs",
                str(root / "traces" / group),
                "--contexts",
                str(root / group),
                "--out",
                str(root / "rendered" / group),
                "--jobs",
                "4",
                python="/home/hjw/qnn-venv/bin/python",
            )
    else:
        run("summarize_optrace.py", "--root", str(root), "--groups", *args.groups)


if __name__ == "__main__":
    main()
