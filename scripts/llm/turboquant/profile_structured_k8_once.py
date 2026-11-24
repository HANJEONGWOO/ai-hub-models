# SPDX-License-Identifier: BSD-3-Clause
"""One separate HTP work trace per frozen K8 group, not a performance repeat."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from benchmark_asymmetric_once import SCRIPTS, digest, read, save_new
from benchmark_structured_k8_once import GROUPS, PREVIOUS, validate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action", choices=("freeze", "build", "measure", "render", "summarize")
    )
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    args = p.parse_args()
    root, experiment = args.root.resolve(), args.experiment.resolve()
    # Shared optrace tool namespaces its device directory by root basename.
    if root.name == "optrace":
        raise ValueError(
            "Use a unique root basename, e.g. structured_k8_20261013_optrace"
        )

    def run(script: str, *options: str, python: str = sys.executable) -> None:
        subprocess.run([python, str(SCRIPTS / script), *options], check=True)

    if args.action == "freeze":
        protocol = validate(experiment)
        root.mkdir(exist_ok=True)
        for group in args.groups:
            bundle = experiment / group
            report = experiment / "reports" / f"{group}_perf_long.json"
            package = Path(protocol["mixed_native_override"])
            record = {
                "bundle": str(bundle),
                "conversion_sha256": digest(bundle / "convert_report.json"),
                "reference_report": str(report),
                "reference_report_sha256": digest(report),
                "native_runtime_package": str(package),
                "native_runtime_manifest_sha256": digest(package / "manifest.json"),
            }
            if group == GROUPS[0]:
                (root / group).symlink_to(
                    PREVIOUS / "optrace" / group, target_is_directory=True
                )
            save_new(
                root / "reference" / group / "experiment.json",
                {
                    "groups": {group: record},
                    "assets": protocol["assets"],
                    "assets_sha256": protocol["assets_manifest"]["sha256"],
                    "protocol": {
                        "sessions_per_group": 1,
                        "diagnostic_only": True,
                        "decode_steps": [0, 63, 126],
                        "prefill": "897 tokens in eight AR128 calls; first has one valid token",
                        "limits": "Physical busy cycles are not wall time or TTFT shares",
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
        for group in args.groups:
            if group == GROUPS[1]:
                run(
                    "build_optrace_contexts.py",
                    "--source",
                    str(experiment / group),
                    "--out",
                    str(root / group),
                    "--parts",
                    "1",
                    "2",
                    "3",
                    "4",
                )
    elif args.action == "measure":
        for group in args.groups:
            frozen = read(root / "reference" / group / "experiment.json")
            run(
                "profile_optrace_once.py",
                "measure",
                "--root",
                str(root),
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
