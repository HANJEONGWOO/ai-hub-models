# SPDX-License-Identifier: BSD-3-Clause
"""Read existing isolated decoder probes; no additional device executions."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path

from benchmark_asymmetric_once import digest, read, save_new
from htp_codec_validation import (
    DEFAULT_QNN_PYTHON,
    DEFAULT_SDK,
    parse_profile,
    qairt_env,
    run_logged,
)
from summarize_optrace import physical_events, union_length


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "probe_profiles"
    if (root / "decoder_comparison.json").exists():
        raise FileExistsError("Completed decoder comparison already exists")
    out.mkdir(exist_ok=True)
    sdk = DEFAULT_SDK
    lib = sdk / "lib/x86_64-linux-clang"
    env = qairt_env(sdk, DEFAULT_QNN_PYTHON)
    viewer = sdk / "bin/x86_64-linux-clang/qnn-profile-viewer"
    config = out / "viewer_config.json"
    features = {
        "features": {
            "htp_json": False,
            "qhas_json": False,
            "qhas_schema": False,
            "traceback": False,
            "memory_info": False,
            "runtrace": True,
            "enable_input_output_flow_events": False,
            "enable_sequencer_flow_events": False,
        }
    }
    if config.exists():
        assert read(config) == features
    else:
        save_new(config, features)
    rows = []
    details = {}
    for tokens in (1, 3, 256, 1023):
        name = f"native_t{tokens}"
        groups = [root / "probe_lut", root / "probe_arithmetic"]
        for suffix in ("_packed.raw", "_scale.raw", "_expected.npy"):
            assert digest(groups[0] / (name + suffix)) == digest(
                groups[1] / (name + suffix)
            )
        results = [
            next((work / "device" / f"out_{name}" / "Result_0").glob("decoded*.raw"))
            for work in groups
        ]
        assert results[0].read_bytes() == results[1].read_bytes()
        for label, work in zip(("lut", "arithmetic"), groups, strict=True):
            numeric = read(work / "correctness.json")[name]
            assert numeric["passed"] and numeric["bit_exact_fraction"] == 1
            target = out / f"{label}_{tokens}"
            target.mkdir(exist_ok=True)
            log = work / "device" / f"out_{name}" / "qnn-profiling-data_0.log"
            schematic = work / f"{name}_schematic.bin"
            trace = target / "trace.json.gz"
            command = [
                str(viewer),
                "--reader",
                str(lib / "libQnnHtpOptraceProfilingReader.so"),
                "--input_log",
                str(log),
                "--schematic",
                str(schematic),
                "--output",
                str(trace),
                "--config",
                str(config),
            ]
            identity = {
                "log_sha256": digest(log),
                "schematic_sha256": digest(schematic),
                "binary_sha256": digest(work / f"{name}.bin"),
                "input_sha256": {
                    s: digest(work / (name + s)) for s in ("_packed.raw", "_scale.raw")
                },
                "output_sha256": digest(results[0 if label == "lut" else 1]),
                "render_argv": command,
            }
            if (target / "identity.json").exists():
                assert read(target / "identity.json") == identity
            else:
                save_new(target / "identity.json", identity)
            if not trace.exists():
                run_logged(command, target / "viewer.log", env, cwd=target)
            textpath = target / "detailed.txt"
            if not textpath.exists():
                run_logged(
                    [
                        str(viewer),
                        "--reader",
                        str(lib / "libQnnHtpProfilingReader.so"),
                        "--input_log",
                        str(log),
                    ],
                    textpath,
                    env,
                    cwd=target,
                )
            text = textpath.read_text()
            if len(re.findall(r"^Execute Stat \d+\s*$", text, re.MULTILINE)) != 1:
                raise ValueError("Expected one execute per decoder shape")
            events = physical_events(json.loads(gzip.decompress(trace.read_bytes())))
            hvx = [
                e
                for e in events
                if "uses_hvx" in e["args"].get("Flags", []) and e["resource"] == "HVX"
            ]
            hmx = [e for e in events if "uses_hmx" in e["args"].get("Flags", [])]
            assert hvx and not hmx
            times = parse_profile(text)
            row = {
                "decoder": label,
                "tokens": tokens,
                "kv_heads": 8,
                "coordinates": 128,
                "HVX_busy_cycles": sum(e["dur"] for e in hvx),
                "HVX_interval_union_cycles": union_length(
                    [(e["ts"], e["ts"] + e["dur"]) for e in hvx]
                ),
                "accelerator_execute_us": times["accelerator_execute_us"][0],
                "accelerator_execute_excluding_wait_us": int(
                    re.search(
                        r"^Accelerator \(execute excluding wait\) time : (\d+)\s+us",
                        text,
                        re.MULTILINE,
                    )[1]
                ),
                "bit_exact_fraction": numeric["bit_exact_fraction"],
            }
            rows.append(row)
            details[f"{label}_{tokens}"] = {
                "timings": parse_profile(text),
                "physical_hvx_events": [
                    {
                        "kernel": e["name"],
                        "qnn_op": e["args"].get("QNN Op Name"),
                        "cycles": e["dur"],
                    }
                    for e in hvx
                ],
            }
            print(row, flush=True)
    save_new(
        root / "decoder_comparison.json",
        {
            "rows": rows,
            "details": details,
            "same_inputs_and_outputs": True,
            "all_eight_probes_bit_exact": True,
            "units": "Physical HVX worker busy cycles, not wall time. SDK accelerator execute microseconds include scheduling overhead.",
            "limits": "One optrace diagnostic per decoder/shape; qnn-net-run default power policy, not full-model burst; no variance/significance claim",
        },
    )
    with (root / "decoder_comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
