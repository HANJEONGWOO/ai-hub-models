# SPDX-License-Identifier: BSD-3-Clause
"""Read existing isolated HTP diagnostics; never launch another device run."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

from benchmark_asymmetric_once import digest, read, save_new
from htp_codec_validation import (
    DEFAULT_QNN_PYTHON,
    DEFAULT_SDK,
    parse_profile,
    qairt_env,
)
from summarize_stage_profiles import semantic_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoder, decoder, validation, sources = [], [], {}, {}
    for bits, directory in (
        (2, "encoder_v2"),
        (4, "encoder_v4"),
        (8, "encoder_v8"),
    ):
        work = args.root / directory
        manifest = read(work / "manifest.json")
        numeric = read(work / "p2_report.json")
        validation[str(bits)] = {
            "passed": numeric["passed"],
            "cases": {
                name: entry["cases"] for name, entry in numeric["graphs"].items()
            },
            "all_ops_on_htp": all(
                not g["dlc_ops_missing_from_accelerator_profile"]
                for g in numeric["graphs"].values()
            ),
        }
        for tokens in (1, 128):
            name = f"tq_encode_value_t{tokens}"
            path = work / "device" / (name + "_detailed.profile.txt")
            sources[str(path)] = digest(path)
            blocks = re.split(
                r"^Execute Stat \d+\s*$", path.read_text(), flags=re.MULTILINE
            )[1:]
            cases = manifest["graphs"][name]["cases"]
            if len(blocks) != len(cases):
                raise ValueError("Missing encoder execute records")
            for case, block in zip(cases, blocks, strict=True):
                stages = Counter()
                for op, cycles in re.findall(
                    r"^\s+(\S+):OpId_\d+ \(cycles\) : (\d+)\s+cycles",
                    block,
                    flags=re.MULTILINE,
                ):
                    stage = (
                        "encode.scale_correction"
                        if op == "norm"
                        else "encode.pack"
                        if op == "packed"
                        else semantic_stage("tq_value_0_" + op)[0]
                    )
                    stages[stage] += int(cycles)
                profile = parse_profile(block)
                encoder.append(
                    {
                        "bits": bits,
                        "tokens": tokens,
                        "case": case,
                        "named_op_cycles": dict(stages),
                        "accelerator_execute_us": profile["accelerator_execute_us"],
                        "accelerator_cycles": profile["accelerator_cycles"],
                    }
                )
    for bits in (2, 4, 8):
        work = args.root / f"decoder_b{bits}"
        correctness = read(work / "correctness.json")
        for tokens in (1, 3, 256):
            raw = work / "device" / f"out_native_t{tokens}" / "qnn-profiling-data_0.log"
            sources[str(raw)] = digest(raw)
            text = subprocess.check_output(
                [
                    str(DEFAULT_SDK / "bin/x86_64-linux-clang/qnn-profile-viewer"),
                    "--reader",
                    str(
                        DEFAULT_SDK
                        / "lib/x86_64-linux-clang/libQnnHtpProfilingReader.so"
                    ),
                    "--input_log",
                    str(raw),
                ],
                env=qairt_env(DEFAULT_SDK, DEFAULT_QNN_PYTHON),
                text=True,
            )
            values = re.findall(
                rf"^\s+decode{bits}:OpId_\d+ \(cycles\) : (\d+)\s+cycles",
                text,
                flags=re.MULTILINE,
            )
            if len(values) != 1:
                raise ValueError("Expected one Native diagnostic execution")
            decoder.append(
                {
                    "bits": bits,
                    "tokens": tokens,
                    "native_op_cycles": int(values[0]),
                    "correctness": correctness[f"native_t{tokens}"],
                    "timings": parse_profile(text),
                }
            )
    save_new(
        args.output or args.root / "isolated_probes.json",
        {
            "scope": "8 heads x tokens x 128 coordinates; standalone codec/native correctness diagnostics, not an LLM speed measurement",
            "limits": "Encoder named-node cycles are not physical optrace intervals and may assign fused work to other nodes; report each distinct input separately, no repeated-run average; decoder probes use qnn-net-run default power policy while full-model benchmarks use burst; do not infer TTFT savings",
            "source_sha256": sources,
            "encoder": encoder,
            "decoder": decoder,
            "encoder_validation": validation,
        },
    )


if __name__ == "__main__":
    main()
