# SPDX-License-Identifier: BSD-3-Clause
"""Aggregate already-completed K8 codec diagnostics, without device execution."""

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    encoder, decoder, validation, sources = [], [], {}, {}
    for group, sub in (
        ("lloyd_max", "encoder_lm"),
        ("structured", "encoder_structured_checked"),
    ):
        work = root / sub
        manifest = read(work / "manifest.json")
        numeric = read(work / "p2_report.json")
        validation[group] = numeric
        sources[str(work / "manifest.json")] = digest(work / "manifest.json")
        sources[str(work / "p2_report.json")] = digest(work / "p2_report.json")
        for tokens in (1, 128):
            name = f"tq_encode_key_t{tokens}"
            path = work / "device" / (name + "_detailed.profile.txt")
            sources[str(path)] = digest(path)
            blocks = re.split(
                r"^Execute Stat \d+\s*$", path.read_text(), flags=re.MULTILINE
            )[1:]
            cases = manifest["graphs"][name]["cases"]
            if len(blocks) != len(cases):
                raise ValueError("Missing encoder executes")
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
                        else semantic_stage("tq_key_0_" + op)[0]
                    )
                    stages[stage] += int(cycles)
                encoder.append(
                    {
                        "group": group,
                        "tokens": tokens,
                        "case": case,
                        "named_op_cycles": dict(stages),
                        "timings": parse_profile(block),
                    }
                )
        work = root / ("decoder_lm" if group == "lloyd_max" else "decoder_structured")
        correctness = read(work / "correctness.json")
        sources[str(work / "correctness.json")] = digest(work / "correctness.json")
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
                r"^\s+decode8:OpId_\d+ \(cycles\) : (\d+)\s+cycles",
                text,
                flags=re.MULTILINE,
            )
            if len(values) != 1:
                raise ValueError("Expected one native execution")
            decoder.append(
                {
                    "group": group,
                    "tokens": tokens,
                    "native_op_cycles": int(values[0]),
                    "correctness": correctness[f"native_t{tokens}"],
                    "timings": parse_profile(text),
                }
            )
    save_new(
        root / "isolated_probes.json",
        {
            "encoder": encoder,
            "decoder": decoder,
            "validation": validation,
            "source_sha256": sources,
            "limits": "Diagnostic named-op counters, not physical busy intervals or TTFT percentages. Same input shapes/cases; no repeat averaging. Native uses qnn-net-run default power; encoder burst; not full-model timings.",
            "preflight_note": "encoder_structured/ preserves an initial CPU-only preflight rejected by the LM-only comparator. Structured quantization uses arithmetic decision edges, not rounded-LUT nearest boundaries. Corrected comparator keeps identical tolerances; encoder_structured_checked/ is the only executed structured device probe.",
        },
    )


if __name__ == "__main__":
    main()
