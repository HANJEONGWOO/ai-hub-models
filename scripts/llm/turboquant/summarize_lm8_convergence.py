# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Report measured Gaussian/KV/NLL/latency results without merging their meanings."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

from benchmark_asymmetric_once import digest, read, save_new
from benchmark_lm8_convergence import GROUPS
from htp_codec_validation import (
    DEFAULT_QNN_PYTHON,
    DEFAULT_SDK,
    parse_profile,
    qairt_env,
    run_logged,
)
from summarize_native_results import performance
from summarize_optrace import physical_events, union_length
from summarize_stage_profiles import semantic_stage


def probes(root: Path) -> None:
    rows, numeric, hashes = [], {}, {}
    for group in GROUPS:
        work = root / ("encoder_" + group)
        manifest = read(work / "manifest.json")
        numeric[group] = read(work / "p2_report.json")
        hashes[str(work / "p2_report.json")] = digest(work / "p2_report.json")
        for tokens in (1, 128):
            name = f"tq_encode_key_t{tokens}"
            path = work / "device" / (name + "_detailed.profile.txt")
            hashes[str(path)] = digest(path)
            blocks = re.split(
                r"^Execute Stat \d+\s*$", path.read_text(), flags=re.MULTILINE
            )[1:]
            cases = manifest["graphs"][name]["cases"]
            if len(blocks) != len(cases):
                raise ValueError("Missing encoder executions")
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
                rows.append(
                    {
                        "group": group,
                        "tokens": tokens,
                        "case": case,
                        "named_op_cycles": dict(stages),
                        "timings": parse_profile(block),
                    }
                )
    out = root / "probe_profiles"
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
    decoders = []
    for group in (*GROUPS, "structured_lut_control"):
        work = root / ("decoder_" + group)
        correctness = read(work / "correctness.json")
        hashes[str(work / "correctness.json")] = digest(work / "correctness.json")
        for tokens in (1, 3, 256, 1023):
            name = f"native_t{tokens}"
            target = out / f"{group}_{tokens}"
            target.mkdir(exist_ok=True)
            log = work / "device" / f"out_{name}" / "qnn-profiling-data_0.log"
            schematic = work / f"{name}_schematic.bin"
            trace = target / "trace.json.gz"
            if not trace.exists():
                run_logged(
                    [
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
                    ],
                    target / "viewer.log",
                    env,
                    cwd=target,
                )
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
                raise ValueError("Expected one decoder execution per shape")
            events = physical_events(json.loads(gzip.decompress(trace.read_bytes())))
            hvx = [
                e
                for e in events
                if "uses_hvx" in e["args"].get("Flags", []) and e["resource"] == "HVX"
            ]
            assert hvx and not any(
                "uses_hmx" in e["args"].get("Flags", []) for e in events
            )
            reference = root / "decoder_lm8_100"
            for suffix in ("_packed.raw", "_scale.raw"):
                assert digest(work / (name + suffix)) == digest(
                    reference / (name + suffix)
                )
            if group == "structured_lut_control":
                a = next(
                    (work / "device" / f"out_{name}" / "Result_0").glob("decoded*.raw")
                )
                b = next(
                    (
                        root / "decoder_structured/device" / f"out_{name}" / "Result_0"
                    ).glob("decoded*.raw")
                )
                assert a.read_bytes() == b.read_bytes()
            c = correctness[name]
            assert c["passed"] and c["bit_exact_fraction"] == 1
            decoders.append(
                {
                    "group": group,
                    "tokens": tokens,
                    "bit_exact_fraction": c["bit_exact_fraction"],
                    "HVX_busy_cycles": sum(e["dur"] for e in hvx),
                    "HVX_interval_union_cycles": union_length(
                        [(e["ts"], e["ts"] + e["dur"]) for e in hvx]
                    ),
                    "accelerator_execute_us": parse_profile(text)[
                        "accelerator_execute_us"
                    ][0],
                    "accelerator_execute_excluding_wait_us": int(
                        re.search(
                            r"^Accelerator \(execute excluding wait\) time : (\d+)\s+us",
                            text,
                            re.MULTILINE,
                        )[1]
                    ),
                    "profile_sha256": digest(log),
                    "binary_sha256": digest(work / f"{name}.bin"),
                }
            )
    save_new(
        root / "codec_costs.json",
        {
            "encoder": rows,
            "decoder": decoders,
            "encoder_numerics": numeric,
            "source_sha256": hashes,
            "structured_lut_arithmetic_same_output": True,
            "limits": "Named-op encoder cycle counters and physical decoder HVX busy cycles are distinct measures, neither is a TTFT percentage. Isolated SDK execute times use qnn-net-run policy, not full-model burst policy. Diagnostic cases/shapes are not performance repeats.",
        },
    )


def results(root: Path) -> None:
    protocol = read(root / "protocol.json")
    rows, document_rows, groups = [], [], {}
    for group in GROUPS:
        perf = performance(root / "reports" / f"{group}_perf_long.json", 897)
        quality = {}
        for split in ("legacy", "heldout"):
            docs = []
            for i in range(4):
                path = root / "reports" / f"{group}_score_{split}_{i}.json"
                report = read(path)
                assert report["scored_tokens"] == 1023
                assert (
                    report["config_hash"] == protocol["profiles"][group]["config_hash"]
                )
                entry = {
                    "group": group,
                    "split": split,
                    "window": i,
                    "scored_tokens": report["scored_tokens"],
                    "nll_sum": report["nll_sum"],
                    "mean_nll": report["nll_sum"] / report["scored_tokens"],
                    "ppl": math.exp(report["nll_sum"] / report["scored_tokens"]),
                    "report_sha256": digest(path),
                }
                document_rows.append(entry)
                docs.append(entry)
            count = sum(d["scored_tokens"] for d in docs)
            nll = sum(d["nll_sum"] for d in docs) / count
            quality[split] = {
                "mean_nll": nll,
                "ppl": math.exp(nll),
                "scored_tokens": count,
                "documents": docs,
            }
        row = {
            "group": group,
            **{
                f"{split}_{metric}": quality[split][metric]
                for split in ("legacy", "heldout")
                for metric in ("mean_nll", "ppl")
            },
            **{
                k: perf[k]
                for k in (
                    "ttft_ms",
                    "prefill_tok_per_s",
                    "decode_tok_per_s",
                    "host_kv_MiB",
                    "io_buffer_MiB",
                    "end_VmRSS_MiB",
                    "process_VmHWM_MiB",
                )
            },
        }
        rows.append(row)
        binaries = {
            str(p): {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted((root / group).glob("part*_of_4.bin"))
        }
        groups[group] = {
            "quality": quality,
            "performance": perf,
            "binaries": binaries,
            "config_hash": protocol["profiles"][group]["config_hash"],
        }
    contrasts = {}
    for left, right in (("lm8_100", "lm8_converged"), ("lm8_converged", "structured")):
        key = right + "_minus_" + left
        contrasts[key] = {}
        for split in ("legacy", "heldout"):
            delta = [
                b["mean_nll"] - a["mean_nll"]
                for a, b in zip(
                    groups[left]["quality"][split]["documents"],
                    groups[right]["quality"][split]["documents"],
                    strict=True,
                )
            ]
            contrasts[key][split] = {
                "per_document_mean_nll_delta": delta,
                "mean_nll_delta": sum(delta) / 4,
                "improved_documents": sum(d < 0 for d in delta),
                "worsened_documents": sum(d > 0 for d in delta),
                "ppl_relative_change": groups[right]["quality"][split]["ppl"]
                / groups[left]["quality"][split]["ppl"]
                - 1,
            }
    save_new(
        root / "comparison.json",
        {
            "rows": rows,
            "groups": groups,
            "contrasts": contrasts,
            "convergence": read(root / "codebook/optimization.json"),
            "offline_quality": read(root / "offline_quality.json"),
            "codec_costs": read(root / "codec_costs.json"),
            "codebook_only_audit": read(root / "reports/codebook_only_audit.json"),
            "limits": protocol["limits"],
            "attribution": protocol["attribution"],
        },
    )
    for name, data in (("comparison.csv", rows), ("document_nll.csv", document_rows)):
        with (root / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probes", "results"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    (probes if args.stage == "probes" else results)(args.root)


if __name__ == "__main__":
    main()
