# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Paired allocation report, fixed-test NLL/KL and offline sensitivity figures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
from benchmark_asymmetric_once import digest, read, save_new
from benchmark_nll_allocation import STATIC, audit_allocation, validate
from search_nll_allocation import kl_stats, mask_key
from summarize_native_results import performance

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nll_from_logits(logits: torch.Tensor, targets: np.ndarray) -> float:
    """Audit dumped row/target alignment independently of the runner's NLL."""
    assert len(logits) == len(targets) and torch.isfinite(logits).all()
    total = 0.0
    for start in range(0, len(logits), 32):
        chunk = logits[start : start + 32].double()
        ids = torch.tensor(targets[start : start + 32], device=logits.device).long()
        total += (
            (torch.logsumexp(chunk, -1) - chunk.gather(1, ids[:, None]).flatten())
            .sum()
            .item()
        )
    return total


def calibration_transfer(root: Path, selection: dict) -> dict:
    """Compare paired calibration changes, never use the test split for selection."""
    groups = list(
        dict.fromkeys(("uniform_odd", "uniform_even", selection["candidate_group"]))
    )
    result = {}
    for group in groups:
        mask = read(root / "plans" / f"{group}.json")["k8_layers"]
        offline = [
            read(
                root / "offline/evaluations" / mask_key(mask) / f"calibration_{i}.json"
            )
            for i in range(4)
        ]
        htp = [
            read(root / "reports" / f"{group}_calibration_{i}.json") for i in range(4)
        ]
        result[group] = {
            "offline_mean_nll": sum(row["nll_sum"] for row in offline) / 4092,
            "HTP_mean_nll": sum(row["nll_sum"] for row in htp) / 4092,
            "documents": [
                {
                    "index": i,
                    "offline_mean_nll": a["nll_sum"] / 1023,
                    "HTP_mean_nll": b["nll_sum"] / 1023,
                }
                for i, (a, b) in enumerate(zip(offline, htp, strict=True))
            ],
        }
    for row in result.values():
        for backend in ("offline", "HTP"):
            row[f"{backend}_delta_vs_uniform"] = (
                row[f"{backend}_mean_nll"]
                - result["uniform_odd"][f"{backend}_mean_nll"]
            )
    return result


def figures(root: Path, selection: dict, search: dict) -> None:
    target = root / "figures"
    target.mkdir(exist_ok=False)
    masks = {**STATIC, "nll_selected": selection["selected_layers"]}
    array = np.array(
        [[8 if i in mask else 4 for i in range(28)] for mask in masks.values()]
    )
    fig, ax = plt.subplots(figsize=(14, 3.8))
    view = ax.imshow(array, aspect="auto", cmap="Blues", vmin=4, vmax=8)
    ax.set_xticks(range(28))
    ax.set_yticks(range(len(masks)), list(masks))
    ax.set_xlabel("Layer (zero-based); V2 in every layer")
    ax.set_title("Structured K8 x 14 / LM K4 x 14: fixed 28.875 MiB KV cache")
    for row in range(len(masks)):
        for col in range(28):
            ax.text(
                col,
                row,
                str(array[row, col]),
                ha="center",
                va="center",
                color="white" if array[row, col] == 8 else "black",
                fontsize=8,
            )
    fig.colorbar(view, ax=ax, ticks=[4, 8], label="K bits")
    fig.tight_layout()
    fig.savefig(target / "layer_masks.svg")
    fig.savefig(target / "layer_masks.png", dpi=150)
    plt.close(fig)
    first = search["history"][0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, key, title in zip(
        axes,
        ("promotions", "demotions"),
        (
            "K4 -> K8: single promotion (15 K8 diagnostic)",
            "K8 -> K4: single demotion (13 K8 diagnostic)",
        ),
        strict=True,
    ):
        entries = first[key]
        ax.bar(
            [r["layer"] for r in entries],
            [r["delta_nll"] for r in entries],
            color=["#2166ac" if r["delta_nll"] < 0 else "#b2182b" for r in entries],
        )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel("Delta full-model mean NLL")
        ax.set_title(title)
    axes[-1].set_xticks(range(28))
    axes[-1].set_xlabel(
        "Layer; offline W4A16 surrogate, four calibration documents (not HTP sensitivity)"
    )
    fig.tight_layout()
    fig.savefig(target / "initial_nll_sensitivity.svg")
    fig.savefig(target / "initial_nll_sensitivity.png", dpi=150)
    plt.close(fig)
    rows = []
    for step in search["history"]:
        for direction in ("promotions", "demotions"):
            for row in step[direction]:
                rows.append(
                    {"iteration": step["iteration"], "direction": direction, **row}
                )
    write_csv(root / "layer_nll_sensitivity.csv", rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    protocol = validate(root)
    selection = read(root / "selection.json")
    groups = {name: name for name in STATIC}
    groups["nll_selected"] = selection["selected_group"]
    performance_rows, document_rows, details, all_kl = [], [], {}, {}
    base_scores = [
        read(root / "reports" / f"uniform_odd_test_{i}.json") for i in range(4)
    ]
    computed = {}
    torch.set_num_threads(4)
    for label, group in groups.items():
        if group not in computed:
            print("Summarizing recorded measurements:", group, flush=True)
            audit_allocation(root, group)
            conversion = read(root / group / "convert_report.json")
            assert conversion["attention_tile"] == 256
            assert conversion["rotated_attention"] and conversion["quantize_current_kv"]
            assert conversion["context_buckets"] == [1024]
            assert conversion["structured_k8_decoder"] == "arithmetic"
            reports = [
                read(root / "reports" / f"{group}_test_{i}.json") for i in range(4)
            ]
            assert all(
                r["scored_tokens"] == 1023 and r["kv_store_bytes"] == 30277632
                for r in reports
            )
            perf = performance(root / "reports" / f"{group}_perf.json", 897)
            assert perf["host_kv_MiB"] == 28.875
            plan = read(root / "plans" / f"{group}.json")
            assert all(
                row["config_hash"] == plan["config_hash"]
                and "htp " in row["backend_api"]
                and row["device"]["soc_model"] == "SM8850"
                and math.isclose(
                    row["ppl"], math.exp(row["nll_sum"] / 1023), rel_tol=1e-7
                )
                for row in reports
            )
            tokens = 4092
            total = sum(r["nll_sum"] for r in reports)
            kls = []
            for i in range(4):
                candidate_path = root / "reports" / f"{group}_test_{i}.logits.bin"
                base_path = root / "reports" / f"uniform_odd_test_{i}.logits.bin"
                assert (
                    candidate_path.stat().st_size
                    == base_path.stat().st_size
                    == 1023 * 151936 * 4
                )
                candidate = torch.tensor(
                    np.memmap(
                        candidate_path, mode="r", dtype="<f4", shape=(1023, 151936)
                    ).copy(),
                    device="cuda",
                )
                target_ids = np.fromfile(
                    root / "assets" / f"wikitext_w{i}.bin", dtype="<i4"
                )[1:]
                dump_nll = nll_from_logits(candidate, target_ids)
                # The C++ report rounds the accumulated double for JSON output.
                assert abs(dump_nll - reports[i]["nll_sum"]) / 1023 < 1e-6
                if group == "uniform_odd":
                    values = [0.0] * 1023
                else:
                    reference = np.memmap(
                        base_path, mode="r", dtype="<f4", shape=(1023, 151936)
                    )
                    values = kl_stats(reference, candidate)["per_token"]
                del candidate
                kls.extend(values)
                document_rows.append(
                    {
                        "group": group,
                        "window": i,
                        "scored_tokens": 1023,
                        "nll_sum": reports[i]["nll_sum"],
                        "dump_nll_sum": dump_nll,
                        "mean_nll": reports[i]["nll_sum"] / 1023,
                        "ppl": reports[i]["ppl"],
                        "delta_nll_vs_uniform": (
                            reports[i]["nll_sum"] - base_scores[i]["nll_sum"]
                        )
                        / 1023,
                        "kl_uniform_to_candidate_mean": float(np.mean(values)),
                        "kl_p95": float(np.quantile(values, 0.95)),
                        "kl_p99": float(np.quantile(values, 0.99)),
                    }
                )
            all_kl[group] = kls
            computed[group] = {
                "physical_group": group,
                "k8_layers": plan["k8_layers"],
                "config_hash": plan["config_hash"],
                "mean_nll": total / tokens,
                "ppl": math.exp(total / tokens),
                "scored_tokens": tokens,
                "window_nll_sums": [r["nll_sum"] for r in reports],
                "performance": perf,
                "kl_uniform_to_candidate": {
                    "mean": float(np.mean(kls)),
                    "p95": float(np.quantile(kls, 0.95)),
                    "p99": float(np.quantile(kls, 0.99)),
                },
                "binaries": {
                    path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
                    for path in (root / group).glob("part*_of_4.bin")
                },
            }
        entry = computed[group]
        details[label] = {
            **entry,
            "measurement_reused_from": group if label != group else None,
            "test_windows_improved_vs_uniform": sum(
                a < b - 1e-6
                for a, b in zip(
                    entry["window_nll_sums"],
                    computed["uniform_odd"]["window_nll_sums"],
                    strict=True,
                )
            ),
            "test_windows_worsened_vs_uniform": sum(
                a > b + 1e-6
                for a, b in zip(
                    entry["window_nll_sums"],
                    computed["uniform_odd"]["window_nll_sums"],
                    strict=True,
                )
            ),
        }
        base = computed["uniform_odd"]
        performance_rows.append(
            {
                "allocation": label,
                "physical_group": group,
                "K8_layers": " ".join(map(str, entry["k8_layers"])),
                "mean_nll": entry["mean_nll"],
                "ppl": entry["ppl"],
                "ppl_improvement_percent_vs_uniform": 100
                * (1 - entry["ppl"] / base["ppl"]),
                "delta_nll_vs_uniform": entry["mean_nll"] - base["mean_nll"],
                **{
                    k: entry["performance"][k]
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
                "kl_mean": entry["kl_uniform_to_candidate"]["mean"],
                "kl_p95": entry["kl_uniform_to_candidate"]["p95"],
                "kl_p99": entry["kl_uniform_to_candidate"]["p99"],
                "config_hash": entry["config_hash"],
            }
        )
    historical = root.parent / "hardware_aligned_mixed_kv_20261010/comparison.csv"
    with historical.open() as stream:
        old = [
            row
            for row in csv.DictReader(stream)
            if row["group"] in ("k6_v2", "mixed_k6_v2_mse", "mixed_k6_v2_uniform")
            and row["condition"] == "long"
        ]
    preflight = [
        read(root / "offline/preflight" / f"calibration_{i}.json") for i in range(4)
    ]
    save_new(
        root / "comparison.json",
        {
            "protocol_sha256": digest(root / "protocol.json"),
            "selection": selection,
            "groups": details,
            "calibration_emulator_vs_HTP": [
                {k: row[k] for k in ("mean_nll", "htp_mean_nll", "delta_nll")}
                | {"kl_mean": row["kl_htp_to_offline"]["mean"]}
                for row in preflight
            ],
            "calibration_change_transfer": calibration_transfer(root, selection),
            "offline_static_and_prior_MSE_mask": read(
                root / "offline/static_calibration.json"
            ),
            "historical_comparisons": {
                "csv": str(historical),
                "sha256": digest(historical),
                "rows": old,
                "limits": "Same legacy WikiText windows; historical K6/V2 and MSE allocation use LM codebooks, not Structured K8. Not a controlled claim that one selection objective is universally superior.",
            },
            "performance_sessions_per_configuration": 1,
            "limitations": protocol["limits"],
        },
    )
    save_new(
        root / "logit_kl.json",
        {
            "direction": "KL(Uniform-Odd || candidate)",
            "tokens_per_window": 1023,
            "window_order": list(range(4)),
            "values": all_kl,
        },
    )
    write_csv(root / "comparison.csv", performance_rows)
    write_csv(root / "document_nll.csv", document_rows)
    figures(root, selection, read(root / "offline/search.json"))
    print("SUMMARY", root / "comparison.csv", flush=True)


if __name__ == "__main__":
    main()
