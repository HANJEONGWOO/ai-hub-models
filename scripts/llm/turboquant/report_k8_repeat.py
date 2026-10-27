# SPDX-License-Identifier: BSD-3-Clause
"""Export descriptive tables and English paper revisions from all 40 samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = {
    "ttft_ms": "TTFT (ms)",
    "prefill_tok_per_s": "Prefill (tokens/s)",
    "decode_tok_per_s": "Decode (tokens/s)",
}


def write_new(path: Path, text: str) -> None:
    with path.open("x") as stream:
        stream.write(text)


def report(root: Path) -> None:
    data = json.loads((root / "summary.json").read_text())
    groups = data["summary"]
    gains = data["improvement_pct"]
    raw = list(csv.DictReader((root / "raw.csv").open()))
    flagged = [row["ordinal"] for row in raw if row["flags"]]
    text = [
        "Structured K8 vs Converged Lloyd-Max — 40-run report",
        "Qwen3-1.7B W4A16 / Galaxy S26 Ultra / QAIRT 2.48.0.260626 / HTP",
        "897 prompt + 128 generated; 127 AR1 decode steps; max cached tokens=1024.",
        "10 rounds x 4 configurations. Every invocation: warm-up session0; reset; measured session1.",
        "40 warm-up sessions excluded, 40 measured sessions retained; no sample exclusion.",
        "Masks: Odd=[1,3,...,27]; Boundary=[0,1,2,3,4,5,6,21,22,23,24,25,26,27].",
        "All: K8x14, K4x14, V2x28; 30277632 bytes (28.875 MiB) host KV.",
        "Model loading and warm-up excluded; no QNN optrace/detailed profiling.",
        "",
    ]
    for placement in ("odd", "boundary"):
        text += [
            placement.upper(),
            "Metric | Converged LM mean +/- sample SD | Structured mean +/- sample SD | Improvement",
        ]
        for metric, label in METRICS.items():
            a, b = [
                groups[placement + suffix][metric] for suffix in ("_lm", "_structured")
            ]
            text.append(
                f"{label} | {a['mean']:.3f} +/- {a['sample_sd']:.3f} | {b['mean']:.3f} +/- {b['sample_sd']:.3f} | {gains[placement][metric]:.2f}%"
            )
        text.append("")
    text += [
        "IMPROVEMENT BY PLACEMENT (positive=better)",
        "Metric | Odd % | Boundary % | Boundary-Odd percentage points",
    ]
    for metric, label in METRICS.items():
        text.append(
            f"{label} | {gains['odd'][metric]:.2f} | {gains['boundary'][metric]:.2f} | {data['boundary_minus_odd_improvement_pp'][metric]:+.2f}"
        )
    text += ["", "ALL DISTRIBUTIONS", "Group | Metric | Median | Min | Max | CV (%)"]
    for group, metrics in groups.items():
        for metric, value in metrics.items():
            text.append(
                f"{group} | {METRICS[metric]} | {value['median']:.3f} | {value['min']:.3f} | {value['max']:.3f} | {100 * value['sample_sd'] / value['mean']:.3f}"
            )
    text += ["", "PAIRED-ROUND BOOTSTRAP (supplemental; 10000 resamples, 95% interval)"]
    for placement, metrics in data["paired_round_bootstrap_95pct"].items():
        for metric, bounds in metrics.items():
            text.append(
                f"{placement} {METRICS[metric]} improvement: [{bounds[0]:.2f}, {bounds[1]:.2f}]%"
            )
    text += [
        "Small-n single-device intervals do not address independent devices/days; thermal drift can violate exchangeability.",
        "",
        "HISTORICAL SINGLE-RUN COMPARISON",
        "Group | Metric | Historical | New warmed mean | Relative change (%)",
    ]
    historical = {
        "odd_lm": data["historical_single_run_odd"]["lm8_converged"],
        "odd_structured": data["historical_single_run_odd"]["structured"],
        "boundary_structured": data["historical_single_run_boundary_structured"],
    }
    for group, metrics in historical.items():
        for metric, label in METRICS.items():
            old, new = metrics[metric], groups[group][metric]["mean"]
            text.append(
                f"{group} | {label} | {old:.3f} | {new:.3f} | {100 * (new / old - 1):+.2f}"
            )
    text += [
        "No historical matched Boundary converged-LM8 measurement exists.",
        "Historical performance had no same-process warm-up. Differences are not evidence of a code optimization: session state/date/device conditions also differ.",
        "",
        "THERMAL AND EXECUTION ORDER",
        f"Flagged samples retained: {flagged or 'none'}.",
    ]
    for group, values in data["diagnostics"].items():
        text.append(group + ": " + json.dumps(values, ensure_ascii=False))
    text += [
        "Only CURRENT temperatures from HAL are summarized; stale 'Cached temperatures' are excluded.",
        "Temperature sampling spans model load, warm-up and measured session, not just measured kernels.",
        "Android status/sensor severity are observable. HTP-specific throttling remains unknown (no direct HTP clock/throttle counter).",
        "Host UTC controls the schedule; original device clock is preserved in each thermal.jsonl and may differ from host time.",
        "Position counts are 2/3 per configuration, and pairwise precedence is exactly 5/5. Correlations/by-position summaries are descriptive, not causal.",
        "",
        "SCOPE AND LIMITATIONS",
        *data["limitations"],
        "Historical encoder numerical tolerance failures are unchanged; this experiment does not re-certify them.",
        "Odd comparison reuses the paper binaries; Boundary Structured reuses the original same-native-library binary. Only missing Boundary converged-LM was compiled before freeze.",
        "Quantizer and decoder both differ between methods; no decoder-only or implementation-independent LM superiority claim.",
        "No PPL was rerun. Do not copy Odd historical PPL into Boundary rows.",
        "",
        "ARTIFACTS",
        "protocol.json: predeclared schedule/conditions",
        "measurement_freeze.json: immutable model/runner/SDK/source/device hashes",
        "raw.csv: all40 measured rows + warm-up diagnostics and thermal summaries",
        "summary.csv / summary.json: full statistics",
        "attempts/*/: runner.json, runner.log, runner_command.json, thermal.jsonl, start.json, result.json",
        "audits/: source/compiled graph and I/O verification",
        "paper_revision.tex: suggested English edits (does not modify original PDF)",
    ]
    write_new(root / "report.txt", "\n".join(text) + "\n")

    def entry(group: str, metric: str) -> str:
        value = groups[group][metric]
        return f"${value['mean']:.3f} \\pm {value['sample_sd']:.3f}$"

    latex = [
        "% Suggested replacements; original ICEIC2027_Overleaf_Ready.pdf is unchanged.",
        "% Replace performance rows of Table I; preserve old quality findings separately and label them historical.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Repeated on-device performance of Structured K8 and converged Lloyd--Max with matched layer masks. Values are mean $\pm$ sample SD across ten measured sessions per configuration. Improvement denotes lower TTFT or higher throughput relative to converged Lloyd--Max, computed from group means. Each configuration allocates 28.875 MiB of host KV storage.}",
        r"\label{tab:lm-repeated}",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Placement & Metric & Converged LM & Structured K8 & Improvement (\%) \\",
        r"\hline",
    ]
    for placement in ("odd", "boundary"):
        for metric, label in METRICS.items():
            latex.append(
                f"{placement.capitalize()} & {label} & {entry(placement + '_lm', metric)} & {entry(placement + '_structured', metric)} & {gains[placement][metric]:.2f}"
                r" \\"
            )
        latex.append(r"\hline")
    latex += [
        r"\end{tabular}",
        r"\end{table*}",
        "",
        "% Experimental Methodology: replace the old one-session LM paragraph.",
        "Uninstrumented performance is measured on Qwen3-1.7B W4A16 using a Galaxy S26 Ultra, QAIRT 2.48.0.260626, and the HTP backend. Each sequence contains 897 prompt tokens and 128 generated tokens. The first generated token is produced by prefill; the subsequent 127 autoregressive steps fill, but do not exceed, the 1024-token cache capacity. TTFT excludes model loading. Prefill throughput uses 897 prompt tokens, while decode throughput is 127 divided by the summed durations of those 127 steps, following the unchanged runner.",
        "",
        "We evaluate Structured K8 with arithmetic HVX restoration and converged Lloyd--Max K8 with LUT restoration under two fixed placements. Odd uses zero-based layers 1, 3, through 27; Boundary uses layers 0--6 and 21--27. Each placement uses K8 in 14 layers, Lloyd--Max K4 in the other 14, and Lloyd--Max V2 throughout. The two quantizers use identical masks within each placement. We hold weights, activation encodings, Dense rotations, norm correction, effective FP16 scales, current/past KV policy, and Tile256 fixed.",
        "",
        "The four configurations are evaluated in ten counterbalanced rounds, giving 40 measured sessions. Before every measured session, the same loaded model performs one full warm-up sequence with the same input, followed by the runner's KV reset. These 40 warm-up sessions are excluded. Pairwise execution precedence is balanced five to five; each configuration appears two or three times at each within-round position. A fixed ten-second pause follows each invocation. All binaries and runtime settings are frozen before measurement and verified by SHA-256 before and after the experiment. No measurements are removed based on performance or thermal state.",
        "",
        "We report arithmetic means, sample standard deviations, medians, and ranges. Current HAL temperatures and Android thermal status are recorded before, during, and after each invocation; stale cached sensor values are not used. This telemetry spans loading, warm-up, and measurement and does not directly expose HTP frequency throttling. Temperature and execution-order analyses are descriptive. Optional paired-round bootstrap intervals characterize this limited repeated experiment, not variation across devices, days, or workloads.",
        "",
        "% Results: replace old 31.41%/13.09% single-run performance claims.",
        f"Table~\\ref{{tab:lm-repeated}} reports the new repeated measurements. Under Odd placement, Structured K8 changes mean TTFT by a {gains['odd']['ttft_ms']:.2f}\\% reduction and increases mean prefill and decode throughput by {gains['odd']['prefill_tok_per_s']:.2f}\\% and {gains['odd']['decode_tok_per_s']:.2f}\\%, respectively, relative to converged Lloyd--Max. Under Boundary placement, the corresponding changes are {gains['boundary']['ttft_ms']:.2f}\\%, {gains['boundary']['prefill_tok_per_s']:.2f}\\%, and {gains['boundary']['decode_tok_per_s']:.2f}\\%. Negative improvements, if any, indicate regressions rather than gains.",
        "",
        f"The Boundary-minus-Odd differences in TTFT, prefill, and decode improvement are {data['boundary_minus_odd_improvement_pp']['ttft_ms']:+.2f}, {data['boundary_minus_odd_improvement_pp']['prefill_tok_per_s']:+.2f}, and {data['boundary_minus_odd_improvement_pp']['decode_tok_per_s']:+.2f} percentage points. Layer placement and implementation jointly affect observed speed; these findings apply to the two fixed masks tested here and do not establish a universal placement-independent gain.",
        "",
        r"The historical one-session Odd comparison reported 31.41\% lower TTFT and 13.09\% higher decode throughput. We retain those values as historical reference, but base current performance claims on the new repeated results. Because the earlier experiment did not use the same-process warm-up protocol, differences from its absolute timings cannot be attributed to algorithm changes. This experiment does not rerun perplexity: earlier Odd-placement quality results remain separately labeled historical evidence, and no Boundary quality claim follows from these timings.",
        "",
        "The comparison changes both encoder quantization and decoder implementation. It therefore measures the complete evaluated implementations, not a decoder-only speedup or implementation-independent superiority over Lloyd--Max. The inherited encoder numerical-tolerance limitations remain unchanged.",
        "",
        "% Also update the Abstract and Limitations wherever the old single-run claims recur.",
        f"% Suggested Abstract performance sentence: Across ten warmed measurements per configuration, Structured K8 reduces mean TTFT by {gains['odd']['ttft_ms']:.2f} percent and increases mean decode throughput by {gains['odd']['decode_tok_per_s']:.2f} percent under Odd placement; matched Boundary placement yields {gains['boundary']['ttft_ms']:.2f} percent and {gains['boundary']['decode_tok_per_s']:.2f} percent, respectively.",
        "% Suggested Limitations replacement: The repeated comparison covers one device/model, two masks, one prompt and context capacity 1024. It does not establish broader workload/day/device generalization or directly rule out HTP-level throttling. The existing three-run Uniform comparison and encoder numerical-validation caveats remain separate limitations.",
    ]
    write_new(root / "paper_revision.tex", "\n".join(latex) + "\n")
    print(root / "report.txt")
    print(root / "paper_revision.tex")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report(args.root.resolve())


if __name__ == "__main__":
    main()
