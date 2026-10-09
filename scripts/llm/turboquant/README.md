<!--
Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
SPDX-License-Identifier: BSD-3-Clause
-->

# TurboQuant KV-cache tools

Developer tools for the opt-in TurboQuant KV-cache codec in
`qai_hub_models.models.templates.llm.turboquant`. The ABI, numeric tolerances
and current results are in
[`tutorials/llm/turboquant_design.md`](../../../tutorials/llm/turboquant_design.md).

## Binary storage on WSL

On this WSL machine, artifact directories are redirected with ordinary directory
symlinks (no automatic copy/archive logic or new CLI options):

| Existing path | Actual storage |
|---|---|
| `~/.qaihm/tmp/turboquant/` | `/mnt/d/ai-hub-models/binaries/turboquant/` |
| `<repo>/export_assets/` | `/mnt/d/ai-hub-models/export_assets/` |

Existing commands, final bundle links and report paths continue to work. New
build outputs and downloads using these paths are written directly to D, along
with their intermediate files and reports. The repo, venv and SDK stay unchanged.
Windows ADB helpers resolve symlinks before converting paths with `wslpath`.
The D drive must be mounted; explicitly choosing another output directory still
uses that directory. These directory links are machine-local, not Git-tracked
export defaults, and need setting up separately on a different machine.

## Scalar indexing: tree search only

All newly exported TurboQuant encoders use exact Lloyd-Max tree search:
four dependent comparisons for 4-bit K/V, three for the QJL 3-bit MSE stage.
The all-boundary broadcast/count implementation has been removed; there is
no strategy flag or additional profile. Existing profile names and commands
continue to work. The offline NumPy oracle remains `searchsorted(..., side="left")`.

Codebooks, boundary ties (lower index), Dense rotation, norm/effective-scale
correction, packing, current-KV policy and Native LUT decoding are unchanged.
Threshold leaves use `bit*hi + (1-bit)*lo`, avoiding static/static HTP `Where`
and preserving FP16 boundary values exactly.

Compressed configurations now record `scalar_indexing: lloyd_tree` in their
config/hash. Old broadcast bundles are rejected by conversion/reuse guards:
use a **new output directory and rebuild the ONNX/DLC/context binaries**.
Existing binaries are not automatically rewritten. The KV storage ABI is
unchanged; uncompressed INT8/INT16/FP16 baseline configuration hashes are unchanged.
Historical reports remain readable, but their performance is not a new tree measurement.

This change is covered by CPU graph/oracle, boundary/FP16-domain, packing,
QJL/Native and metadata regression tests. No model rebuild or device performance
measurement is performed as part of this code-only switch.

## K-only Dense rotation quality experiment (opt-in)

`--key-rotation-file <artifact.json>` replaces **only** the shared K rotation
constant in `k4_v4_scaled`. The matching Query rotation uses the same constant.
It retains LM-tree / LM centroids / Native LUT, Dense MatMul geometry, K4/V4,
QJL-off, quantized current KV, norm correction and one FP16 effective scale.
V rotation (seed 542), V codebook, AV, weights, and calibration are unchanged.
No Structured codebook or Bit-plane operator is involved. Omit the new flag to
retain the existing default and its unchanged configuration hash.

The experiment uses Qwen3-1.7B CL1024 and one 128x128 K matrix shared across all
layers and KV heads. A is seed 42, B is selected from seeds 42..49 by validation
Attention MSE, and C is initialized from B and trained offline. FP32 orthogonality
is validated; the existing deployment rounds the matrix to FP16. Artifact and
matrix hashes are recorded; cache configuration hashes prevent mixed rotations.

Data collection uses **the same W4A16 HTP model with the FP16-KV attention path**,
not a Hugging Face FP-weight proxy. Collection-only graphs expose Q and Attention
outputs alongside existing current K/V outputs. Those instrumented runs are
explicitly excluded from performance results. Four training, two validation,
and four heldout 1024-token windows come from distinct WikiText articles and
official train/validation/test splits, frozen before rotation selection.

Offline training uses hard LM bins in every forward and identity STE gradients
for discrete selection / FP16 and affine rounding. Adam (lr 0.001) is followed
by an FP64 polar/SVD orthogonal retraction each step, for at most 240 steps,
validation every 20, patience four. B itself is an eligible step-zero C checkpoint.
The CPU model emulates encoder affine-pair centroid arithmetic for norm correction,
Native FP16 LUT products, global masked softmax and tiled AV. CPU operation-output
rounding does not claim bit-exact HTP reductions or fused quantization; an FP16
reference-fidelity check and subsequent actual HTP probe are separate gates.
No Gaussian samples are used to select or learn the rotation.

Reproduction (fresh output root required; existing attempts/results are preserved):

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
TQ_ROT=/mnt/d/ai-hub-models/binaries/turboquant/k_rotation_quality_new
TQ_SPLIT=/mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_w4a16_split
mkdir -p "$TQ_ROT/reports"
venv/bin/python scripts/llm/turboquant/learn_key_rotation.py freeze --root "$TQ_ROT"
venv/bin/python scripts/llm/turboquant/rotation_data.py prepare --root "$TQ_ROT"
venv/bin/python scripts/llm/turboquant/convert_parts.py \
  --split-dir "$TQ_SPLIT" --out "$TQ_ROT/capture_bundle" \
  --profile baseline_fp16_kv_fp16_attn --context-length 1024 --capture-attention
bash scripts/llm/turboquant/qnn_runner/build_android.sh "$TQ_ROT/runner"
venv/bin/python scripts/llm/turboquant/rotation_data.py capture --root "$TQ_ROT"
venv/bin/python scripts/llm/turboquant/rotation_data.py pack --root "$TQ_ROT"
venv/bin/python scripts/llm/turboquant/learn_key_rotation.py train --root "$TQ_ROT"
venv/bin/python scripts/llm/turboquant/learn_key_rotation.py heldout --root "$TQ_ROT"
for group in A B C; do
  venv/bin/python scripts/llm/turboquant/rotation_probe.py all --root "$TQ_ROT" --group "$group" || break
  venv/bin/python scripts/llm/turboquant/benchmark_key_rotation.py build --root "$TQ_ROT" --group "$group" || break
done
for stage in audit push functional performance quality summarize; do
  venv/bin/python scripts/llm/turboquant/benchmark_key_rotation.py "$stage" --root "$TQ_ROT" || break
done
```

Performance uses three predeclared crossed orders (`ABC`, `BCA`, `CAB`) for both
35+128 and 897+128 token conditions, without profiling or capture outputs. All
samples and median/min/max are retained; quality uses four heldout windows once
each, with aggregate PPL computed from total NLL. Three repeats are descriptive,
not proof of statistical equivalence. Existing encoder numerical-gate failures
are reported separately and their tolerance is not relaxed. The primary question
is whether C improves heldout quality over B at unchanged runtime and KV storage;
negative outcomes do not trigger a larger training/search campaign.

The HTP probe reports both the unconditioned FP32 graph-oracle error and an
isolated FP16 Attention error using HTP's actual current codes/scales. The latter
does not waive encoder or full-graph failures; both are retained in
`probes/<group>/validation_isolated.json`. See design §25 for the experiment and
its limitations, including a step-zero C rollback when training does not beat B.

In the 2026-10-06 run, B selected seed 48: heldout CPU Attention MSE fell 9.48%
and device PPL fell from 36.458652 to 33.678515 on four new heldout documents.
Training stopped at step 80 without beating B, so C selected step zero and is
identical to B. This is a random-selection result, **not a learned-rotation gain**.
Long-input decode medians were A/B/C 37.598/37.628/37.706 tok/s, with 28.875 MiB
KV in every run. Defaults remain unchanged; older PPL tables use different windows.

## Per-layer K selection and shared-update diagnosis (opt-in)

`--key-rotation-file` also accepts `qaihm-k-layer-dense-rotation-v1` artifacts.
Each layer shares one fixed K matrix across its KV heads; its Query uses the same
matrix. Matrices with the same seed are shared as constants within a graph.
The policy's layer order and matrix hashes are included in the cache/config hash.
Missing layers, tampered matrices and mixed cache policies are rejected. The V
path and default shared seed42 profile are unchanged. This is not Structured
codebook or Bit-plane attention.

The followup freezes its protocol/data before any new losses. P selects each
layer's minimum **original validation absolute MSE** among seeds42..49, without
changing the criterion. Four additional validation documents gate promotion;
four new heldout test documents are reserved for whole-model PPL. A/B binaries
are hash-checked and reused, with new writable wrapper directories for runtime
metadata. No original experiment result is overwritten.

Independently, shared B is diagnosed using fixed original train/validation subsets
(windows0,1; layers0,4,8,12,16,20,24,27; eight fixed query positions; all heads).
Three independent one-step Adam+STE+SVD updates use lr1e-3/1e-4/1e-5. Only if train
hard loss improves is one predeclared-best learning rate continued to 20 total
steps; B remains step zero. Learned D and layer-selected P are never combined.
D must improve both fixed subsets, full original validation and new validation
before HTP/full-model evaluation; otherwise it is retained as a negative diagnosis.

Fresh-root reproduction (requires the original experiment, cached WikiText data,
local SDK/device and `matplotlib` for the scientific heatmap):

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
TQ_FOLLOW=/mnt/d/ai-hub-models/binaries/turboquant/k_rotation_followup_new
TQ_ORIGINAL=/mnt/d/ai-hub-models/binaries/turboquant/k_rotation_quality_20261006
venv/bin/python scripts/llm/turboquant/rotation_followup.py freeze \
  --root "$TQ_FOLLOW" --source "$TQ_ORIGINAL"
venv/bin/python scripts/llm/turboquant/rotation_followup.py analyze --root "$TQ_FOLLOW"
venv/bin/python scripts/llm/turboquant/rotation_followup.py diagnose --root "$TQ_FOLLOW"
venv/bin/python scripts/llm/turboquant/rotation_data.py capture --root "$TQ_FOLLOW" --splits validation
venv/bin/python scripts/llm/turboquant/rotation_data.py pack --root "$TQ_FOLLOW" --splits validation
venv/bin/python scripts/llm/turboquant/rotation_followup.py validate --root "$TQ_FOLLOW"
venv/bin/python scripts/llm/turboquant/rotation_followup.py invariants --root "$TQ_FOLLOW"
venv/bin/python scripts/llm/turboquant/benchmark_rotation_followup.py probes --root "$TQ_FOLLOW"
# Build only a candidate listed in cpu_quality.json / accepted_groups.
venv/bin/python scripts/llm/turboquant/benchmark_rotation_followup.py build --root "$TQ_FOLLOW" --group P
for stage in audit push functional performance quality summarize; do
  venv/bin/python scripts/llm/turboquant/benchmark_rotation_followup.py "$stage" --root "$TQ_FOLLOW" || break
done
```

If D independently passes its gates, build it with `--group D` before `audit`;
the frozen four-group crossed order is then used. Otherwise the orders are
`ABP`, `BPA`, `PAB`, three repeats each for 35+128 and 897+128 tokens, CL1024.
Old encoder numerical failures and unconditioned FP32-oracle Attention errors
remain explicit; conditioned Attention checks do not waive them.

Outputs include `layer_selection.json`, `reports/layer_seed_table.csv`, PNG/SVG
heatmaps, `diagnostics/` per-step logs/matrices, `diagnostics.json`,
`cpu_quality.json`, `rotations/P.json`, and the actual device reports. Detailed
followup results and limitations are recorded in design §26.

The 2026-10-09 run is a **negative whole-model result**: P improved new-validation
Attention MSE by 2.11% versus B but worsened heldout PPL by 4.17%
(A/B/P 25.051740 / 25.553117 / 26.618682 on four new documents). Shared D's selected
step16 improved its fixed subset but worsened full original/new validation, so
it was not built or device-tested. Long decode medians were 37.661 / 37.684 /
37.722 tok/s, all with 28.875 MiB KV and 96.929 MiB I/O. P added 524 KiB of actual
context binaries. Neither candidate was promoted; default A and all old results
remain unchanged. See `k_rotation_followup_20261009/reports/comparison.json`.

## Same-validation Attention MSE versus whole-model NLL (evaluation only)

`benchmark_rotation_mse_nll.py` reuses the followup's immutable A/B/P binaries,
rotations, runner and saved validation Attention MSE. It scores exactly the same
four validation windows (Slammiversary, Sorry, Meridian, Fort Scott) once per
configuration, CL1024 / 1023 next-token targets per document. No heldout inputs,
new captures, MSE recomputation, compilation, selection or training are involved.
The fresh asset directory contains only those four token files and RoPE. Original
text, offset and token bytes are verified using the pinned local dataset/tokenizer.
Runtime metadata and reports are written only to a fresh experiment directory;
binary payloads are immutable links to the originals. Local and device hashes are
checked, and an existing attempt is never automatically retried or overwritten.

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
TQ_NLL=/mnt/d/ai-hub-models/binaries/turboquant/k_rotation_mse_nll_new
TQ_FOLLOW=/mnt/d/ai-hub-models/binaries/turboquant/k_rotation_followup_20261009
venv/bin/python scripts/llm/turboquant/benchmark_rotation_mse_nll.py freeze \
  --root "$TQ_NLL" --source "$TQ_FOLLOW"
for stage in push score summarize; do
  venv/bin/python scripts/llm/turboquant/benchmark_rotation_mse_nll.py "$stage" \
    --root "$TQ_NLL" || break
done
```

The fixed document orders are ABP / BPA / PAB / ABP, **not repetitions**.
`reports/document_nll.{json,csv}`, `document_deltas.csv`, `summary.csv`,
`layer_mse_contributions.csv` and `comparison.json` retain all measurements.
NLL uses natural logarithms; aggregate PPL is `exp(sum(NLL) / 4092)`, not an
arithmetic mean of document PPL. Saved MSE covers 32 query positions/window;
NLL covers all 1023 next-token targets. Per-document MSE was not saved, so no
documentwise MSE–NLL coefficient or layerwise causal attribution is claimed.
These already-observed validation documents are not a new heldout evaluation.
Defaults and the old encoder numerical-validation status are unchanged.

The 2026-10-09 same-validation result is **not monotonic in Attention MSE**:
A/B/P mean NLL is 3.449334 / 3.388063 / 3.439958 and PPL is
31.479416 / 29.608540 / 31.185639. P improves aggregate Attention MSE by 2.11%
versus B, but worsens NLL by 0.051895 nats/token and PPL by 5.33%; NLL is worse
on all four documents. B improves both metrics versus A. This is evidence against
treating a small local-MSE gain as a sufficient whole-model quality criterion,
not proof of no statistical correlation or a particular causal mechanism.
Use whole-model validation NLL as an adoption gate; new objective/training work
was not performed. See design §27 and `k_rotation_mse_nll_20261009/reports/`.

## FP16 KV + FP16-input attention control (opt-in)

`baseline_fp16_kv_fp16_attn` is a separate, uncompressed control. It does **not**
replace `baseline_int16_kv`, change the default model, or change TurboQuant.
Unlike the old int16-cache baseline's quantized attention-side KV boundary:

- Past KV graph I/O and host storage are FP16, without affine cache encodings.
- Current KV is rounded to the same FP16 cache output **before** attention,
  in both AR128 prefill and AR1 decode. There is no raw-current bypass.
- Both operands of QK and AV are FP16; no int8 conversion is allowed on the
  cache read path. The HTP accumulation dtype is not claimed to be FP16.
- W4A16 weights, parameter encodings and non-KV producer encodings are kept.
  KV-specific 8-bit taps are regridded to 16-bit using the same calibrated range
  as TurboQuant, then converted to FP16. Query, score, mask, softmax and final
  attention-output calibrated boundaries are retained. This is **not** an
  all-FP16 model or a recalibrated/retrained checkpoint.
- No extra TurboQuant rotation, codec or QJL is added; original checkpoint
  operations (including SpinQuant) remain. This control uses untiled attention;
  TurboQuant still uses rotated tiled attention, so throughput/PPL differences
  are not attributable solely to compression.

Build explicitly from the existing 1.7B split, into a **new** directory:

```bash
TQ_FP16=/mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_fp16_attention_new
TQ_SPLIT=/mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_w4a16_split
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src venv/bin/python \
    scripts/llm/turboquant/convert_parts.py --split-dir "$TQ_SPLIT" \
    --out "$TQ_FP16/bundle" --profile baseline_fp16_kv_fp16_attn \
    --context-length 1024 --sequence-lengths 128 1
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src venv/bin/python \
    scripts/llm/turboquant/verify_fp16_attention.py --bundle "$TQ_FP16/bundle" \
    --split-dir "$TQ_SPLIT" --report "$TQ_FP16/reports/fp16_graph_audit.json"
```

The audit checks compiled DLC types and cache dependencies, not just ONNX
`Cast` nodes. `--split-dir` additionally checks unchanged original weights and
parameter encodings, and restricts activation edits to recorded KV taps.
Separate part builds can be assembled using `assemble_native_bundle.py
--fp16-attention`; its usual Native-package validation remains the default.

`benchmark_fp16_attention_once.py` stages immutable historical int16/TurboQuant
bundles through new symlink directories, plus the newly built FP16 control.
All groups use **fixed C1024**, the same runner/tokenizer/RoPE/input assets,
35+128 and 897+128 generation conditions (one session each), and four separate
1024-token PPL windows (4092 scored tokens). No repeated timing attempts are
overwritten; reset/EOS diagnostics are separate and EOS failures are labelled.
This fixed-context experiment must not be mixed with historical bucketed
short-prompt or PPL runs. Single-run timings have no variance estimate.

```bash
bash scripts/llm/turboquant/qnn_runner/build_android.sh "$TQ_FP16/runner"
for stage in push functional performance quality summarize; do
    OPENBLAS_NUM_THREADS=1 PYTHONPATH=src venv/bin/python \
        scripts/llm/turboquant/benchmark_fp16_attention_once.py "$stage" \
        --int16-bundle /mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_baseline_int16_kv_cl1024 \
        --fp16-bundle "$TQ_FP16/bundle" \
        --turboquant-bundle /mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_current_kv_native_20260922_final \
        --assets /mnt/d/ai-hub-models/binaries/turboquant/device_assets_cl1024 \
        --runner "$TQ_FP16/runner/qnn-llm-runner" \
        --reports "$TQ_FP16/reports" --name qwen3_1_7b_fp16_attention_new || break
done
```

## Stage profiling (diagnostic only)

`profile_stages_once.py` reuses the frozen FP16/TurboQuant binaries from an
existing FP16 comparison. It verifies local/device context hashes and uploads
only a separately named runner, leaving the benchmark runner untouched. Each
group runs **one** 897-prompt + 128-generation session at fixed C1024. All eight
AR128 prefill chunks and AR1 decode steps 0/63/126 are captured in that session.
These are diagnostic profiles, **not** new throughput/PPL benchmark results.

```bash
TQ_PROFILE=/mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_stage_profile_new
bash scripts/llm/turboquant/qnn_runner/build_android.sh "$TQ_PROFILE/runner"
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/profile_stages_once.py \
    --reference-reports /mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_fp16_attention_20260929/reports \
    --assets /mnt/d/ai-hub-models/binaries/turboquant/device_assets_cl1024 \
    --runner "$TQ_PROFILE/runner/qnn-llm-runner" --out "$TQ_PROFILE/reports"
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/summarize_stage_profiles.py \
    --reports "$TQ_PROFILE/reports"
```

The reference device bundles must already exist and match the historical hashes.
Use a new output directory; an attempted inference cannot be silently retried.
Raw events retain type/unit/parent/children, and host graph-execution wall time
is separated from profile-event retrieval. The analyzer attributes **leaf NODE
cycle events** using the frozen ONNX/DLC, excludes inclusive parent counters,
and exports per-node CSV plus `stage_summary.json`. Percentages are counter
shares, not wall-time fractions or predicted speedups. Zero-cycle nodes may be
fused/eliminated/uninstrumented; encoder rotation GEMMs and many AV products
are zero in this SDK's detailed report, so their standalone cost is unresolved.

Direct runner options are `--profile-prefill-all`, `--profile-decode-steps 0 63
126` and `--runner-name <uploaded-filename>`. Existing single-step flags and
unprofiled defaults are unchanged. `run_device_llm.py` marks profiled reports
`diagnostic_only` because the profile handle is active for the whole runtime,
even on steps whose events are not exported. See design §22 for the measured
stage split and limitations; no model, weights or calibration are modified.

### HTP optrace (same stage taxonomy)

`build_optrace_contexts.py` adds `--profiling_level detailed --profiling_option
optrace` when rebuilding **contexts only** from frozen quantized DLCs. It keeps
the original graph set/weight sharing, native package and O3/v81 settings. All
source DLCs are hash-checked before/after; source contexts are never overwritten.
The SDK emits a schematic per graph in the build working directory.

```bash
TQ_TRACE=/mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_optrace_new
BUILD_OPTRACE_SMOKE=1 bash scripts/llm/turboquant/qnn_runner/build_android.sh "$TQ_TRACE/runner"
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/build_optrace_contexts.py \
    --source /mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_fp16_attention_20260929/bundle \
    --out "$TQ_TRACE/fp16" --parts 1 2 3 4
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/build_optrace_contexts.py \
    --source /mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_current_kv_native_20260922_final \
    --out "$TQ_TRACE/turboquant" --parts 1 2 3 4
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/profile_optrace_once.py measure \
    --root "$TQ_TRACE" \
    --reference-reports /mnt/d/ai-hub-models/binaries/turboquant/qwen3_1_7b_fp16_attention_20260929/reports \
    --assets /mnt/d/ai-hub-models/binaries/turboquant/device_assets_cl1024
for group in fp16 turboquant; do
    venv/bin/python scripts/llm/turboquant/render_optrace.py \
        --logs "$TQ_TRACE/traces/$group" --contexts "$TQ_TRACE/$group" \
        --out "$TQ_TRACE/rendered/$group" --jobs 4 || break
done
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/summarize_optrace.py --root "$TQ_TRACE"
```

Run commands only after the preceding stage succeeds. The optional `pilot`
action of `profile_optrace_once.py` uses the same arguments but executes only
TurboQuant part 2 with synthetic zero inputs. It checks trace serialization,
**not performance or quality**; part 2 must already have been built. A failed
pilot may be preserved under a separate `--pilot-label`. Real `measure` sessions
are attempt-guarded and run once per group, using 8 prefill chunks and decode
0/63/126 (44 graph trace logs each). A separately named runner and device bundle
preserve the original benchmark artifacts. Direct runner support is exposed via
`run_device_llm.py run --optrace-out <new-local-directory>` with profiling step
selection; extended QNN events, including opaque trace objects, are serialized.

The analyzer selects physical Core Overview HVX/HMX lanes and excludes duplicate
views and Non Executed Tensors. **Trace durations are cycles, not microseconds.**
It reports summed busy cycles, interval unions, DMA transfer/wait/control, and
SDK graph-time counters separately. Summing parallel HVX workers is work, not
wall latency. HMX arithmetic attributed to `*_post_reshape` is associated with
its verified source FullyConnected/MatMul; real reshape/transpose/format kernels
remain layout. Gzip traces retain per-kernel QNN names and hardware resource.
Optional topology/duplicated-view rendering is disabled to limit host export
cost, without changing the captured device data. See design §23 for results.

## Model selection: Qwen3-4B is opt-in

`benchmark_model_once.py` supports `qwen3_1_7b` (the unchanged default),
`qwen3_0_6b`, `qwen3_4b` and `qwen3_8b` (explicit `--model-id` only). It prepares a W4A16 split checkpoint,
builds **int16 KV** and **current Dense+Native K4/V4, QJL-off, current KV
quantized** bundles, audits them, and measures both newly. It does not reuse
1.7B metrics for the 4B comparison. Model definitions and global defaults are
unchanged. Builds run one part/process at a time to limit peak host memory.

Example (from the repo root; every stage must succeed before the next):

```bash
TQ_WORK=/mnt/d/ai-hub-models/binaries/turboquant/qwen3_4b_experiment
TQ_RUNNER=$HOME/.qaihm/tmp/turboquant/qnn_runner/model_compare/qnn-llm-runner
bash scripts/llm/turboquant/qnn_runner/build_android.sh "$(dirname "$TQ_RUNNER")"
for stage in prepare build audit push functional performance quality summarize; do
    PYTHONPATH=src OPENBLAS_NUM_THREADS=1 venv/bin/python \
        scripts/llm/turboquant/benchmark_model_once.py "$stage" \
        --model-id qwen3_4b --work-dir "$TQ_WORK" --runner "$TQ_RUNNER" || break
done
```

The stages can also be invoked individually for inspection. `build --groups
baseline_int16 --parts 2` selects a group/part without enabling parallel builds.
Build and measurement attempt logs are retained; an existing attempt log is
never silently overwritten. Use a fresh experiment directory for a new complete
comparison, and do not repeat successful performance sessions to pick a result.

Conditions remain CL1024, short chat prompt or 897 prompt tokens followed by
128 generated tokens, **one performance session per group/input condition**,
plus separate reset/EOS/cache-bucket diagnostics and four 1024-token WikiText
windows (4092 scored tokens in total). Int16 retains the historical fixed C1024
policy; TurboQuant uses C128/256/512/1024. Only the long case has identical
decode graph context lengths for both. Reports include TTFT, prefill/decode,
host KV, end VmRSS, peak VmHWM and token-weighted PPL; results are written to
`$TQ_WORK/reports/comparison.json`. A single run has no variance estimate.

Split manifests record model identity and ONNX/external-weight hashes. Conversion,
bundle assembly and benchmark stages reject mismatched model/checkpoint metadata;
the on-device wrapper rejects the wrong tokenizer/RoPE asset set before execution.
Qwen3-4B is validated as 36 layers, hidden size 2560, 32 query heads, 8 KV heads and
**explicit head dimension 128** (not 2560/32). CL1024 host KV expectations are
144 MiB for int16 and 37.125 MiB for K4/V4 with FP16 scales; these are layout
checks, not total process or device memory estimates.

The v6 4B checkpoint places `K / sqrt(128)` before attention. The rotated
path preserves this attention scale as `(Q / sqrt(128)) @ R.T`, without
scaling the whole past K cache; legacy unrotated tiles keep K-side division.
Only positive constant scalar divisors are accepted. Cache guards are inserted
before their first consumer, including the earlier RoPE layout nodes in this
checkpoint. Neither adaptation changes the 1.7B codec defaults.

On this machine the 4B checkpoint cache is also a directory link:
`~/.qaihm/qai-hub-models/models/qwen3_4b` →
`/mnt/d/ai-hub-models/checkpoints/qwen3_4b`. Other machines keep their configured
checkpoint cache; `--checkpoint PATH` accepts an already downloaded local
checkpoint. Compilation outputs use the chosen `--work-dir` directly.

The completed Qwen3-4B device comparison (2026-09-23) is recorded in design
§19: long-context decode **20.105→16.645 tok/s**, host KV **144→37.125 MiB**,
and PPL **18.4237→21.7333** (int16→current TurboQuant). Both controls were
built and measured anew, once per input condition. These results do not change
the default model or establish a speed/quality advantage for the 4B codec.

## Qwen3-8B: FP16 attention versus current TurboQuant

The 8B model is opt-in and uses its published v5 W4A16 checkpoint, five split
parts and 36 transformer layers (32 query heads, 8 KV heads, head dimension 128).
It does not change the default model or recalibrate/retrain the checkpoint.
The FP16 control retains W4A16 outside the audited FP16 KV/attention path.
The compiled FP16 audit recognizes the checkpoint's K/scalar division as
QAIRT `Eltwise_Binary` operation 2 only when it matches the source `Div`, has
a static scalar divisor and preserves FP16 throughout; other elementwise
operations and hidden int8 conversions remain rejected.

Use a fresh work directory on D. Prepare the model-specific split and input
assets, then build one part per process, serially:

```bash
TQ_WORK=/mnt/d/ai-hub-models/binaries/turboquant/qwen3_8b_fp16_turboquant_new
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 venv/bin/python \
    scripts/llm/turboquant/benchmark_model_once.py prepare \
    --model-id qwen3_8b --cl1024-only --work-dir "$TQ_WORK"
for tq_part in 1 2 3 4 5; do
    PYTHONPATH=src OPENBLAS_NUM_THREADS=1 venv/bin/python \
        scripts/llm/turboquant/convert_parts.py --split-dir "$TQ_WORK/split" \
        --out "$TQ_WORK/fp16" --profile baseline_fp16_kv_fp16_attn \
        --context-length 1024 --sequence-lengths 128 1 --parts "$tq_part" || break
done
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 venv/bin/python \
    scripts/llm/turboquant/benchmark_model_once.py build --groups turboquant \
    --model-id qwen3_8b --cl1024-only --work-dir "$TQ_WORK"
```

Only continue after all five parts of each group have completed. Audit the
compiled attention boundaries and unchanged FP16 source weights/encodings:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 venv/bin/python \
    scripts/llm/turboquant/verify_fp16_attention.py --bundle "$TQ_WORK/fp16" \
    --split-dir "$TQ_WORK/split" --report "$TQ_WORK/reports/fp16_graph_audit.json"
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 venv/bin/python \
    scripts/llm/turboquant/verify_rotated_attention.py --bundle "$TQ_WORK/turboquant" \
    --report "$TQ_WORK/reports/turboquant_graph_audit.json"
bash scripts/llm/turboquant/qnn_runner/build_android.sh "$TQ_WORK/runner"
for stage in push functional performance quality summarize; do
    PYTHONPATH=src OPENBLAS_NUM_THREADS=1 venv/bin/python \
        scripts/llm/turboquant/benchmark_fp16_attention_once.py "$stage" \
        --model-id qwen3_8b --groups fp16 turboquant \
        --fp16-bundle "$TQ_WORK/fp16" --turboquant-bundle "$TQ_WORK/turboquant" \
        --assets "$TQ_WORK/assets" --runner "$TQ_WORK/runner/qnn-llm-runner" \
        --reports "$TQ_WORK/reports" --name "$(basename "$TQ_WORK")" || break
done
```

Both groups use fixed C1024 for short and 897-token prompts, with 128 generated
tokens and **one session per group/input condition**. Four separate 1024-token
WikiText windows supply token-weighted PPL (4092 scored tokens). TTFT,
prefill/decode throughput, host KV, end VmRSS and peak VmHWM follow the existing
FP16 comparison. Expected cache capacity is **144 MiB FP16 / 37.125 MiB TQ**;
these are layout checks, not estimates of total process/device memory. Reset
and EOS diagnostics are separate; EOS-failed groups remain explicitly labelled.
Hashes freeze both groups' binaries, runner, assets and compiled graph audits.
The historical three-group 1.7B CLI remains the default.

The completed 8B comparison (2026-10-04) is recorded in design section 24 of
[`turboquant_design.md`](../../../tutorials/llm/turboquant_design.md).
For the long prompt, FP16 versus TQ measured **9.136 / 10.431 tok/s decode**,
**144.000 / 37.125 MiB host KV**, and **12.086954 / 12.658172 PPL**.
TQ's observed decode gain was 14.18%, with 74.22% less cache and 4.73% higher
PPL; prefill fell from 824.437 to 692.561 tok/s. These are single runs, not a
variance-controlled performance claim. Both reset/EOS checks and both compiled
graph audits passed. Peak process VmHWM did not decrease (1463.125 / 1480.098
MiB), despite lower end VmRSS; neither metric represents total NPU memory.

Artifacts are under
`/mnt/d/ai-hub-models/binaries/turboquant/qwen3_8b_fp16_turboquant_20261003/`
(the directory uses the build start date). `reports/summary.json` contains all
metrics, `reports/experiment.json` freezes the inputs, and raw generation/PPL
reports, audit reports and build logs are retained beside them. The two groups'
serial builds took 4:31:27, excluding split/assets preparation and evaluation;
`reports/build_resource_usage.json` records the build process resource usage.

## Qwen3-0.6B: fixed C1024 comparison

Use `--model-id qwen3_0_6b --cl1024-only` on **every stage** for the smaller
model and a fixed-context comparison. This mode builds only AR1 and AR128 at
C1024 for both int16 KV and current Dense+Native K4/V4 (QJL-off, current KV
quantized). No C128/256/512 graphs or short-input performance sessions are run.
The existing model and bucket-policy defaults are unchanged when omitted.

```bash
TQ_WORK=/mnt/d/ai-hub-models/binaries/turboquant/qwen3_0_6b_cl1024_experiment
TQ_RUNNER=$HOME/.qaihm/tmp/turboquant/qnn_runner/model_compare/qnn-llm-runner
for stage in prepare build audit push functional performance quality summarize; do
    PYTHONPATH=src OPENBLAS_NUM_THREADS=1 venv/bin/python \
        scripts/llm/turboquant/benchmark_model_once.py "$stage" \
        --model-id qwen3_0_6b --cl1024-only \
        --work-dir "$TQ_WORK" --runner "$TQ_RUNNER" || break
done
```

Reuse an existing compatible runner or build it using the command above.
Formal performance is **one 897-prompt + 128-generation session per group**.
PPL remains four 1024-token WikiText windows per group, evaluated separately.
The functional `switches` report name is retained for compatibility, but in
this mode checks cache growth while staying at C1024, not bucket switching.
The summary contains only `long` performance results; changing the context
policy between build/staging/measurement is rejected. Attempt logs are never
silently overwritten.

The published v2 checkpoint uses the model's W4A16 recipe, including its
designated int8 weight exception; both KV configurations share exactly that
checkpoint. Qwen3-0.6B has 28 layers, hidden size 1024, 16 query heads, 8 KV
heads and explicit head dimension **128**, not 1024/16. Its two split parts are
embedding and all transformer layers + LM head. Its CL1024 host KV capacity is
the same as 1.7B: **112 MiB int16 / 28.875 MiB K4/V4**, despite fewer weights.
The compiled boundary audit recognizes wide Concat → int8 Convert → 16×8
attention by graph structure, including this checkpoint's ordinary ONNX tensor
names. It still rejects narrowing before the Concat or an incorrect operand/type;
the compatibility change does not alter the checkpoint or execution graph.
On this machine its checkpoint cache is linked to
`/mnt/d/ai-hub-models/checkpoints/qwen3_0_6b` so downloads remain on D.

The 2026-09-24 0.6B comparison completed, but **current TurboQuant failed the
generation-quality check**: repetitive output, no EOS within 600 generated
tokens, and four-window PPL **25.7000 → 59.0611** (int16 → TurboQuant).
Its decode **49.324 → 48.344 tok/s** is therefore a diagnostic comparison,
not evidence of a quality-valid 0.6B TurboQuant baseline. See design §20.

The normal workflow stops on functional failure. For an explicitly labelled
diagnostic measurement after reviewing an **EOS-only** failure, pass
`--allow-eos-failure` to `performance` and `summarize` only. Reset and cache
checks must still pass, and the failed EOS result is retained. The option is
off by default; it does not change the model or make the functional stage pass.
`performance_policy.json` records this decision, the original check results
and whether quality was measured before timing. EOS-failed groups receive
`diagnostic_only: true` in the summary. The policy and existing attempt files
prevent repeated performance runs; summary requires the same explicit policy.

## Codec defaults

**The default PolarQuant rotation is now `dense_qr`, with QJL off.** Both
`get_profile(...)` and `PolarQuantReference(...)` select it when rotation is
omitted. K/V remain 4-bit for `k4_v4[_scaled]`; Native LUT, norm correction,
tile size and buckets are independent options and are not changed by rotation.
Dense QR is generated once on the host (K seed 42, V seed 542), then embedded
as graph constants; no QR or RNG executes per token on the device. The actual
float32 matrix digests are included in the config hash.

For the current `k4_v4_scaled` export profile, **Native Decode4 is also the
default**, together with rotated attention and 256-token tiles. No Native
enable flag is needed. `--no-native-decoder` explicitly selects the graph
decoder while keeping those attention defaults. Baselines and legacy raw-norm
profiles keep their existing paths; context-bucket defaults are unchanged.
The default package is `~/.qaihm/tmp/turboquant/native_decoder_hvx_20260918`;
override it with `--native-decoder-package` or `TURBOQUANT_NATIVE_DECODER_PACKAGE`.
Missing libraries cause an error, not an automatic build or silent fallback.

**Current-token KV is quantized before attention by default.** The final export
pass reads the same packed outputs and stored-precision scales that the runner
appends to cache. This applies to AR1 decode and all current tokens in AR128
prefill; the causal mask and past/current ordering are unchanged. Dense rotation,
Native decoding, K4/V4, and QJL-off remain the defaults. The encoder is shared
with cache output, and each current K/V decode is shared across GQA heads/tiles.
No extra full-past-cache restoration or host-side quantization is introduced.

Use `--no-quantize-current-kv` only to reproduce the historical raw-current-KV
path. The compilation policy is recorded as `quantize_current_kv` in bundle and
runtime reports, separate from the unchanged codec/storage `config_hash`.
Conversion and bundle assembly reject mixing these policies. Build into a new
directory: existing context binaries do not change when Python defaults change.
The host-only `evaluate_qwen3_kv.py` uses the same default current-token policy
and legacy override, but remains a float64 codec oracle, not Native FP16 emulation.

The 2026-09-22 device comparison is recorded in design §18: long-context decode
38.112→37.931 tok/s, unchanged 28.875 MiB host KV, and **PPL 20.608→25.661**.
Historical controls were reused; the new default was measured once per input
condition. `benchmark_current_kv_once.py` runs that comparison with the stages
`push`, `functional`, `performance`, `quality`, `summarize`. Review functional
results before performance and quality. The previous encoder numerical gate
limitation remains; current-token coverage is not a claim of quality neutrality.

## Orthogonal K-only QJL (K3+1 / V4)

`--profile k3qjl_v4_scaled` enables the separate format-3 QJL path. Existing
`k4_v4_scaled` defaults stay Dense+Native **without QJL**. QJL requires Dense QR,
Native Decode4, and rotated tiled
attention; these are selected automatically for the new profile.

- K uses three MSE index bits and one QJL sign bit per coordinate; V stays
  four-bit MSE. Each stored nibble is `index | (positive_sign << 3)`.
- Orthogonal QJL projection `S` uses seed 1042 (`K seed + 1000`), QR column-sign
  correction, and **no determinant correction**. A zero projection maps to +1.
- The append encoder forms `r = K - K_mse` using the actual Native FP16 MSE
  decode. It stores `sqrt(pi/2)/sqrt(128) * ||r||` as `tq_key_L_qjlscale_out`.
  There is no `2/pi` shrinkage. The QNN runner preserves this extra stream
  across reset, append, and bucket switches; rebuild the runner for format 3.
- Attention adds `(q @ S.T) @ (signs * qjlscale).T` per K tile, before the
  existing mask and global softmax. Current K uses its packed K3+1 reconstruction
  and residual correction too (only the explicit legacy option leaves it raw).
  No full residual K cache or per-past-token inverse rotation is constructed.
- The same Native Decode4 package reads MSE indices using a repeated eight-entry
  LUT and reads signs using a `[-1]*8 + [+1]*8` LUT. No DSP binary change is needed.
  This is tiled graph composition, **not one fused packed-attention kernel**.
- K payload remains 4-bit, but metadata grows by one FP16 scalar per K vector.
  Qwen3-1.7B / C1024 host KV is 29.3125 MiB versus 28.875 MiB without QJL.
  This is host KV storage, not peak NPU memory.

```bash
PYTHONPATH=src python scripts/llm/turboquant/convert_parts.py \
    --split-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_w4a16_split \
    --out ~/.qaihm/tmp/turboquant/qwen3_1_7b_qjl_native_new \
    --profile k3qjl_v4_scaled --context-buckets 128 256 512 1024
bash scripts/llm/turboquant/qnn_runner/build_android.sh \
    ~/.qaihm/tmp/turboquant/qnn_runner/qjl
```

Use `validate_qjl_encoder.py` for reference/HTP encoder checks and
`validate_native_decoder.py --lut qjl_sign` (or `--lut mse3`) for Native LUT
correctness. `verify_rotated_attention.py` also audits QJL I/O, products,
constants, and tile sizes. The generic Python `TurboQuantKVCache` rejects
format 3 instead of silently dropping QJL; use `QJLKeyReference` for its CPU
oracle and the dedicated QNN runner for LLM execution.

The standalone 2026-09-19 probe passed the QJL stage at the unchanged 0.2%
scale tolerance and 0.001 normalized sign-boundary tolerance. The MSE stage
still failed two T128 real-KV cases (maximum scale error 0.21314%); the overall
encoder gate therefore remains failed. Numerical validation and end-to-end
performance/PPL are reported separately.

`benchmark_qjl_once.py` stages the int16 / Dense+Native / QJL bundles and runs
one performance session per configuration and input condition (35 or 897 prompt
tokens, plus 128 generated tokens). Reset/EOS/bucket diagnostics and four
WikiText score windows are separate. It refuses existing report/attempt files;
use a fresh report directory and unique `--device-prefix` for a new experiment.
Run `push`, then `functional`, review those results, then run `performance` and
`quality`. Audit the final QJL bundle with `verify_rotated_attention.py` into
`REPORTS/boundary_qjl_native.json` before using `summarize_qjl_results.py`.
Design §17 records the measured results, limitations, and artifact paths.

The older results/examples below use explicit `--rotation fwht`. Use a **new
bundle directory** for dense models: existing binaries do not change with Python
defaults, and FWHT/dense packed cache states are not interchangeable. CLI
conversion, standalone validation and host evaluation accept `--rotation fwht`
for historical reproduction. Historical conversion/evaluation commands also need
`--no-quantize-current-kv` to reproduce their raw-current-KV policy. Uncompressed
baselines are unchanged.

| Script | Purpose | Needs |
|---|---|---|
| `generate_constants.py` | Freeze codebooks and FWHT signs from turboquant_plus at the pinned commit; `--check` verifies `constants.py` | turboquant_plus checkout |
| `verify_reference.py` | Compare the in-repo oracle with the pinned reference; `--write-golden` refreshes the unit-test fixture | turboquant_plus checkout |
| `evaluate_qwen3_kv.py` | Host-only reference quality run (PPL/KL/KV stats) and real KV snapshot for device inputs | HF checkpoint, GPU recommended |
| `htp_codec_validation.py` | Build codec graphs, run them on an Android HTP device over adb, compare with the oracle | QAIRT 2.48, adb, SM8850 device |
| `split_checkpoint.py` | Split the shipped quantized checkpoint into per-part ONNX + encodings locally | downloaded checkpoint |
| `convert_parts.py` | Apply a TurboQuant profile via graph surgery, then QAIRT convert/quantize and build weight-shared HTP context binaries | QAIRT 2.48 |
| `qnn_runner/` | Dedicated on-device LLM runner on the public QNN C API; `build_android.sh` builds it | Android NDK r26c, QAIRT headers |
| `run_device_llm.py` | Prepare runner assets, push bundles over adb, run generate/score and collect JSON reports | adb, SM8850 device |
| `verify_kv_boundary.py` | Check from `qairt-dlc-info` output where the KV path is quantized: 16-bit write path from the tap (float16 for the codec, one explicit uFxp_16 grid for the uncompressed cache), codec/packed I/O without encodings, int8 only at the attention Concat | converted bundle |
| `verify_tiled_attention.py` | Check every compiled tile's KV/attention boundary, absence of full restores, and maximum codec tensor size | tiled bundle |
| `verify_rotated_attention.py` | Check effective-scale I/O, FP16 rotated attention, removal of repeated KV inverse/norm operations, and tile sizes | rotated bundle |

Run from the repo root with `PYTHONPATH=src` so the repo sources shadow any
installed `qai_hub_models` wheel. None of these tools submit AI Hub jobs.

For restore throughput, benchmark the **past-cache length**, not only the
number of newly generated tokens. An AR=1, context=1024 graph restores 1023
tokens per KV tensor:

```bash
PYTHONPATH=src python scripts/llm/turboquant/htp_codec_validation.py all \
    --rotation fwht \
    --work-dir ~/.qaihm/tmp/turboquant/p2_optimized_hub_1023 \
    --snapshot ~/.qaihm/tmp/turboquant/qwen3_1_7b_kv_snapshot.npz \
    --tokens 1023 --operations decode --head-major
```

Long codec probes repeat the recorded KV vectors to fill the requested shape;
they measure codec accuracy/throughput, not long-context model quality.
`--head-major` flag uses the actual model's `[heads, 1, tokens, dim]` I/O;
without it the standalone `[1, heads, tokens, dim]` layout can hide HTP batch
overhead. The codec normalizes its internal layout while preserving either ABI.
The `run` and `compare` stages use the built manifest, so the shape options only
need to be supplied to `build` (or `all`). Always reconvert the full model into
a new bundle after changing the codec lowering; an existing context binary
does not pick up Python source changes.

For format-2 scalar validation, add `--profile k4_v4_scaled --range-scale 0.5`
to `build`/`all`. The range option only scales synthetic inputs, not real KV:
effective scale can exceed FP16 even when the original norm fits. This is a
bounded-domain probe, not full-FP16-domain coverage. `compare` selects the
oracle from the built manifest and checks its config hash. The current scale
encoder exceeds the existing 0.2% norm-only tolerance (maximum 0.2564%);
that gate remains failed and its tolerance is not relaxed. See design §14.2.

## Tiled KV restore and attention

The opt-in `--attention-tile 256` conversion replaces full-cache restores with
two passes: K tiles feed QK, then V tiles feed partial AV products. The original
global masked softmax is retained. Only attention scores are concatenated;
restored K/V tiles are never concatenated into a full FP16 cache. The compiler
still controls allocation and scheduling: this is graph tiling, not a custom
single-kernel FlashAttention implementation or a guarantee of reduced peak HTP
memory/latency. Partial AV products introduce extra rounding, so validate PPL.

```bash
PYTHONPATH=src python scripts/llm/turboquant/convert_parts.py \
    --split-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_w4a16_split \
    --out ~/.qaihm/tmp/turboquant/qwen3_1_7b_k4_v4_tiled256_int8_cl1024 \
    --context-length 1024 --profile k4_v4 --rotation fwht --attention-tile 256
PYTHONPATH=src python scripts/llm/turboquant/verify_tiled_attention.py \
    --bundle ~/.qaihm/tmp/turboquant/qwen3_1_7b_k4_v4_tiled256_int8_cl1024 \
    --report ~/.qaihm/tmp/turboquant/reports/boundary_k4_v4_tiled256.json
PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py push \
    --bundle-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_k4_v4_tiled256_int8_cl1024 \
    --name k4_v4_tiled256_cl1024
PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py run \
    --name k4_v4_tiled256_cl1024 \
    --assets ~/.qaihm/tmp/turboquant/device_assets_cl1024 \
    --mode generate --n-gen 128 --sessions 1 \
    --report ~/.qaihm/tmp/turboquant/reports/perf_k4_v4_tiled256_once.json
```

The tiling pass fails closed for unrecognized attention patterns. It preserves
packed cache I/O, encoder, weights and the global softmax; tiled KV Concat and
QK/AV products reuse the corresponding calibrated quantization grids. Tile
size is a compilation option, not a new cache format. The default remains the
untiled codec. Performance comparisons from this change onward use **one
generation session per configuration**, separate from correctness/profiling.

The S26/CL1024 single-run result was **10.81 tok/s**, versus 10.87 for the
untiled codec and 42.25 for the int8 baseline: tiling did not improve decode
speed. The largest FP16 codec tensor fell from about 2 MiB to 512 KiB and the
compiler reported zero decode spill/fill buffer, but host RSS increased.
See design §13 for the full results and memory/measurement limitations.

## Rotated attention, precomputed scales and context buckets

The separate `k4_v4_scaled` profile uses cache **format 2**: packed indices are
unchanged, but `tq_*_scale_{in,out}` holds `norm / ||centroids||` instead of the
original norm. It cannot share cache state with format 1. Encode computes the
correction once; decode does no norm reduction, square root or division.

`--rotated-attention` rotates Q and the current K/V, consumes the cached K/V in
the rotated domain, and inverse-rotates the accumulated attention output once.
The rotated products use FP16, not the original-domain int8 calibration grids.
The global masked softmax and calibrated original-domain output grids remain.
This is an opt-in graph rewrite, not a native fused HTP kernel.

```bash
PYTHONPATH=src python scripts/llm/turboquant/convert_parts.py \
    --split-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_w4a16_split \
    --out ~/.qaihm/tmp/turboquant/qwen3_1_7b_rotated_scaled_buckets \
    --context-length 1024 --context-buckets 128 256 512 1024 \
    --profile k4_v4_scaled --rotation fwht --attention-tile 256 --rotated-attention \
    --no-native-decoder
PYTHONPATH=src python scripts/llm/turboquant/verify_rotated_attention.py \
    --bundle ~/.qaihm/tmp/turboquant/qwen3_1_7b_rotated_scaled_buckets \
    --report ~/.qaihm/tmp/turboquant/reports/boundary_rotated_scaled_buckets.json
bash scripts/llm/turboquant/qnn_runner/build_android.sh \
    ~/.qaihm/tmp/turboquant/qnn_runner/rotated-buckets
PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py push \
    --bundle-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_rotated_scaled_buckets \
    --runner ~/.qaihm/tmp/turboquant/qnn_runner/rotated-buckets/qnn-llm-runner \
    --name rotated_scaled_buckets
PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py run \
    --name rotated_scaled_buckets \
    --assets ~/.qaihm/tmp/turboquant/device_assets_cl1024 \
    --mode generate --n-gen 128 --sessions 1 \
    --report ~/.qaihm/tmp/turboquant/reports/perf_rotated_scaled_buckets_once.json
```

`push` stages a runtime manifest so `run` selects the compiled buckets without
extra flags. The runner chooses the smallest context `C` with
`cached_tokens <= C - AR`; AR128 skips C128. It retains the full-capacity host
store, copies only valid KV right-aligned into the selected graph, and preserves
absolute RoPE positions across bucket switches. Padded slots inside the chosen
bucket still execute; this is bounded padding, not arbitrary-length execution.
Each recorded step includes `graph_context` to verify the actual selection.

To measure the same binary without bucketing, run a separate one-session
configuration with `--context-buckets 1024` and a different report path.
Multiple graph variants increase compilation time, graph metadata and resident
I/O buffers. Validate PPL and transitions as well as throughput before enabling.

S26 single-run decode was **44.60 tok/s** for the 35-token prompt, versus 10.51
for the previous tiled codec and 32.24 for the fixed-CL1024 int16 KV baseline.
With a 897-token prompt it was **13.88 versus 34.91 tok/s**: the full-context
target is not met. No bucketed int16 baseline was tested. PPL was 20.388335
(baseline 20.108306), and the standalone scale-encoder tolerance gate still
fails. The profile remains opt-in. See design §14 for the one-run comparison,
memory overhead, correctness checks and remaining packed-KV restore bottleneck.

## Native unpack + LUT decoder (HTP V81, default for k4_v4_scaled)

Native decoding replaces only each rotated KV tile's decoder with
the `TurboQuantNative::Decode4` QHPI op. Its HVX kernel unpacks MSB-first nibbles,
uses a 16-entry halfword LUT, and multiplies by the stored FP16 effective scale.
The format-2 cache ABI, encoder, query/output rotations, QK/softmax/AV operations,
and calibrated attention grids are unchanged. This is **not** a fused attention
kernel. Use `--native-decoder-package` to override the built package location,
or `--no-native-decoder` to explicitly use the original graph decoder.

The x86 library provides offline preparation and a scalar implementation; the
V81 library executes vector instructions on HTP. Neither SDK source nor SDK
libraries are vendored. QHPI from QAIRT 2.48 and a V81-capable Hexagon compiler
are required. `--test-hvx` additionally uses the SDK's host libnative simulator.

```bash
python scripts/llm/turboquant/native_decoder/build.py \
    --out ~/.qaihm/tmp/turboquant/native_decoder_v81 --test-hvx
PYTHONPATH=src python scripts/llm/turboquant/validate_native_decoder.py all \
    --package ~/.qaihm/tmp/turboquant/native_decoder_v81 \
    --work-dir ~/.qaihm/tmp/turboquant/native_decoder_validation \
    --tokens 1 3 127 128 255 256 1023
PYTHONPATH=src python scripts/llm/turboquant/convert_parts.py \
    --split-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_w4a16_split \
    --out ~/.qaihm/tmp/turboquant/qwen3_1_7b_native_lut_buckets \
    --context-length 1024 --context-buckets 128 256 512 1024 \
    --profile k4_v4_scaled --rotation fwht --attention-tile 256 --rotated-attention \
    --native-decoder-package ~/.qaihm/tmp/turboquant/native_decoder_v81
PYTHONPATH=src python scripts/llm/turboquant/verify_rotated_attention.py \
    --bundle ~/.qaihm/tmp/turboquant/qwen3_1_7b_native_lut_buckets \
    --report ~/.qaihm/tmp/turboquant/reports/boundary_native_lut.json
bash scripts/llm/turboquant/qnn_runner/build_android.sh \
    ~/.qaihm/tmp/turboquant/qnn_runner/native-decoder
PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py push \
    --bundle-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_native_lut_buckets \
    --runner ~/.qaihm/tmp/turboquant/qnn_runner/native-decoder/qnn-llm-runner \
    --name native_lut_buckets
```

`push` includes the DSP library and its checksum in the bundle. `run` verifies
that checksum and registers the package **before** loading context binaries.
Normal non-native bundles require no package registration. Use the preceding
generation commands with the new bundle name and **new report paths**; measure
35-token and 897-token prompts with 128 generated tokens once each. Functional,
quality, and detailed-profile runs must be reported separately.

For memory-bounded builds, `--skip-context` separates DLC conversion from
`--context-only` finalization, which checks metadata before reusing DLCs.
`assemble_native_bundle.py` can combine separately finalized parts with a base
bundle (including a partial base), but refuses incomplete/mismatched part sets
or reuse of any old part that contains graph-decoder KV I/O.

The native LUT intentionally rounds centroids and products to FP16. An ONNX
function used only in tests (`with_reference_decoder`) supplies an independent
oracle; it is never embedded in deployed models. Native decoder correctness
does not resolve the existing format-2 **encoder** scale tolerance failure
described in design §14.2. End-to-end PPL must still be checked.

On S26/SM8850, the one-run **897-token prompt + 128-token generation** result
was **38.15 tok/s**, versus 13.89 for the unchanged graph decoder and 34.76 for
the int16 KV baseline: 2.746x over graph and 9.77% over int16. All decode steps
used C1024. Native TTFT was 567.2ms (graph 718.4ms, int16 318.2ms); prefill is
still slower than int16. Native's QNN-call time alone is also slightly slower
than int16; the end-to-end win includes smaller host KV I/O.

Host KV remains 28.875MiB. PPL over four windows was **20.498138**, versus
20.388335 for graph and 20.108306 for int16 (unchanged binaries' prior PPL).
This is not numerically identical to the compiled graph path. The 35-token
prompt's native decode was 54.43 tok/s, but the int16 control was not bucketed.
Tests, EOS/reset/bucket transitions, the CL1024 boundary and the isolated HTP
decoder passed; the inherited encoder tolerance failure remains unresolved.
See design §15 and `reports/comparison_native_lut_buckets.json` for all metrics,
source reports, single-run limitations and the distinction between decoder
optimization and fused attention.

## Dense rotation + Native decoder (default rotation)

Dense QR and Native decoding are selected by default for `k4_v4_scaled`:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/llm/turboquant/convert_parts.py \
    --split-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_w4a16_split \
    --out ~/.qaihm/tmp/turboquant/qwen3_1_7b_dense_native_new \
    --context-length 1024 --context-buckets 128 256 512 1024 \
    --profile k4_v4_scaled
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/llm/turboquant/verify_rotated_attention.py \
    --bundle ~/.qaihm/tmp/turboquant/qwen3_1_7b_dense_native_new \
    --report ~/.qaihm/tmp/turboquant/reports/boundary_dense_native_new.json
```

The verifier checks actual rotation constants as well as FP16 attention, native
decoders and tile sizes. The previous FWHT export also used dense MatMul ops;
this change replaces the matrix values/distribution, not an O(d log d) device
butterfly with O(d²) work. It is still QJL-off PolarQuant, not the full
QJL-enabled TurboQuant algorithm. See design §16 for the three-way comparison.

On S26, the one-run 897+128 result was **38.08 tok/s** for dense+Native,
**39.11** for unchanged FWHT+Native, and **35.66** for int16 KV. Dense's TTFT
was 563.26ms (FWHT 522.93ms, int16 303.25ms). Host KV remains 28.875MiB,
versus int16's 112MiB. Fresh four-window PPL was **20.607753 / 20.498138 /
20.108306**, respectively. This change did not improve measured quality or
throughput over FWHT; no statistical speed difference is claimed from one run.

All 21 attention graphs passed the audit and changed only rotation constants
in the source graphs. Reset, EOS, bucket switching and the CL1024 boundary
passed. The dense encoder's effective-scale gate still fails: maximum error
0.2752% versus the unchanged 0.2% tolerance. It is not hidden by the successful
Native decoder checks. See `reports/comparison_dense_native_buckets.json` and
`summarize_rotation_results.py` for the complete metrics and source reports.
