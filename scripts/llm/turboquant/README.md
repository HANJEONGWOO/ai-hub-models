<!--
Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
SPDX-License-Identifier: BSD-3-Clause
-->

# TurboQuant KV-cache tools

Developer tools for the opt-in TurboQuant KV-cache codec in
`qai_hub_models.models.templates.llm.turboquant`. The ABI, numeric tolerances
and current results are in
[`tutorials/llm/turboquant_design.md`](../../../tutorials/llm/turboquant_design.md).

## Structured K8 versus frozen Lloyd-Max K8 (opt-in, 2026-10-13)

Branch `exp/structured-k8-codebook` starts at `exp/hardware-aligned-mixed-kv`
(`c022fecee`). Existing profiles, hashes, binaries and earlier reports are
preserved. No automatic commit or default promotion is performed.

Compare `mixed_k6_v2_uniform` with `mixed_k6_v2_uniform_structured_k8`.
Both use **K8 on odd zero-based layers 1,3,...,27**, K4 on the other 14 layers
and **V2 on all 28 layers**. Average index payload is K6/V2 = eight bits;
FP16 scales are additional. Each allocates 28 MiB of indices + 0.875 MiB of
scales = **28.875 MiB** at CL1024. No placement search, retraining,
recalibration or extra token metadata. Dense K42/V542, current/past compression,
norm correction, W4A16 weights, tile256 FP16-input attention, GQA and V/AV
remain unchanged.

### Structured rule and offline fit

For normalized/rotated K, the logical code has sign1 + region1 + local6 bits:

```text
m = abs(y)
r = (m > region_boundary)
k = clip(floor((m - offset[r]) * reciprocal_step[r] + 0.5), 0, 63)
j = 64*r + k
byte = (y >= 0) ? 128+j : 127-j
centroid_magnitude = FP16(FP16(k * step[r]) + offset[r])
```

Negative magnitude fields are complemented to retain the existing ascending
symmetric LUT ABI. Zero maps to code128. The index is monotone and saturates
to [0,255]; arithmetic ties follow the floor rule, not LM's lower-index tie
convention. FP16 execution may move decisions near edges and is validated
separately. UINT8 stores the index directly, without regrouping.

| Shared coefficient | Central region | Tail region |
|---|---:|---:|
| Offset | 0.00089263916015625 | 0.1268310546875 |
| Spacing | 0.001956939697265625 | 0.00376129150390625 |

Region boundary: **0.125518798828125** before HTP FP16 rounding.
`optimize_structured_k8.py` fits four shared coefficients using analytic
integrated scalar MSE under **N(0,1/128)**, nine fixed initial points, and
positive-spacing/ordering constraints. The region edge is the midpoint of
the last central and first tail centroid. No LM centroid fitting, actual-KV
tuning, attention loss, PPL selection or global-optimum claim.
`optimization.json` records all initializations, optimizer statuses and hashes.
The deployed constants are frozen in `structured_k8.py`; experiment preparation
rejects a mismatching offline fit.

Encoder scalar-index and centroid selection use region/arithmetic operations
instead of an expanded boundary tree. Norm correction uses the selected
centroid vector and still stores one FP16 effective scale. The **same existing
Native HVX LUT decoder binary** serves both groups; only the static K8 LUT
contents differ. This isolates quantizer changes, not an arithmetic HVX
decoder, bit-plane attention or CPU inference fallback.

Standalone K8 encoder ONNX nodes fall from **837 to 76**: scalar indexing
547 to 23 and centroid/norm-scale 268 to 31, while the single Dense MatMul
and normalization path remain. Node counts are structural, not latency ratios.

### Accuracy and numerical validation

Offline errors use two frozen pre-existing validation documents from
`k_rotation_quality_20261006`, all 28 layers, 32 fixed queries/window and
causal scores only. Reference: the same W4A16 model's FP16-KV captures, not
full-FP16 weights. Q includes attention scaling. The emulator uses FP16
operation boundaries and FP32 reductions, **not HTP bit-exact execution**.
These documents never influence coefficient selection.

| Offline metric | Frozen LM K8 | Structured K8 |
|---|---:|---:|
| K reconstruction MSE, K8 layers only | 0.002105271 | 0.000921581 |
| QK score MSE, K8 layers only | 0.006115046 | 0.002429114 |
| K reconstruction MSE, all 28 layers | 0.125806955 | 0.125215109 |
| QK score MSE, all 28 layers | 0.291169141 | 0.289326175 |
| Analytic Gaussian nearest-centroid scalar MSE | 1.305468e-6 | 4.793545e-7 |

K reconstruction includes inverse-rotation FP16 rounding. QK compares raw
scaled scores before masking/softmax and existing affine score grids. JSON
also retains relative MSE, rotated-coordinate/reference variants and each
layer/document. All-layer metrics include unchanged K4 layers. The Gaussian
row evaluates nearest frozen centroids, **not** the complete arithmetic,
FP16, norm-corrected codec. Structured's ideal pre-rounding fit is 4.791253e-7.
The existing LM table uses only **100 Lloyd iterations** and is not assumed
converged: this comparison does **not** establish superiority over an optimal
unrestricted LM codebook.

CPU regression: **643 passed**, with two existing SWIG deprecation warnings.
Tests include boundary neighbors, zero/sign symmetry, saturation, exact
arithmetic/LUT centroid agreement, legacy hashes, memory budget, K4/V isolation,
GQA/current/past attention, packing, reset and incompatible-cache rejection.
Both Native K8 LUTs pass HTP probes at 1/3/256 tokens, bit-exact against the
independent FP16 LUT-times-scale reference. All encoder index mismatches are
explained by the unchanged boundary tolerance; all encoder ops execute on HTP.

| HTP max scale-relative error | AR1 | AR128 |
|---|---:|---:|
| LM K8 | 0.1326% (pass) | 0.2570% (fail) |
| Structured K8 | 0.1106% (pass) | 0.1726% (pass) |
| Unchanged V4 probe | 0.1653% (pass) | 0.2477% (fail) |

The **0.2% scale tolerance is unchanged**. V4 is the standalone probe's
companion, not the mixed full model's V2 setting. Failures are retained, not
waived by PPL/decoder correctness. Initial `encoder_structured/` preflight
used an LM-only nearest-centroid comparator and stopped **before device
execution**. It is preserved; the corrected comparator checks arithmetic
decision edges with unchanged tolerances. `encoder_structured_checked/` is
the sole executed structured encoder probe.

### Isolated HTP work, not end-to-end speed

Same captured layer27 K, eight heads x 128 tokens x 128 coordinates:

| Diagnostic | LM K8 | Structured K8 |
|---|---:|---:|
| Scalar named-op cycles | 16,264,853 | 815,868 |
| Centroid/norm-scale named-op cycles | 12,872,992 | 777,423 |
| Packing/output-cast named-op cycles | 40,979 | 248,925 |
| SDK encoder accelerator execute (us) | 2,408 | 611 |
| Native Decode8 cycles, 256 tokens | 592,816 | 581,015 |

Both store direct bytes. The packing row includes casts and compiler
ownership/fusion attribution, not additional bit regrouping. Named-op counters
are **not** additive wall latency or physical busy intervals. Native uses
qnn-net-run's default power policy; encoder/full-model policies differ. The
same decoder kernel means its small timing difference is not evidence of a
new decoder speedup. Physical optrace is a separate diagnostic, not another
performance sample.

### Full-model results and reproduction

Conditions: Qwen3-1.7B, S26 Ultra / SM8850 V81, QAIRT 2.48.0.260626, CL1024, HTP burst;
one unprofiled short35+128 and one long897+128 session per group; four legacy
WikiText windows scored once, aggregate `exp(total_NLL/4092)`. No timing
dispersion estimate or independent-heldout generalization claim. LM was
measured first and Structured after its build; no interleaved repeats or
thermal-equivalence claim.

| Full-model metric | Frozen LM K8 | Structured K8 |
|---|---:|---:|
| Mean NLL, 4092 scored tokens | 3.177364372 | 3.161762996 |
| PPL, same four WikiText windows | 23.983459 | 23.612187 |
| Short35 TTFT (ms) | 98.014635 | 64.918489 |
| Short35 prefill (tok/s) | 360.649493 | 543.416300 |
| Short35 decode (tok/s) | 33.524392 | 36.379635 |
| Long897 TTFT (ms) | 809.351198 | 551.787239 |
| Long897 prefill (tok/s) | 1109.644960 | 1628.566330 |
| Long897 decode (tok/s) | 32.782912 | 35.569449 |
| KV cache allocation (MiB) | 28.875000 | 28.875000 |
| I/O buffers (MiB) | 96.929203 | 96.929203 |
| Long897 ending VmRSS (MiB) | 154.664063 | 151.484375 |
| Long897 process VmHWM (MiB) | 605.996094 | 606.226563 |
| Four context binaries (MiB) | 1635.695313 | 1621.328125 |
| Long897 mean decode KV prepare (ms/token) | 1.153872 | 1.153801 |
| Long897 mean decode QNN execute (ms/token) | 28.744591 | 26.361148 |

Cache size is actual runner allocation (30,277,632 bytes), not compressed
model weight size or total NPU/OS memory. VmRSS/VmHWM are host-process
observations; they do not include every device allocation. Graph audits pass
for identical I/O, all non-K8-quantizer nodes and shared constants; K4, V,
rotation, MatMul and attention paths remain fixed. Both cache-reset checks
pass.

| WikiText window | LM PPL | Structured PPL | Structured minus LM mean NLL |
|---|---:|---:|---:|
| w0 | 12.935116 | 12.735461 | -0.015556 |
| w1 | 32.120600 | 30.832466 | -0.040929 |
| w2 | 23.846844 | 24.072429 | +0.009415 |
| w3 | 33.393533 | 32.885319 | -0.015336 |

In this single-run comparison, Long897 TTFT falls **31.82%**, prefill rises
**46.76%**, and decode rises **8.50%**, with identical KV/I/O storage. PPL
falls **1.55%**, but one of four windows worsens; this is not a generalization
or significance claim. Improvement is relative to the existing frozen,
100-iteration LM implementation, not an optimally converged LM quantizer.
The unchanged HVX decoder and KV preparation leave substantial decode work
untouched. Do not extrapolate the encoder probe's speed ratio to the whole
model, or interpret profile cycle percentages as TTFT reductions.

### Full-model physical optrace

One separate 897+128 diagnostic per group, **44 traces each**: eight AR128
prefill calls across four parts, then decode steps 0/63/126 across four parts.
The first prefill call has one valid token; the other seven have 128. Generated
IDs match each group's unprofiled run. Units below are **million physical
HVX/HMX busy cycles**, summed across workers, layers and calls, not ms or wall
latency. DMA/waits and physical interval unions remain separate in raw reports.

| Stage, all K/V layers | Prefill LM | Prefill Structured | Decode126 LM | Decode126 Structured |
|---|---:|---:|---:|---:|
| Normalize | 237.763 | 226.977 | 1.407 | 1.369 |
| Dense rotation | 1.380 | 1.510 | 0.0175 | 0.0172 |
| Scalar index | 1891.748 | 371.655 | 12.755 | 1.992 |
| Centroid/norm-scale correction | 1624.942 | 363.636 | 8.277 | 1.805 |
| Packing/output cast | 260.123 | 286.421 | 4.055 | 3.829 |
| Native current-KV decode | 85.973 | 81.099 | 4.989 | 4.990 |
| Native past-KV decode | 491.510 | 489.890 | 65.248 | 66.067 |

K8-only attribution, sum over the fixed 14 odd layers, same cycle units:

| K8 stage | Prefill LM | Prefill Structured | Decode126 LM | Decode126 Structured |
|---|---:|---:|---:|---:|
| Scalar index | 1615.501 | 96.051 | 11.378 | 0.629 |
| Centroid/norm-scale correction | 1354.557 | 87.253 | 7.011 | 0.520 |
| Packing/output cast | 2.585 | 33.145 | 0.0036 | 0.2038 |
| Native past-K decode | 212.546 | 206.192 | 26.287 | 27.043 |

The encoder saving is from **index and centroid selection**, not eliminating
packing: both K8 variants already store direct UINT8. K8 prefill scalar work
falls about 94.1% and correction work about 93.6%. Across all K/V layers these
reductions are 80.4% and 77.6%. Packing/output-cast attribution increases
(10.1% across all K/V prefill), reflecting real cast/format work and compiler
fusion/ownership; it is retained rather than reported as zero or attributed
to new bit packing. The arithmetic path's extra attributed cast work does
not offset its much larger index/correction savings in this experiment.

Dense rotation still executes on HMX; Native decoding executes on HVX.
Physical stage counters include compiler-fused kernels, not just standalone
named-op totals. Past-KV Native decoding is essentially unchanged across
the full model and dominates the remaining codec work at decode126, which
helps explain why decode improves less than prefill. Different downstream
activation values and single-session variability can also affect counters;
the isolated same-input probes above complement, not replace, this trace.
No cycle-percentage reduction is used as a latency-percentage estimate.

Artifacts: `/mnt/d/ai-hub-models/binaries/turboquant/structured_k8_20261013`.
Reproduce into a **fresh** root; repeated measurements are refused:

```bash
export PYTHONPATH=src
TQ_SK8_ROOT=/mnt/d/ai-hub-models/binaries/turboquant/structured_k8_NEW
venv/bin/python scripts/llm/turboquant/optimize_structured_k8.py --out "$TQ_SK8_ROOT/codebook/optimization.json"
venv/bin/python scripts/llm/turboquant/benchmark_structured_k8_once.py prepare --root "$TQ_SK8_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_structured_k8_once.py probe --root "$TQ_SK8_ROOT"
venv/bin/python scripts/llm/turboquant/evaluate_structured_k8_quality.py --root "$TQ_SK8_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_structured_k8_once.py build --root "$TQ_SK8_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_structured_k8_once.py evaluate --root "$TQ_SK8_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_structured_k8_once.py summarize --root "$TQ_SK8_ROOT"
TQ_SK8_TRACE="$TQ_SK8_ROOT/structured_k8_NEW_optrace"
for stage in freeze build measure render summarize; do
  venv/bin/python scripts/llm/turboquant/profile_structured_k8_once.py "$stage" \
    --experiment "$TQ_SK8_ROOT" --root "$TQ_SK8_TRACE" || break
done
venv/bin/python scripts/llm/turboquant/summarize_mixed_profiles.py \
  --experiment "$TQ_SK8_ROOT" --root "$TQ_SK8_TRACE" \
  --groups mixed_k6_v2_uniform mixed_k6_v2_uniform_structured_k8
```

The harness reuses frozen LM model binaries and `native_symmetric8` from the
previous mixed experiment, **not its performance measurements**. Source
audits check weights/calibration, all V/K4/rotation/attention nodes; compiled
audits check FP16 boundaries and cache I/O. Config hashes include coefficients,
LUT and index-rule identity, rejecting mixed LM/structured caches.

Key artifacts under the root: `protocol.json`, `codebook/optimization.json`,
`offline_quality.json`, `isolated_probes.json`, `comparison.json`/`.csv`,
`reports/k8_only_audit.json`, `tests/cpu_regression_complete.xml` and
`final_verification.json`. Physical comparisons and detailed per-stage/per-bit
CSVs are under `structured_k8_20261013_optrace/`. Previous result files,
failed preflight artifacts and numerical gate failures are preserved.

Implementation: `structured_k8.py`, `config.py`, `reference.py`, `export.py`,
`native_decoder.py`, `numerics.py`; experiment/quality/profile/probe-summary
scripts named `*structured_k8*`; opt-in HTP probe arguments and extended
codec/current-KV tests. Historical experiment sections below are unchanged.

## Hardware-aligned mixed-bit experiment (opt-in)

`mixed_k5_v3_uniform`, `mixed_k6_v2_uniform` and their `*_mse` variants
are separate Qwen3-1.7B/28-layer profiles. The default `k4_v4_scaled`, legacy
profiles and their configuration hashes are unchanged. The old `k8_v3` name
still means affine INT8 K; **`k8_v4_scaled` is the new Lloyd–Max K8 probe**.

The pinned reference generator now freezes 7/8-bit tables as well as the
unchanged 2–6-bit tables. It retains the reference's 100 Gaussian Lloyd
iterations; this is not a claim of convergence or a globally optimal codebook.
K8 uses 256 nonuniform centroids and eight exact tree
decisions, stores the index directly as one UINT8 byte (no Slice/Shift/Concat
packing), and retains the existing norm correction/effective FP16 scale.
`Decode8` reads those bytes directly and uses banked HVX LUT lookup followed by
one FP16 product. Bitwise-symmetric LM tables use the positive half table and
restore the FP16 sign; arbitrary tables retain the full HVX LUT path. It is not
affine INT8, bit-plane attention or a CPU fallback.
The expanded scalar tree and encoder centroid lookup can still be expensive;
removing packing does not by itself imply faster inference.
V2 deliberately reuses the existing tight-stream packer (including its graph
Slice/arithmetic/Concat operations); only K8 bypasses packing. These profiles
test byte-aligned K allocation, not a claim that every encoder is fully fused
or that V2 packing has been newly optimized.

| Profile | K allocation | V allocation | Payload budget | KV including FP16 scales, CL1024 |
|---|---|---|---:|---:|
| Mixed K5/V3 | 21 K4 + 7 K8 layers | 14 V2 + 14 V4 layers | average K+V = 8 bits | 28.875 MiB |
| Mixed K6/V2 | 14 K4 + 14 K8 layers | 28 V2 layers | average K+V = 8 bits | 28.875 MiB |

Uniform placement uses K8 on every fourth/second layer (zero-based layer IDs
3,7,… / 1,3,…) and V4 on odd layers for Mixed K5/V3. MSE placement uses actual
captured Q/K/V and FP16-KV Attention outputs from the same W4A16 model, with
fixed Dense K seed42/V seed542. Four source training documents select the plan;
two separate source validation documents check it. The legacy four WikiText
PPL windows are never used to select layers. An exact quota-constrained dynamic
program minimizes the **sum of local absolute Attention-output MSE** over the
four possible K4/K8 × V2/V4 choices. This is not end-to-end NLL optimization;
absolute MSE may favor late high-magnitude layers.

Layer plans, codebook hashes and selection policy are part of the immutable
configuration hash. Cache loading rejects different plans. MSE profiles require
their hash-checked `--layer-plan` JSON; they cannot silently fall back to uniform
placement. Every layer's packed I/O width is exported separately. No per-token
bit metadata is added. Native packages containing the new ops require new
compiled contexts; do not replace libraries inside existing baseline bundles.

Reproduce with a **fresh** artifact root (no automatic commits):

```bash
TQ_MIXED_ROOT=/mnt/d/ai-hub-models/binaries/turboquant/hardware_aligned_mixed_kv_NEW
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/native_decoder/build.py \
  --out "$TQ_MIXED_ROOT/native" --test-hvx
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/select_mixed_layers.py \
  --captures /mnt/d/ai-hub-models/binaries/turboquant/k_rotation_quality_20261006 \
  --out "$TQ_MIXED_ROOT/selection"
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/benchmark_mixed_once.py prepare --root "$TQ_MIXED_ROOT"
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/benchmark_mixed_once.py build --root "$TQ_MIXED_ROOT"
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/benchmark_mixed_once.py evaluate --root "$TQ_MIXED_ROOT"
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/benchmark_mixed_once.py summarize --root "$TQ_MIXED_ROOT"
```

Performance is **one unprofiled session per configuration/condition**:
short35+128 and long897+128. No run-to-run mean/std or confidence interval is
reported. Each of the four PPL windows is scored once, with aggregate
`exp(sum(NLL_sum)/4092)`. Cache-reset checks and HTP probes/optrace are separate
diagnostics, not extra throughput samples. The comparison contains seven rows:
three old profiles plus two mixed budgets × two placement policies.

Optional separate stage profiling after the unprofiled runs:

```bash
for stage in freeze build measure render summarize; do
  PYTHONPATH=src venv/bin/python scripts/llm/turboquant/profile_mixed_once.py "$stage" \
    --experiment "$TQ_MIXED_ROOT" --root "$TQ_MIXED_ROOT/optrace" || break
done
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/summarize_mixed_profiles.py \
  --experiment "$TQ_MIXED_ROOT" --root "$TQ_MIXED_ROOT/optrace"
```

Artifacts for the current experiment are under
`/mnt/d/ai-hub-models/binaries/turboquant/hardware_aligned_mixed_kv_20261010/`.
The initial Native package and diagnostics remain in `native/` and
`isolated_probes.json`. Before any mixed full-model measurement,
`native_runtime_amendment.json` froze a header-only symmetric-LUT runtime in
`native_symmetric8/`; package registration and the x86 prepare library match.
`isolated_probes_final.json` records its separate correctness diagnostic. Old
baseline packages remain unchanged. A fresh reproduction with the commands
above builds this final kernel directly, without an amendment.
Full-model single-run performance/quality results are in `comparison.json` and
`comparison.csv`; per-window scores, reset checks and source/compiled graph
audits are in `reports/`. Binary/config/input hashes are recorded there and in
`protocol.json`. Stage optrace analysis is separate from these measurements.

### Validation

- Existing/default configuration hashes remain unchanged; Python regression:
  628 tests passed.
- HVX host oracle: 840 shapes/alignment/tail cases across 2–8 bits, including
  guard-page cases, passed.
- K8 Native HTP decoder: 1/3/256 tokens, all outputs bit-exact to independent
  FP16 LUT × FP16 scale reference.
- K8 encoder: no unexplained index mismatches, all graph ops accounted for on
  HTP. AR1 passed. AR128 **fails the unchanged 0.2% scale-relative-error gate**
  (maximum 0.256993%); the accompanying unchanged V4 path also fails (0.247664%).
  This numerical limitation is tracked separately, not hidden by HTP success.
- All seven full-model source/compiled audits passed: unchanged non-codec
  weights/calibration, matched K/Query/V rotations, FP16 attention/native
  boundaries, correct layer-specific UINT8 I/O and compressed current/past KV.
  Cache reset was deterministic. The three reused baselines reproduced their
  previous four-window PPL exactly.

### Single-run comparison: 2026-10-10, S26 Ultra / SM8850 V81

Qwen3-1.7B W4A16, CL1024, AR128 prefill, AR1 decode, tile256, HTP burst power.
Model load is excluded from TTFT. Each row/condition has **one** unprofiled
session; PPL uses the same legacy four windows / 4092 targets. Means/std across
runs cannot be estimated. Existing baselines ran first, followed by the four
mixed variants after building; this is not an interleaved/thermal-controlled
repetition study. Do not treat small speed differences as statistically proven.

Long condition: 897 prompt tokens + 128 generated tokens.

| Configuration | PPL ↓ | TTFT ms ↓ | Prefill tok/s ↑ | Decode tok/s ↑ |
|---|---:|---:|---:|---:|
| K4/V4 | 25.661 | 564.0 | 1593.19 | 37.72 |
| K5/V3 | 22.470 | 650.6 | 1380.86 | 35.76 |
| K6/V2 | 19.845 | 671.3 | 1338.16 | 34.48 |
| Mixed K5/V3 uniform | 25.369 | 688.5 | 1304.76 | 35.01 |
| Mixed K6/V2 uniform | 23.983 | 812.2 | 1105.76 | 32.89 |
| Mixed K5/V3 MSE | 27.600 | 669.6 | 1341.63 | 34.99 |
| Mixed K6/V2 MSE | 27.242 | 755.2 | 1189.32 | 32.85 |

Short condition: 35 prompt tokens + 128 generated tokens.

| Configuration | TTFT ms ↓ | Prefill tok/s ↑ | Decode tok/s ↑ |
|---|---:|---:|---:|
| K4/V4 | 65.80 | 535.78 | 38.56 |
| K5/V3 | 77.58 | 454.13 | 36.89 |
| K6/V2 | 81.20 | 433.92 | 35.27 |
| Mixed K5/V3 uniform | 86.15 | 410.97 | 35.71 |
| Mixed K6/V2 uniform | 101.92 | 346.39 | 33.52 |
| Mixed K5/V3 MSE | 80.96 | 434.90 | 35.92 |
| Mixed K6/V2 MSE | 95.35 | 370.81 | 33.60 |

**Actual allocated host KV is 30,277,632 bytes = 28.875 MiB in every row**:
28 MiB indices + 0.875 MiB FP16 scales. The eight-bit budget describes K+V
index payload, not scales. Runner I/O buffers are also identical at
96.929203 MiB. Long-session ending VmRSS is 151.18–154.99 MiB and process
VmHWM is 605.42–606.15 MiB. These are host/process metrics, not a measurement
of every internal HTP allocation. Exact per-profile memory and binary hashes
are in the JSON/CSV. Context binary totals are 1623.312/1625.262/1627.500 MiB
for K4/V4, K5/V3, K6/V2 and 1630.988/1635.695/1629.219/1634.949 MiB for the
four mixed rows in the table's order.

### Placement quality: local MSE did not predict end-to-end PPL

| Allocation | Train Attention MSE | Separate validation Attention MSE | Test PPL |
|---|---:|---:|---:|
| Mixed K5/V3 uniform | 0.296910 | 0.305943 | 25.369 |
| Mixed K5/V3 MSE | 0.052217 | 0.051684 | 27.600 |
| Mixed K6/V2 uniform | 0.440162 | 0.449656 | 23.983 |
| Mixed K6/V2 MSE | 0.392957 | 0.398296 | 27.242 |

The MSE-selected K8 layers are `20,22,23,24,25,26,27` for Mixed K5/V3,
and `10,15,16,...,27` for Mixed K6/V2 (zero-based). MSE Mixed K5/V3 puts V4
on layers 14–27; MSE Mixed K6/V2 keeps V2 everywhere. Plans were frozen using
training MSE before validation/PPL, and were not changed after seeing results.

Absolute local output MSE can favor high-magnitude late layers. Captured
FP16-KV inputs do not include the accumulated upstream errors of the fully
compressed model, and this objective is not NLL. Those are plausible reasons
for the disagreement, not individually established causes without ablations.
Local validation MSE and PPL use different documents, so this comparison does
not isolate objective mismatch from data-distribution/generalization effects.
The CPU hard-forward implementation reproduces the earlier seed42 K4/V4
validation MSE (0.1683944842), which is a useful emulator regression check but
not a proof of full-model/HTP bit-exactness.

### Isolated codec diagnostics (not LLM latency)

Same captured layer27 K input, 8 KV heads × 128 tokens × 128 coordinates.
Encoder columns are **millions of named-op cycles**, not physical interval
latencies; fused work can be attributed to its owning node. No average of the
three distinct layer0/layer27/range inputs is presented.

| K width | Scalar index Mcycles | Packing Mcycles | Centroid lookup + norm/scale Mcycles | Native decoder cycles, 256 tokens |
|---|---:|---:|---:|---:|
| 4 | 1.396 | 0.340 | 1.346 | 325,333 |
| 5 | 2.413 | 3.408 | 1.967 | 370,208 |
| 6 | 4.186 | 3.657 | 3.315 | 386,229 |
| 8 | 15.856 | 0.033 | 12.444 | 580,093 |

K8 eliminates bit regrouping but still casts/stores its UINT8 index. Eight
logical tree comparisons do **not** mean eight exported operators: selecting
their thresholds expands the HTP graph (547 index nodes versus 47 for K4),
and the encoder's centroid lookup/norm path also grows. Native Decode8 runs
HVX, not a CPU fallback, but its larger LUT is not free. Its standalone probe
uses default `qnn-net-run` power, unlike the full-model burst policy; the two
sets of timing values must not be combined. The symmetric-LUT change's
256-token diagnostic was 580,093 cycles versus 585,260 initially; small-case
probes were slower, so no robust kernel speedup is claimed from this alone.

This implementation did **not** achieve the joint goal of matching/improving
existing asymmetric PPL while reducing overall latency. Uniform mixed layouts
improve PPL over K4/V4 but are slower; both MSE layouts worsen PPL.
All mixed rows are worse in PPL and measured latency than their same-average
K5/V3 or K6/V2 comparator. Defaults remain unchanged.

### Full-model physical optrace

Seven separate diagnostic sessions produced 44 traces each (308 total). All
generated token IDs match their corresponding unprofiled reference. These
diagnostics are **not** additional throughput samples. Tables use millions of
physical HVX/HMX busy cycles summed across parallel execution lanes, **not
milliseconds or wall-latency percentages**. Source/compiled-owner attribution
and verified HMX MatMul-parent attribution are retained from the prior profiler.
Shared/fused work is not an independently timed Python algorithm function.

Prefill: all eight AR128 calls for the 897-token prompt, including the padded
first call. `Correction` includes encoder centroid lookup and norm/FP16 scale
correction; `Normalize` is the input-vector normalization stage.

| Configuration | Normalize | Scalar index | Correction | Packing | Native current | Native past |
|---|---:|---:|---:|---:|---:|---:|
| K4/V4 | 225.402 | 660.450 | 578.108 | 136.178 | 57.760 | 374.376 |
| K5/V3 | 235.047 | 719.633 | 568.249 | 1056.346 | 63.842 | 407.168 |
| K6/V2 | 229.627 | 1028.265 | 842.336 | 1111.707 | 65.933 | 441.544 |
| Mixed K5/V3 uniform | 235.589 | 1269.048 | 1104.026 | 203.327 | 73.094 | 436.340 |
| Mixed K6/V2 uniform | 238.442 | 1897.370 | 1626.086 | 260.770 | 86.067 | 490.803 |
| Mixed K5/V3 MSE | 231.816 | 1297.332 | 1128.307 | 198.256 | 72.661 | 432.067 |
| Mixed K6/V2 MSE | 234.350 | 1903.891 | 1644.928 | 258.262 | 85.085 | 493.846 |

Last decode call (`decode_126`, 1023 cached tokens), same units:

| Configuration | Normalize | Scalar index | Correction | Packing | Native current | Native past |
|---|---:|---:|---:|---:|---:|---:|
| K4/V4 | 1.420 | 2.905 | 2.534 | 1.660 | 2.384 | 51.106 |
| K5/V3 | 1.379 | 4.026 | 3.050 | 6.061 | 2.477 | 56.311 |
| K6/V2 | 1.415 | 6.196 | 4.359 | 5.466 | 3.207 | 59.559 |
| Mixed K5/V3 uniform | 1.365 | 8.033 | 5.603 | 3.391 | 3.756 | 58.872 |
| Mixed K6/V2 uniform | 1.408 | 12.849 | 8.344 | 4.059 | 5.024 | 65.337 |
| Mixed K5/V3 MSE | 1.374 | 7.850 | 5.391 | 2.733 | 3.761 | 58.014 |
| Mixed K6/V2 MSE | 1.386 | 12.587 | 8.262 | 3.692 | 5.031 | 64.719 |

Relative to the corresponding conventional asymmetric profile, mixed prefill
packing work falls by **76.5–81.2%**, but scalar index work rises by
**76.3–85.2%** and centroid/norm-scale correction by **93.0–98.6%**.
In the last decode call, past Native work rises by **3.0–9.7%** and current
Native work by **51.6–56.9%**. The expanded encoder and larger decoder LUT
therefore offset packing savings in this implementation. This supports the
observed latency direction, but these work ratios are not TTFT/decode latency
ratios. It is not a CPU fallback: Native unpack/LUT work executes on HVX;
existing FP16-input attention and Dense MatMul paths are retained.

For K-only past decoding at `decode_126`, per-layer work is 0.897/1.038/1.181
Mcycles for the K4/K5/K6 baseline rows. In uniform Mixed K6/V2, K4 is 0.938
and K8 is 1.878 Mcycles per layer (14 layers each): roughly twice the work for
K8 inside that model. This is a different-layer/input comparison, not another
isolated kernel benchmark. All recorded Native decode busy cycles are on HVX.

`optrace/comparison.json`, `stage_comparison.csv` and `bit_width_comparison.csv`
include every stage, K/V split, HVX/HMX resource split and decode steps 0/63/126.
The bit-width table normalizes by the number of participating layers; those
layers have different real inputs, so the isolated matched-input diagnostics
above remain a separate comparison. Instrumented accelerator event times are
saved separately from unprofiled TTFT and must not replace it.

### Changed-file map

- Core package (`src/qai_hub_models/models/templates/llm/turboquant/`):
  `config.py`, `constants.py`, `packing.py`, `cache.py` add frozen 7/8-bit
  tables, direct-byte storage, immutable layer plans and cache identity;
  `export.py`, `graph_surgery.py`, `current_attention.py`, `tiled_attention.py`,
  `native_decoder.py` resolve each layer's codec and preserve current/past ABI.
- Native package (`scripts/llm/turboquant/native_decoder/`): `decoder.cpp`,
  `hvx_decode.h`, `Decode4.xml`, `build.py`, `test_hvx.cpp` register/test 7/8-bit
  HVX operations while retaining the old operation IDs and 2–6-bit kernels.
- Export/reference/audits: `convert_parts.py`, `evaluate_qwen3_kv.py`,
  `generate_constants.py`, `htp_codec_validation.py`, `validate_native_decoder.py`,
  `verify_kv_boundary.py`, `verify_rotated_attention.py` support the new profiles.
- New experiment tools: `select_mixed_layers.py`, `benchmark_mixed_once.py`,
  `profile_mixed_once.py`, `summarize_mixed_probes.py`,
  `summarize_mixed_profiles.py` freeze choices, refuse repeated measurements,
  preserve provenance and separate diagnostic work from wall-clock speed.
- Tests (`src/qai_hub_models/test/test_models/`): new
  `test_turboquant_mixed.py`, extended `test_turboquant_asymmetric.py`,
  `test_turboquant_codec.py`, `test_turboquant_current_attention.py`.
  This README records the protocol, results and limitations. No default profile
  was promoted and no commit was created automatically.

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

## Asymmetric K5/V3 and K6/V2 (opt-in, Qwen3-1.7B)

`k5_v3_scaled` and `k6_v2_scaled` retain Dense QR K seed42/V seed542,
exact LM tree indexing, QJL-off, compressed current/past KV, norm correction
and one FP16 effective scale per vector. The default remains `k4_v4_scaled`.
The frozen 2/5/6-bit LM tables are generated from the same pinned reference as
the unchanged 3/4-bit tables; no calibration or rotation selection is performed.

| Profile | K bytes / 128 coordinates | V bytes / 128 coordinates | FP16 scales | CL1024 host KV (28 layers, 8 KV heads) |
|---|---:|---:|---:|---:|
| K4/V4 | 64 | 64 | 4 bytes per K/V pair | 28.875 MiB |
| K5/V3 | 80 | 48 | 4 bytes per K/V pair | 28.875 MiB |
| K6/V2 | 96 | 32 | 4 bytes per K/V pair | 28.875 MiB |

Non-nibble widths use continuous MSB-first bit streams without padding to
4/8 bits per coordinate. The encoder uses exact power-of-two fragments and
INT32 byte sums. `Decode2/3/5/6` use HVX byte gathers, shifts, banked LUT lookup
and FP16 scale multiplication. The legacy `Decode4` HVX kernel is unchanged;
QJL's K3+sign remains a **four-bit nibble**, not the new tight three-bit stream.
QK/AV remain FP16-input matrix products; this is not bit-plane attention or a
CPU decoder fallback. New profiles require an updated Native package, supplied
explicitly; the existing default package location is not overwritten.

Reproduction (choose a fresh artifact directory; commands do not create commits):

```bash
TQ_ASYM_ROOT=/mnt/d/ai-hub-models/binaries/turboquant/asymmetric_kv_NEW
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/native_decoder/build.py \
  --out "$TQ_ASYM_ROOT/native" --test-hvx
bash scripts/llm/turboquant/qnn_runner/build_android.sh "$TQ_ASYM_ROOT/runner"
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/benchmark_asymmetric_once.py prepare \
  --work-dir "$TQ_ASYM_ROOT" --runner "$TQ_ASYM_ROOT/runner/qnn-llm-runner"
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/benchmark_asymmetric_once.py build --work-dir "$TQ_ASYM_ROOT"
for stage in audit push functional performance quality summarize; do
  PYTHONPATH=src venv/bin/python scripts/llm/turboquant/benchmark_asymmetric_once.py "$stage" --work-dir "$TQ_ASYM_ROOT" || break
done
```

The driver fixes CL1024 for all groups and reuses the LM-tree seed42 baseline
binary after configuration/hash checks. The quality input is the **legacy four
WikiText windows** (`device_assets_cl1024`, historical K4/V4 PPL approximately
25.66), not the rotation-selection experiment's heldout documents. Performance
is unprofiled, one session for short35+128 and one for long897+128 per group.
Each quality window is scored once (1023 targets); aggregate PPL is
`exp(sum(NLL_sum) / 4092)`. Reset checks and standalone codec probes are diagnostics,
not repeated performance samples. Existing attempt logs are never overwritten.

Use `validate_native_decoder.py --bits {2,3,4,5,6} --tokens 3 256` with a fresh
`--work-dir`, `--device-dir` and `--package` for standalone Native correctness.
`htp_codec_validation.py` also accepts the asymmetric profiles; keep each probe
in a fresh `--device-dir` to preserve earlier device outputs. FP16 encoder
numeric-tolerance failures are reported independently of successful HTP execution
and exact Native LUT correctness; do not silently relax the existing tolerances.

Experiment artifacts: `/mnt/d/ai-hub-models/binaries/turboquant/asymmetric_kv_20261010/`.
`protocol.json`, `runner_amendment.json`, probe reports, build/audit logs and the
single-run measurement JSONs preserve provenance; `comparison.json` and
`comparison.csv` contain the final comparison. The amendment records replacement
of a stale pre-Native runner **before any successful benchmark or quality run**.

### Single-run result: 2026-10-10, S26 Ultra / SM8850 V81

All groups use Qwen3-1.7B W4A16, fixed CL1024 (no context bucketing),
AR128 prefill, AR1 decode, tile256 and compressed current/past KV. Each
performance condition has **one** unprofiled session and each WikiText window
is scored **once**, using the existing AR128 teacher-forced scoring path.
Model loading is excluded from TTFT. Baseline binaries were reused unchanged,
but all three groups were freshly measured with the same current-main runner.

Long condition: **897 prompt tokens + 128 generated tokens**.

| Metric | K4/V4 | K5/V3 | K6/V2 |
|---|---:|---:|---:|
| TTFT (ms) | 556.62 | 2304.20 | 2458.90 |
| Prefill (tok/s) | 1614.11 | 389.46 | 364.94 |
| Decode (tok/s) | 37.68 | 4.23 | 3.98 |
| Decode total (ms/token) | 26.539 | 236.210 | 251.266 |
| Decode KV prepare (ms/token) | 1.166 | 3.063 | 3.097 |
| Decode QNN calls (ms/token) | 24.774 | 230.582 | 245.508 |
| Four-window mean NLL | 3.244966 | 3.112172 | 2.987942 |
| Four-window PPL | 25.660831 | 22.469806 | 19.844793 |
| Host KV (MiB) | 28.875 | 28.875 | 28.875 |
| I/O buffers (MiB) | 96.929 | 96.929 | 96.929 |
| End VmRSS (MiB) | 149.941 | 151.527 | 153.129 |
| Context binaries, total (MiB) | 1623.3125 | 1625.2617 | 1627.5000 |

Short condition: **35 prompt tokens + 128 generated tokens**, also fixed CL1024.

| Metric | K4/V4 | K5/V3 | K6/V2 |
|---|---:|---:|---:|
| TTFT (ms) | 66.53 | 293.34 | 308.22 |
| Prefill (tok/s) | 530.07 | 119.67 | 113.84 |
| Decode (tok/s) | 38.64 | 4.27 | 4.01 |

Quality windows each contain 1023 scored targets. Window PPLs below are
descriptive; aggregate PPL above is computed from the **sum of NLL**, not their
arithmetic mean.

| Legacy WikiText window | K4/V4 PPL | K5/V3 PPL | K6/V2 PPL |
|---|---:|---:|---:|
| w0 | 13.876348 | 12.410554 | 11.628604 |
| w1 | 33.952901 | 26.303588 | 25.016727 |
| w2 | 25.053085 | 27.910367 | 19.536115 |
| w3 | 36.734116 | 27.978578 | 27.289152 |

**Interpretation:** K5/V3 and K6/V2 reduce aggregate PPL by 12.44% and 22.67%
on these four windows at identical cache storage. K5/V3 improves three windows
but worsens w2; K6/V2 improves all four. This supports further investigation of
allocating more bits to K on this model/data, not a general accuracy guarantee.
Neither implementation is a speed win: observed long-context per-token latency
increases 8.90x and 9.47x. Single samples do not establish timing variance.

**Known implementation bottleneck:** the non-nibble native kernel is not at
the legacy Decode4 optimization level. Disassembly of the measured DSP library
shows its partial-row `memcpy` lowered to **32/48/80/96 scalar byte loads** for
2/3/5/6-bit input rows, followed by vector assembly, instead of efficient bulk
HVX loads. LUT lookup, bit extraction and FP16 products do execute on HVX;
there is **no ARM/CPU inference fallback**. This scalar DSP loading overhead
must not be described as an intrinsic penalty of asymmetric bit allocation.
Existing isolated decoder correctness profiles (no extra device runs) recorded
0.303 million Decode4 cycles versus 4.531/9.198/14.279/21.599 million cycles for
Decode2/3/5/6 at 8 heads x 256 tokens. These are diagnostic op counters, **not**
full-model wall-time fractions or end-to-end speedup estimates. Whole-model
timings locate the major increase inside QNN, but do not separate every op.
Native input-load/vectorization optimization is a follow-up, not silently
applied after the single-run measurements.

Validation: **573 CPU tests passed**; HVX simulator and isolated device LUT
decoders were exact against the FP16 oracle for every tested width. All six
attention graphs per group passed the compiled audit; non-codec source nodes,
constants, weights and activation calibration were unchanged. Device reset
checks passed, and long generation reached 1024 cached tokens. The strict
AR128 encoder scale gate remains **failed**, with maximum relative errors
0.2199% (K5/V3) and 0.2720% (K6/V2), against the unchanged 0.2% tolerance;
no unexplained scalar-index mismatches were found. This limitation is separate
from successful HTP execution/decoder correctness and is not waived by PPL.

All artifacts remain under the experiment directory above, including full
configuration/data/binary hashes, original failed diagnostics, partitioned
build provenance and disassembly/probe evidence in `comparison.json`.
The branch remains `main`, no commit was created, and K4/V4 stays the default.

### Follow-up: bounded HVX input loads for asymmetric decoding

The subsequent optimization replaces the **historical partial-row scalar
loads** above without changing the cache format, codebooks, encoder graphs or
FP16 decode results. K4/V4's existing nibble kernel and default profile remain
unchanged. Work is on `exp/asymmetric-kv-quantization`, without a new commit.

The 2/3/5/6-bit kernels now load 128 bytes at a time with HVX. Loads must fit
inside the current QHPI slice; final rows use an end-anchored vector and lane
rotation. Slices smaller than one vector use one zero-padded scratch copy.
No allocator padding or neighbouring slice is assumed readable. LUT banks that
cannot contribute to valid indices are skipped. Disassembly confirms **zero
scalar byte-load instructions** in the inspected decoder functions, replacing
the former 32/48/80/96 byte loads per row.

The unchanged x86 prepare library, registration source/XML, operation order,
function names, layouts and SDK are checked before an explicit runtime-only
QHPI override. Original `convert_report.json`, context binaries and graph hashes
are preserved; runtime reports separately identify both compiled and execution
packages. This reuse is specific to the verified execute-only change, not a
general promise that arbitrary Native packages can be swapped. Default push
behavior is unchanged; runtime overrides are opt-in and additionally require
valid original/replacement prepare libraries.

Reproduction from the preserved first experiment:

```bash
TQ_HVX_ROOT=/mnt/d/ai-hub-models/binaries/turboquant/asymmetric_hvx_NEW
TQ_PREVIOUS=/mnt/d/ai-hub-models/binaries/turboquant/asymmetric_kv_20261010
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/native_decoder/build.py \
  --out "$TQ_HVX_ROOT/native" --test-hvx
for bits in 2 3 4 5 6; do
  PYTHONPATH=src venv/bin/python scripts/llm/turboquant/validate_native_decoder.py all \
    --bits "$bits" --tokens 1 3 256 --package "$TQ_HVX_ROOT/native" \
    --work-dir "$TQ_HVX_ROOT/probe_b${bits}" \
    --device-dir "/data/local/tmp/qaihm_turboquant/asymmetric_hvx_NEW_b${bits}" || break
done
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/benchmark_asymmetric_hvx_once.py prepare \
  --work-dir "$TQ_HVX_ROOT" --previous "$TQ_PREVIOUS"
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/benchmark_asymmetric_hvx_once.py push \
  --work-dir "$TQ_HVX_ROOT"
for group in k5_v3 k6_v2; do
  for stage in functional performance quality; do
    PYTHONPATH=src venv/bin/python scripts/llm/turboquant/benchmark_asymmetric_hvx_once.py "$stage" \
      --groups "$group" --work-dir "$TQ_HVX_ROOT" || break 2
  done
done
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/benchmark_asymmetric_hvx_once.py summarize \
  --work-dir "$TQ_HVX_ROOT"
```

All probe cases must match the FP16 oracle bit-for-bit before full-model runs.
The driver also requires a successful full-model load/reset check before its
one short35+128 and one long897+128 performance session per profile. Each legacy
WikiText window is scored once. Historical K4/V4 and asymmetric results are
reused, not contemporaneously remeasured; no timing variance estimate is made.
Artifacts are under `/mnt/d/ai-hub-models/binaries/turboquant/asymmetric_hvx_20261010/`.

Long-condition results (897+128, fixed CL1024; old columns are preserved
historical measurements, optimized columns are fresh single measurements):

| Metric | K4/V4 reference | K5/V3 before | K5/V3 optimized | K6/V2 before | K6/V2 optimized |
|---|---:|---:|---:|---:|---:|
| TTFT (ms) | 556.62 | 2304.20 | 642.54 | 2458.90 | 672.17 |
| Prefill (tok/s) | 1614.11 | 389.46 | 1398.19 | 364.94 | 1336.47 |
| Decode (tok/s) | 37.68 | 4.23 | 35.81 | 3.98 | 34.57 |
| PPL, four WikiText windows | 25.660831 | 22.469806 | 22.469806 | 19.844793 | 19.844793 |

Decode throughput improves about **8.46x / 8.69x** over the initial asymmetric
implementation. It is still approximately **5.0% / 8.3% below** the historical
K4/V4 throughput; prefill is approximately **13.4% / 17.2% below**. Do not claim
identical performance or statistical equivalence. The catastrophic decoder
inefficiency is removed, while the higher-bit encoder's tree/norm/packing
work remains unchanged. In particular, earlier encoder diagnostics identify
K5/K6's small-column INT32 packing concat as a possible follow-up; this run does
not individually attribute the remaining full-model latency to that operation.

Short-condition (35+128) optimized TTFT is **75.70 / 81.50 ms**, prefill
**464.43 / 433.22 tok/s**, and decode **36.80 / 35.34 tok/s** for K5/V3 and K6/V2.
Host KV remains **28.875 MiB**, and I/O buffers remain **96.929 MiB** for both.
All four reported window NLL sums and both generated token sequences are
identical to each profile's prior result. This is not a bit-exact-logit claim.

Validation: **601 Python regression tests**, **600 host HVX cases** (including
210 protected-page cases), and all five bit widths on HTP at 1/3/256 tokens
passed. Every isolated HTP output matches the FP16 LUT oracle bit-for-bit.
Existing encoder strict scale-tolerance failures are unchanged and remain
separate from decoder correctness. `decoder_diagnostics.json` records the
measured library disassembly and isolated probe counters: Decode2/3/4/5/6 use
**0.303 / 0.338 / 0.327 / 0.356 / 0.396 million cycles**, respectively, for the
same 8-head x 256-token probe. These diagnostic counters are not end-to-end
latency fractions. `comparison.json` and `comparison.csv` contain full results,
including QNN/host timing, process memory and the preserved historical controls.

### Asymmetric KV stage profiling (diagnostic only)

`profile_asymmetric_optrace.py` compares K4/V4 with the **optimized** K5/V3 and
K6/V2 execution packages. It preserves the original quantized DLCs, graphs,
calibration, context binaries and prior performance reports. Only separate
optrace-enabled contexts and a uniquely named diagnostic runner are deployed.
Runtime package overrides use the same registration/prepare-library/hash checks
as the validated HVX follow-up; original compilation provenance is retained.

Each configuration runs **one** 897+128-token CL1024 generation session. The
eight prefill chunks and decode steps 0/63/126 are captured within that session;
they are different cache positions, not repeated benchmark trials. Greedy
continuations can differ across configurations. No new uninstrumented timing or
PPL run is part of this diagnostic. Generated IDs are checked against each
configuration's historical uninstrumented result.

```bash
TQ_PROFILE_ROOT=/mnt/d/ai-hub-models/binaries/turboquant/asymmetric_optrace_NEW
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/profile_asymmetric_optrace.py freeze \
  --root "$TQ_PROFILE_ROOT" \
  --original /mnt/d/ai-hub-models/binaries/turboquant/asymmetric_kv_20261010 \
  --optimized /mnt/d/ai-hub-models/binaries/turboquant/asymmetric_hvx_20261010 \
  --assets /mnt/d/ai-hub-models/binaries/turboquant/device_assets_cl1024
for stage in build measure render summarize; do
  PYTHONPATH=src venv/bin/python scripts/llm/turboquant/profile_asymmetric_optrace.py "$stage" \
    --root "$TQ_PROFILE_ROOT" || break
done
PYTHONPATH=src venv/bin/python scripts/llm/turboquant/summarize_asymmetric_optrace.py \
  --root "$TQ_PROFILE_ROOT"
```

Stage attribution uses physical HVX/HMX tracks, excludes duplicate views, and
recovers verified HMX arithmetic hidden behind fused post-reshape names.
Non-nibble `enc_pack_*` operations are packing, **not normalization**. All
stages, K/V splits, hardware resources, host timings and source node counts are
saved as JSON/CSV. Parallel busy cycles are work counters, not microseconds or
TTFT/decode latency fractions. Source node counts are not executed kernel counts.
Existing strict encoder numerical-validation limitations remain open.

#### Measured results: asymmetric_optrace_20261010

Artifacts: `/mnt/d/ai-hub-models/binaries/turboquant/asymmetric_optrace_20261010/`.
All three configurations completed exactly one session and 44 graph traces;
all 128 generated IDs match their respective historical uninstrumented run.
The source DLCs were hash-checked before/after context generation and before
deployment. The runtime library hashes match the historical results. K4/V4
uses its unchanged nibble package; K5/V3 and K6/V2 use the optimized HVX package
(`dab7bb8c...e19`), not the original scalar-byte-load implementation.

The table below reports **million physical HVX/HMX busy cycles**, summed over
parallel lanes. Prefill sums all eight chunks; decode is the final sampled
step, with 1023 cached tokens. These are **not milliseconds, exclusive stage
latencies, or TTFT fractions**. Different-stage intervals can overlap. Fused
work is attributed to its surviving QNN owner (with the documented HMX reshape
recovery), so these are execution-stage attributions, not isolated microbenchmarks.

| Stage | Prefill K4/V4 | K5/V3 | K6/V2 | Decode K4/V4 | K5/V3 | K6/V2 |
|---|---:|---:|---:|---:|---:|---:|
| KV normalize | 224.721 | 235.893 | 229.491 | 1.450 | 1.362 | 1.350 |
| KV Dense rotation | 1.633 | 1.530 | 1.550 | 0.018 | 0.021 | 0.018 |
| Scalar index tree | 660.874 | 720.176 | 1028.252 | 2.969 | 4.029 | 6.114 |
| Norm/effective-scale correction | 578.312 | 569.031 | 842.750 | 2.551 | 3.041 | 4.322 |
| Packing | 136.077 | 1060.654 | 1108.444 | 1.720 | 6.162 | 5.338 |
| Native past KV restore | 374.142 | 407.674 | 442.079 | 51.164 | 56.139 | 60.028 |
| Native current KV restore | 57.820 | 63.919 | 65.873 | 2.383 | 2.471 | 3.188 |
| Query rotation | 2.088 | 2.157 | 2.211 | 0.170 | 0.170 | 0.161 |
| QK | 105.443 | 102.482 | 105.032 | 6.527 | 6.532 | 6.421 |
| Score/mask/softmax | 390.987 | 384.557 | 395.069 | 3.928 | 3.815 | 3.773 |
| AV | 70.748 | 69.145 | 70.124 | 8.007 | 8.205 | 7.957 |
| Attention output rotation | 19.227 | 19.386 | 19.579 | 0.129 | 0.135 | 0.128 |
| Layout/precision conversion | 894.710 | 902.815 | 905.716 | 60.869 | 58.798 | 56.006 |
| Model linear | 171.684 | 172.120 | 172.480 | 6.807 | 7.141 | 6.747 |
| Model other | 715.320 | 717.768 | 719.525 | 10.845 | 10.576 | 10.103 |
| Unattributed named (mostly graph I/O) | 113.486 | 112.739 | 116.119 | 0.080 | 0.105 | 0.083 |
| Total compute busy cycles | 4517.271 | 5542.049 | 6224.293 | 159.616 | 168.701 | 171.736 |

Findings and implementation evidence:

1. **Packing is the dominant additional prefill cost.** Its work increases
   **7.795x / 8.146x**. K5/V3's packing delta is 924.578 million cycles out of
   a net 1024.778-million-cycle increase (about 90% of *additional work*, not
   90% of TTFT). K6/V2 adds 972.367 million packing cycles. The K-side
   `enc_pack_columns` INT32 `q::Concat` alone takes **444.625 / 562.385 million
   cycles** across prefill. The compiled inputs are `[8,128,16,1]` columns,
   concatenated into widths 5/6. `_repack()` in `export.py` also emits slices,
   shifts via Mul/Floor, masking arithmetic and float/INT32 conversions.
   K4's dedicated nibble path avoids this generic stream-regrouping graph.
   This is an implementation penalty, not an inevitable property of asymmetric
   bit allocation. V3/V2 packing is also more expensive than V4 here.

2. **K6 adds substantial scalar-tree and norm-lookup work.** Total prefill
   index work is **1.556x**, and correction work is **1.457x** K4/V4. K5/V3's
   corresponding ratios are **1.090x / 0.984x**: its lower V correction cost
   offsets the higher K correction cost in prefill, so norm correction is
   **not** a measured net K5/V3 prefill bottleneck. In `_scalar_index_tree()`,
   comparisons are one per bit, but selecting their thresholds builds additional
   Mul/Add/Where tensors. In `_select_centroid()`, the K-side affine-pair count
   grows 4 -> 8 -> 16. Layer-0 source node counts (K / V, not executed kernels):

   | Stage | K4/V4 | K5/V3 | K6/V2 |
   |---|---:|---:|---:|
   | Scalar index | 47 / 47 | 84 / 26 | 153 / 13 |
   | Scale correction | 29 / 29 | 45 / 21 | 77 / 18 |
   | Packing | 6 / 6 | 68 / 52 | 67 / 35 |

3. **Decode retains a non-nibble restoration penalty despite using HVX.**
   Past restoration costs **1.097x / 1.173x** at the final sample. The K-only
   component is **25.098 -> 29.090 -> 33.311 million cycles**, whereas V is
   **26.066 -> 27.049 -> 26.717**: fewer V bits do not recover the K overhead
   in this implementation. `tq_decode_bits_hvx()` performs byte-LUT gathers,
   variable shifts, shuffles and multi-bank centroid selection; 5-bit needs
   two centroid banks and 6-bit four, versus one in the dedicated 4-bit path.
   K4 also unpacks two rows per input vector. The old catastrophic partial-load
   scalarization is already removed; these are the *remaining* costs.

4. **Encoding still matters during decode.** Packing is **3.583x / 3.103x**,
   scalar indexing **1.357x / 2.060x**, correction **1.192x / 1.694x**. At AR1,
   the packing delta is distributed across slicing/layout/casts/arithmetic;
   the prefill INT32 Concat is not its sole explanation. All three decode
   positions show the same direction of extra codec work. K6's packing is
   actually cheaper than K5's at AR1; its larger K tree/norm/LUT costs explain
   why packing alone cannot rank the two configurations.

5. **Missing HMX/HVX execution, larger cache capacity and Dense rotation are
   not supported as primary explanations.** Rotation executes on HMX; Native
   restoration on HVX. QK/AV and model-linear work are essentially unchanged.
   Layout/precision remains a large *shared* decode cost but does not increase
   with asymmetric packing in these samples. Stored KV is **28.875 MiB** and
   I/O buffers **96.929 MiB** for all three. Equal storage does not imply equal
   load instructions, intermediate traffic or vector utilization.

SDK-reported accelerator execute times from the instrumented sessions, summed
over four graph parts (ms; not full QNN calls and not benchmark replacements):

| Sample | K4/V4 | K5/V3 | K6/V2 |
|---|---:|---:|---:|
| Prefill, all chunks | 448.277 | 531.317 | 548.667 |
| Decode 0, 897 cached tokens | 23.468 | 26.582 | 26.434 |
| Decode 63, 960 cached tokens | 26.049 | 27.138 | 29.199 |
| Decode 126, 1023 cached tokens | 26.902 | 28.006 | 26.519 |

In particular, **K6/V2 is not slower at every sampled wall-time point** despite
its extra codec work. These three cache positions are not repeat trials. There
is no variance estimate or proven frequency/scheduling cause for that reversal.
Do not turn stage work ratios into end-to-end speedup predictions. Instrumented
prefill QNN calls total **8.025 / 8.447 / 9.087 seconds** because profiling adds
large serialization/transport overhead; they are not production TTFT.
Battery temperatures before/after were 27.0/27.9, 27.2/28.3 and 27.5/28.5 C;
these are not direct DSP temperature or frequency measurements.

The existing **uninstrumented** results remain 1614.11 / 1398.19 / 1336.47
prefill tok/s and 37.68 / 35.81 / 34.57 decode tok/s. Their mean decode QNN time
is 24.774 / 26.065 / 27.149 ms and host KV preparation 1.166 / 1.252 / 1.186 ms.
Thus the historical gap is primarily inside QNN, not host KV preparation.
Those runs are historical single observations, not contemporaneous reruns.

Recommended follow-up order, **not implemented in this diagnostic**: native
bit-width-specific packing avoiding narrow INT32 Concat; K6 threshold/centroid
selection without materializing the expanded graph; specialized non-nibble
HVX unpack/LUT kernels. Preserve exact indexing, FP16 rounding, cache format
and quality checks when evaluating any of these changes. Optimizing Dense
rotation is not the evidence-backed first target for this particular gap.

Outputs include `comparison.json`, `stage_comparison.csv`,
`instrumented_host_timings.csv`, `source_inventory.json`, each group's full
optrace summary JSON/CSV, raw traces, schematics and hash-guarded manifests.
`protocol_clarification.json` corrects only an initial descriptive error:
the runner pads the **first** one-valid-token AR128 chunk, followed by seven
full chunks; inputs/execution never changed. One host SDK viewer stalled after
writing `Complete!`; its output was preserved, and rerendering the same log
exited normally with a byte-identical trace gzip. `viewer_retry.json` records
this; **no device session was repeated**. Profiling/runtime regression tests:
**55 passed**. No model algorithm/default/branch/commit change was made.

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
