<!--
Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
SPDX-License-Identifier: BSD-3-Clause
-->

# TurboQuant KV-cache tools

Developer tools for the opt-in TurboQuant KV-cache codec in
`qai_hub_models.models.templates.llm.turboquant`. The ABI, numeric tolerances
and current results are in
[`tutorials/llm/turboquant_design.md`](../../../tutorials/llm/turboquant_design.md).

## Structured K8 layer search v2 (opt-in, 2026-10-20–21)

Branch `exp/structured-k8-layer-search-v2`, based on
`exp/uniform-k8-baseline` commit `cdd68c447`. Artifact root:
`/mnt/d/ai-hub-models/binaries/turboquant/structured_k8_layer_search_v2_20261020`.
No automatic commit, default promotion, new codebook, weight training or
activation recalibration. Previous results are not overwritten.

All deployable swap candidates and exported allocations keep **14 Structured K8 + 14 LM K4,
all LM V2**, Dense K42/V542, norm correction, FP16 effective scales,
current/past compression, tile256 and arithmetic HVX `Decode8Structured`.
Qwen3-1.7B W4A16, S26 Ultra and CL1024 are fixed. Allocated host KV remains
**30,277,632 bytes = 28.875 MiB**. “Uniform-Odd/Even” refers to layer placement,
**not** the Uniform scalar quantizer from the preceding experiment.
The existing allocation JSON/profile and inference implementation are reused.

### Pre-registered documents and methods

`protocol.json` and `data_manifest.json` were frozen before allocation scoring.
New WikiText-2 train documents provide **12 search + 4 selection-validation**
windows. Eight unused documents from the official **validation partition**
are reserved as this experiment's independent test. This does not mean they
are absent from pretraining or unknown original activation calibration.
Prior recorded document titles and 16-token spans are excluded, as are
cross-split overlaps. Every full window has 1024 tokens/1023 targets.
Historical four-window WikiText NLL/PPL is reference-only after selection.
Two historical inputs are used earlier only for eager/replay bit-exactness
regression, not for an allocation score or selection decision.

Selection is salted content-hash order with offset `min(256,length-1024)`,
never a model score. Search, validation and test token/content/source hashes
are retained. The scorer rejects `test` as an offline selection split.
The HTP scoring helper also refuses independent-test and historical-reference
evaluation until the final selection file exists, including direct helper calls.

| Method | Starts | Search rule |
|---|---|---|
| Legacy Greedy | Uniform-Odd | Existing 28 single-flip sensitivity probes, top 4 promotions x top 4 demotions, at most 3 improving steps |
| Multi-start Greedy | Odd, Even, Early, Late, Middle, Boundary | Same legacy rule/budget independently at all six starts |
| Multi-start Beam/Swap | Same six starts | Width 2, at most 8 steps, all 196 simultaneous swaps per frontier parent |

The legacy single-flip probes alone have 13/15 K8 layers; they are explicitly
nondeployable diagnostics. Every combined swap, Structured HTP configuration
and exported plan has exactly 14 K8 layers. Odd's greedy execution is shared
between the legacy and multi-start comparison, not duplicated.

For Beam/Swap, screen all 196 swaps on the **first 256 tokens of two fixed
search documents**, then evaluate the four best unvisited swaps per parent
on all **12 full search windows**. The beam may temporarily worsen; a separate
incumbent always retains the lowest full-search NLL seen. This allows limited
exploration beyond a greedy local minimum. Ties use ascending mask integer;
incumbent improvements require more than `1e-6` NLL. No attention-MSE objective.

Each start has the same budget within a method. Total budgets differ between
methods: any quality comparison must also show the workload, not claim a
matched-compute algorithm advantage. A SQLite cache keys immutable document
scores by numerical-evaluator identity, configuration/mask, actual input hash
and length. Logical unique requests and physical cache misses are separately
recorded. Incremental wall time with this shared cache is not cold-cache
standalone latency; method order is fixed Greedy then Beam.
The final report also saves a standalone `offline/scores_release.sqlite`
backup with all committed WAL rows, table counts and SHA-256, so the score
cache can be reopened without depending on live journal files.

### Offline fidelity and final HTP gate

The existing exported-weight W4A16 surrogate is preserved numerically.
`allocation_offline_model.codec` accepts optional pre-created Structured
constants; its default path remains unchanged. `allocation_v2_replay.py`
captures both key codecs and a device-side layer flag in CUDA Graphs to reduce
Python launch overhead. This acceleration is **offline only**, not a deployment
fallback or change to HTP attention.

Shape-matched eager versus CUDA Graph logits are bit-exact in **24 checks**
(six static masks, two historical documents, 256/1024 tokens). Observed replay
cost is approximately **0.30 s / 0.85 s** for 256/1024 tokens on RTX 3090.
This proves replay fidelity to the existing surrogate, **not HTP equivalence**.
The first capture attempt failed because of per-call host-to-device constants;
the optional pre-created constants resolved capture without changing arithmetic.
Both successful preflight versions and later logs are retained.

Choose each method's nominee by its lowest **full search-12** final NLL.
Deduplicate nominees, filling to two distinct finalists from other completed
searches when possible, at most three. Measure actual HTP search-12 and
validation-4 NLL for the nominees and Uniform-Odd. Only **HTP validation-4**
selects the final mask; ties within `1e-6` prefer Odd, otherwise mask integer.
Freeze `selection.json` before any independent-test scoring.
Calibration of Odd and already-fixed method nominees can run concurrently with the
remaining GPU search. The final calibration stage validates and reuses these
same per-document reports; it does not select a winner until all nominees
are available, and does not rerun completed documents.

After selection, check **all 196 swaps on full offline search-12** around that
final mask. If any improves, report failure of this local-optimality check;
do not silently extend search or change the selected mask. Even a passing
check establishes neither HTP-local nor global optimality.
`figures/local_swap_delta.svg` visualizes all 14 x 14 full-search NLL changes,
with the best checked swap marked; it does not change the frozen selection.

HTP comparisons include Odd, Early, Boundary, the three method nominees and
uniform LM K6/V2. Exact aliases share binaries and scores. Performance uses
897+128 tokens and three crossed rounds (forward, reverse, rotated order),
with mean/sample SD/min/max. Independent test and historical reference PPL
use `exp(total_NLL/total_targets)`. Existing encoder numerical limitations
remain separate; no tolerance is relaxed.

Early, Boundary and K6/V2 retain their original DLCs but receive private
context rebuilds against the same current Native package used by Odd.
The archived K6/V2 bundle is bin-only; its graph files are resolved from the
original per-part directories with recorded hashes. The initial missing-DLC
failure and retry logs are retained. No old bundle is rewritten.
On-device copies share only the SHA-identical embedding `part1_of_4.bin`
within this new experiment, using symbolic links to reduce staging space
(the device denied hard-link creation; no permissions/policies were changed).
No previous device artifact is deleted. Short load/reset probes consume only
the pre-existing generic prompt, not a new calibration/test document.

Final implementation/search regressions pass **723 tests**, with no failures,
errors or skips. The completed HTP calibration, independent test and three-round
timing results are reported below, including negative search/optimality results.
Offline `physical_evaluation_seconds` measures synchronized scoring wall time,
including transfer and loss reduction, not GPU busy time. Coarse 256-token and
full 1024-token document evaluations are counted separately. Incremental wall
times also reflect concurrent CPU build activity, not isolated cold-start runs.
Method clocks start after model/replay setup and cover offline search only;
HTP compilation and calibration are additional workflow costs, not included
as separate charges in the method timing table.
`offline_document_nll.csv` exports the cache's per-mask/document scores,
diagnostic-only 13/15-layer flags and paired Odd differences where measured;
`document_nll.csv` contains the actual HTP results. No extra inference is
performed to fill missing offline reference scores.
The final report also checks whether coarse screening excluded a neighbor
whose full search-12 score was already cached by Greedy before Beam began.
This uses only existing search scores, excludes visited/evaluated masks, and
does not change nomination or selection. Coverage is partial; it is not an
oracle over all neighbors or evidence of independent-test benefit.

### Completed offline search

All six starts completed both searches without changing the frozen budget.
The following NLLs use only the full **search-12** set, not independent test.

| Start | Initial NLL | Greedy final NLL | Beam final NLL | Greedy minutes | Beam minutes |
|---|---:|---:|---:|---:|---:|
| Uniform-Odd | 3.405098 | 3.142621 | 3.154515 | 23.04 | 36.54 |
| Uniform-Even | 3.180709 | 3.137719 | 3.144728 | 23.11 | 35.76 |
| Early | 3.152349 | 3.126730 | 3.136665 | 22.63 | 34.62 |
| Late | 3.506036 | 3.169020 | 3.165531 | 22.97 | 35.96 |
| Middle | 3.495229 | 3.153958 | 3.161785 | 23.08 | 35.12 |
| Boundary | 3.158131 | 3.126435 | 3.144114 | 22.72 | 28.22 |

| Method nominee | Full search NLL | Incremental wall minutes | 1024-token document evaluations, logical / physical | 256-token document evaluations, logical / physical |
|---|---:|---:|---:|---:|
| Legacy Greedy (Odd) | 3.142621 | 23.04 | 1,548 / 1,548 | 0 / 0 |
| Multi-start Greedy (Boundary) | 3.126435 | 137.54 | 9,228 / 9,228 | 0 / 0 |
| Multi-start Beam/Swap (Early) | 3.136665 | 206.22 | 4,080 / 3,900 | 28,170 / 28,170 |

Legacy's work is included in Multi-start Greedy, not an additional execution.
The first two methods request 49/289 deployable 14-layer masks and 80/480
diagnostic-only 13/15-layer masks respectively at full length. Beam requests
340 full-length deployable masks and 14,085 short-input masks. Shared-cache
reuse saves 180 full document evaluations in Beam; these are measured
incremental times after Greedy, not isolated cold-cache timings.
Model/replay setup, HTP build/calibration, and the final local-swap check
are outside these method clocks.

Multi-start Greedy lowers search NLL relative to the legacy run with about
six times its measured offline cost. Beam is worse than Multi-start Greedy
despite more elapsed time; it beats the matching Greedy run at only the Late
start. This does not establish a search-efficiency advantage.

A **post-hoc read-only diagnostic** finds 15 of 48 Beam steps where screening
excluded an unvisited neighbor with a better full-search score already known
from the preceding Greedy runs. For example, Middle step 3 retained NLL
3.181716 while a neighbor scoring 3.154127 ranked **102nd** on the short
proxy and was excluded by the top-four shortlist. Its additional potential
search-NLL reduction was 0.027589. This demonstrates false negatives in the
two-document/256-token proxy, but is only partial neighbor coverage and does
not prove the sole cause, HTP benefit or test improvement. No candidate,
screening rule or final selection is changed using this diagnostic.

### Actual HTP calibration and frozen selection

| Configuration | HTP search-12 NLL | HTP validation-4 NLL | HTP validation PPL |
|---|---:|---:|---:|
| Uniform-Odd | 3.381782 | 3.641847 | 38.162252 |
| Legacy Greedy | 3.138629 | 3.349474 | 28.487745 |
| Multi-start Greedy | 3.132733 | 3.330721 | 27.958498 |
| Multi-start Beam/Swap | 3.131240 | **3.323396** | **27.754447** |

The pre-registered HTP validation rule selects **Multi-start Beam/Swap**,
not the lowest offline-search nominee. Its K8 layers are:

```text
0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 16, 20, 26
```

`plans/final_selected.json` and `selection.json` freeze this allocation before
any new test scoring. Configuration SHA256:
`bf4f94221ea619e19260b3cecfb53bbe62b86f9b245d379ade1a6465c9af5a6f`.
The reversed Greedy/Beam ranking between the offline and HTP search-12
evaluators demonstrates that eager-exact CUDA replay is not a replacement
for the actual HTP gate. Calibration gains alone are not test gains.

### Independent test and historical reference

All groups use the same eight new test windows, **8,184 scored targets per
group**. The historical four-window column is a separate dataset, evaluated
only after selection, and is not used to change the selected allocation.

| Configuration | New test mean NLL | New test PPL | Historical WikiText PPL |
|---|---:|---:|---:|
| Uniform-Odd | 3.220856 | 25.049564 | 23.612187 |
| Early | **2.977995** | **19.648390** | 20.295398 |
| Boundary | 2.999249 | 20.070465 | 20.168807 |
| Legacy Greedy | 3.010950 | 20.306679 | 19.884610 |
| Multi-start Greedy | 2.982032 | 19.727861 | 20.117655 |
| Multi-start Beam/Swap — selected | 2.985034 | 19.787172 | 19.905564 |
| Uniform LM K6/V2 | 3.031423 | 20.726705 | **19.844793** |

Uniform-Odd's historical four **per-document NLL sums are exactly unchanged**
from the preceding `uniform_k8_baseline_20261020` Structured run.
The selected Beam allocation has the following paired new-test differences;
negative delta NLL means improvement:

| Reference | Selected minus reference NLL | Selected PPL reduction | Improved / worsened documents |
|---|---:|---:|---:|
| Uniform-Odd | -0.235823 | 21.008% | 8 / 0 |
| Early | +0.007038 | -0.706% | 3 / 5 |
| Boundary | -0.014215 | 1.411% | 5 / 3 |
| Legacy Greedy | -0.025916 | 2.558% | 7 / 1 |
| Multi-start Greedy | +0.003002 | -0.301% | 3 / 5 |
| Uniform LM K6/V2 | -0.046389 | 4.533% | 6 / 2 |

Against Legacy Greedy, the largest single document accounts for 26.53% of
the selected allocation's positive document gains; the observed improvement
is not confined to one document. Nevertheless, only eight independent test
documents were evaluated. These descriptive differences do not establish
statistical significance or general superiority.

**Interpretation:** the selected candidate improves on the original Odd-start
Greedy on this independent test, but Beam's added search is not supported as
an efficiency/accuracy win over Multi-start Greedy or static Early.
Beam uses about 8.95 times Legacy's observed incremental offline time, and
more time than Multi-start Greedy while producing worse new-test NLL.
Early is the best tested Structured allocation on this test, but it is
**not retrospectively selected or promoted using test results**.
The historical WikiText ranking differs: uniform K6/V2 is slightly better
than the selected candidate there. Neither dataset is silently substituted
for the other.

### Three crossed HTP performance rounds

897 prompt tokens + 128 generated tokens, CL1024, uninstrumented runs on
the same S26 Ultra. Values below are **mean ± sample standard deviation**.
All 21 scheduled sessions are retained; no outlier is removed.

| Configuration | TTFT ms | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| Uniform-Odd | 510.14 ± 25.25 | 1763.1 ± 83.9 | 37.099 ± 0.127 |
| Early | 491.01 ± 2.72 | 1828.4 ± 10.2 | 37.391 ± 0.029 |
| Boundary | 490.69 ± 3.33 | 1829.5 ± 12.2 | 37.860 ± 0.349 |
| Legacy Greedy | 503.11 ± 28.62 | 1788.6 ± 97.5 | 37.309 ± 0.009 |
| Multi-start Greedy | 510.99 ± 20.81 | 1759.4 ± 69.0 | 37.270 ± 0.135 |
| Multi-start Beam/Swap — selected | 508.78 ± 26.64 | 1768.2 ± 89.1 | 37.272 ± 0.012 |
| Uniform LM K6/V2 | 626.00 ± 13.43 | 1434.6 ± 29.9 | 34.777 ± 0.163 |

Observed minima/maxima, not confidence intervals:

| Configuration | TTFT ms range | Prefill tok/s range | Decode tok/s range |
|---|---:|---:|---:|
| Uniform-Odd | 495.46–539.30 | 1666.3–1811.9 | 37.005–37.243 |
| Early | 488.07–493.43 | 1819.4–1839.5 | 37.365–37.423 |
| Boundary | 487.08–493.64 | 1818.6–1842.7 | 37.457–38.069 |
| Legacy Greedy | 485.26–536.13 | 1676.2–1850.0 | 37.300–37.317 |
| Multi-start Greedy | 498.49–535.00 | 1679.8–1801.0 | 37.124–37.391 |
| Multi-start Beam/Swap — selected | 489.99–539.26 | 1666.4–1832.1 | 37.261–37.284 |
| Uniform LM K6/V2 | 617.41–641.48 | 1400.1–1453.8 | 34.596–34.909 |

The selected candidate's mean decode is 7.17% higher and mean TTFT 18.73%
lower than uniform K6/V2 in these runs. Decode is essentially unchanged
relative to Legacy Greedy (37.272 versus 37.309 tok/s); this experiment does
not demonstrate a speed gain from the search itself. Three crossed runs
describe variability, not statistical superiority.

Every run reports **28.875 MiB host KV** and **96.9292 MiB I/O buffers**.
These allocations are distinct from process RSS/HWM; none is a measurement
of total accelerator memory. The following process values are three-run
means, and binary sizes sum all four context parts:

| Configuration | End VmRSS MiB | Process VmHWM MiB | Binary MiB |
|---|---:|---:|---:|
| Uniform-Odd | 149.055 | 605.215 | 1621.328 |
| Early | 147.816 | 605.100 | 1621.160 |
| Boundary | 150.008 | 605.112 | 1621.332 |
| Legacy Greedy | 150.355 | 605.254 | 1621.355 |
| Multi-start Greedy | 147.983 | 605.090 | 1621.309 |
| Multi-start Beam/Swap — selected | 149.964 | 605.160 | 1621.309 |
| Uniform LM K6/V2 | 152.159 | 605.109 | 1627.500 |

### Full local-swap check and evidence

The frozen selected mask **fails the full offline single-swap local-optimality
check**: 19 of 196 neighbors reduce search-12 NLL beyond `1e-6`.
The best checked swap demotes **layer 9 from K8 to K4** and promotes
**layer 22 from K4 to K8**:

```text
Selected offline search NLL: 3.136664700679
Best checked neighbor NLL:   3.128829304168
Delta:                     -0.007835396511
```

This neighbor is not substituted into the frozen selection, built, or measured
on HTP/test. Its offline improvement does not establish a device/test benefit.
The result, together with the screening false negatives, shows that the bounded
Beam/proxy procedure did not even establish local optimality for its full
offline search objective. No global-optimum claim is made.

The check requests **197 masks including the center x 12 documents = 2,364**
full document evaluations. It reuses 192 cached results and performs **2,172**
new evaluations. Synchronized scoring consumes 1,870.37 s; complete certificate
stage wall time is **2,002.31 s = 33.37 minutes**, including initialization and
bookkeeping. This is additional to the method search times above.
The raw certificate's `cost.attempts` is empty and its
`completed_attempt_wall_seconds` is therefore zero: that field is **not**
the certificate duration. The reported stage wall time comes from its start/end
events; `physical_evaluation_seconds` is the separately measured scoring sum.

Completed evidence includes:

- 24 eager/replay bit-exact logit cases; numerical evaluator sources remain frozen.
- Seven compiled-graph/allocation audits and seven identical-output two-session
  reset checks; all device scoring/generation uses the same HTP backend and
  Native package.
- 21 uninstrumented performance sessions, 56 new-test document scores,
  28 historical-reference scores, and 64 HTP calibration document scores.
- 723 passing regression tests; unchanged historical Odd NLL; no tolerance
  relaxation or claim that the inherited standalone V4 AR128 scale failure
  has been resolved.
- `comparison.json`/CSV, per-document NLL, per-start finals and histories,
  layer-mask/search-history/local-swap SVGs, immutable configuration plans,
  and a standalone cache snapshot containing **43,482** physical document
  score records (28,170 short + 15,312 full).

`verify_allocation_v2.py` audits hashes, prior profiles/artifacts, data separation,
selection ordering, device conditions, metrics and unchanged base commit;
`final_verification.json` is the release evidence ledger. No default promotion,
additional training, 4B experiment or automatic commit is performed.

The parallel certificate/evaluation stage-start messages exposed a mounted-drive
append race: one journal line was lost, while pipeline stdout retained both
events. `pipeline_journal_observation.json` records the exact original event.
Existing logs are preserved; future `EventJournal` writes are thread-serialized
and concurrency-tested. Neither running measurement was restarted or changed.

### Reproduction

Use a fresh root; resume only with unchanged protocol/scorer/algorithm sources.
Existing results refuse overwrite. Static HTP builds can run while the offline
search runs; selection and test stages remain ordered.
Plot reporting requires `matplotlib`; its test is optional in minimal test
environments. Tokenizer/model-specific dependencies are loaded only when
preparing new documents, not when importing orchestration or report helpers.

```bash
TQ_SEARCH_ROOT=/mnt/d/ai-hub-models/binaries/turboquant/structured_k8_layer_search_v2_repro
mkdir "$TQ_SEARCH_ROOT"
venv/bin/python scripts/llm/turboquant/allocation_v2_replay.py \
  --preflight-report "$TQ_SEARCH_ROOT/replay_preflight_v2.json"
venv/bin/python scripts/llm/turboquant/prepare_allocation_v2.py --root "$TQ_SEARCH_ROOT" \
  --data-manifest /mnt/d/ai-hub-models/binaries/turboquant/structured_k8_layer_search_v2_20261020/data_manifest.json
venv/bin/python -m pytest -q src/qai_hub_models/test/test_models/test_turboquant*.py
venv/bin/python scripts/llm/turboquant/search_allocation_v2.py search --root "$TQ_SEARCH_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_allocation_v2.py static --root "$TQ_SEARCH_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_allocation_v2.py probe --root "$TQ_SEARCH_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_allocation_v2.py candidates --root "$TQ_SEARCH_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_allocation_v2.py calibrate --root "$TQ_SEARCH_ROOT"
venv/bin/python scripts/llm/turboquant/search_allocation_v2.py certificate --root "$TQ_SEARCH_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_allocation_v2.py evaluate --root "$TQ_SEARCH_ROOT"
venv/bin/python -m pytest -q src/qai_hub_models/test/test_models/test_turboquant*.py \
  --junitxml="$TQ_SEARCH_ROOT/regression_final.xml"
venv/bin/python scripts/llm/turboquant/report_allocation_v2.py summary --root "$TQ_SEARCH_ROOT"
venv/bin/python scripts/llm/turboquant/verify_allocation_v2.py --root "$TQ_SEARCH_ROOT"
```

After static builds, `run_allocation_v2.py --root "$TQ_SEARCH_ROOT"` orders
the remaining stages, overlaps the GPU swap certificate with HTP evaluation,
and stops on errors without reducing search budgets. With an already-running
search, `--wait-for-search-pid <PID>` waits for that exact process instead of
launching duplicate inference. It finishes at the comparison report; README
interpretation and the final evidence audit remain explicit review steps.
`report_allocation_v2.py progress --root "$TQ_SEARCH_ROOT"` is a read-only
status check and does not consume a calibration or test window.

Optional build overlap while the GPU search is running:

```bash
# Only after greedy_uniform_odd.json is complete:
venv/bin/python scripts/llm/turboquant/benchmark_allocation_v2.py prebuild-legacy --root "$TQ_SEARCH_ROOT"
# Only after all six greedy_<start>.json files are complete:
venv/bin/python scripts/llm/turboquant/benchmark_allocation_v2.py prebuild-greedy --root "$TQ_SEARCH_ROOT"
```

These commands require completed, hash-matched results and use the same frozen
nomination rule. They do not build provisional winners, add candidates, consume
validation/test data or change the search budget. Later nomination serialization
must match the prebuilt plan exactly; the ordinary candidate stage reuses it.

## Uniform K8 versus Structured K8 (opt-in, 2026-10-20)

Branch `exp/uniform-k8-baseline`, based on `exp/lloyd-max-k8-convergence`
commit `70b8b006e`. Existing defaults, codebooks and experiment contents are
preserved; no automatic commit, model training or activation recalibration.
Artifacts: `/mnt/d/ai-hub-models/binaries/turboquant/uniform_k8_baseline_20261020`.

Both groups use Qwen3-1.7B W4A16, S26 Ultra, CL1024, odd14 K8/even14 K4,
V2 on every layer, Dense K42/V542, norm correction, FP16 effective scales,
current/past KV compression and tile256 attention. K4/V2 remain Lloyd–Max.
Index payload is 28 MiB plus 0.875 MiB of scales: **28.875 MiB** allocated KV.

| Group | Profile | K8 decoder |
|---|---|---|
| Uniform | `mixed_k6_v2_uniform_uniform_k8` | new `Decode8Uniform` HVX arithmetic |
| Structured | `mixed_k6_v2_uniform_structured_k8` | existing `Decode8Structured` HVX arithmetic |

The first `uniform` in the profile name describes layer placement, the final
`uniform_k8` describes the new quantizer. `uniform_k8_v4_scaled` is the
isolated encoder probe, not the full model's V2 allocation. The opt-in
configuration records coefficients and codebook hashes and rejects cache
state from another quantizer. Legacy profile hashes remain unchanged.

### One-spacing Gaussian optimization and FP16 contract

`optimize_uniform_k8.py` optimizes the single FP64 spacing for a symmetric
256-level **mid-rise** quantizer on `N(0,1/128)`. Positive centroids are
`(j+1/2)*delta`, `j=0..127`; tails saturate. Three fixed bounded searches
converge to the same nearest FP16 spacing. A fixed 2049-point broad grid and
independent numerical integral check the result. No real KV, NLL or PPL is
used to choose the spacing. No general global-optimality claim is made.

- Ideal FP64 delta: `0.002719036573080505`.
- Frozen FP16 delta: `0.002719879150390625` (`0x1.648p-9`).
- Offset: `delta/2 = 0.0013599395751953125`, not an independently fitted parameter.
- Ideal Gaussian MSE: `6.85048326435e-7`; numerical integral agrees to `4.6e-12` relative error.
- FP16 centroid SHA256: `30d0dfd34829196d478b0ec8b77e71880f8f217ac236a189d7e74805bffa905e`.

Encoder uses `j=clip(floor(abs(y)*reciprocal_delta),0,127)` and code
`128+j` for nonnegative `y`, `127-j` otherwise. The code is stored directly
as UINT8; no boundary tree, distance search, region branch or bit packing.
The decoder reconstructs
`FP16(FP16(j*delta)+offset)`, applies sign, then performs the FP16 effective
scale multiply. These separate rounds match Structured's arithmetic contract;
the compiled Uniform kernel has no centroid LUT. Normalization, rotation and
the selected-centroid norm correction remain unchanged.

Exact uniform spacing describes the pre-rounding codebook. FP16 execution
necessarily perturbs the stored centroid spacing; ideal-nearest MSE must
not be confused with executed arithmetic-index MSE. Zero vectors still
restore to zero through a zero effective scale, despite the mid-rise codebook
having no zero centroid.

| Diagnostic metric | Uniform K8 | Structured K8 |
|---|---:|---:|
| FP16 centroids, ideal nearest-rule Gaussian MSE | 6.8467158e-7 | 4.7935449e-7 |
| Gaussian MSE including FP16 input/index/centroid arithmetic | 6.8695588e-7 | 4.7989992e-7 |
| Captured K reconstruction MSE, K8 layers only | 0.0012707398 | 0.0009215808 |
| Captured QK score MSE, K8 layers only | 0.0033231618 | 0.0024291137 |

The scalar FP16 calculation integrates over all 31,744 nonnegative finite
binary16 input rounding cells, including saturation and the signed-zero
convention. Vector diagnostics reuse two frozen historical W4A16 FP16-KV
captures, all 28 layers and 32 fixed causal query positions/document, with
GQA=2. QK errors here are for the unscaled dot product, before the usual
`1/sqrt(head_dim)` attention scaling. The vector emulator has FP16 operation
boundaries and FP32 reductions;
it is **not HTP bit-exact**. All-layer/rotated-reference errors and a fixed-seed
synthetic final-KV check are also in `offline_quality.json`.

### Numerical validation and frozen evaluation protocol

The CPU/ONNX tests cover sign symmetry, zero/extremes, decision boundaries,
raw-byte format, cache-hash rejection and current/past Native routing. Native
host tests retain all previous 2..8-bit tests and additionally check Uniform
arithmetic against its FP16 LUT oracle under all codes, edge scales, signed
zero, subnormals, unaligned I/O, partial slices and guard pages. The LUT is
a **test oracle**, not a deployed Uniform decoder.

The final TurboQuant regression suite passes **705 tests** (two existing SWIG
deprecation warnings). All **17** previously registered profile hashes are
unchanged. Ruff passes. An additional mypy check reports two pre-existing
tuple/list typing errors in `_select_centroid` (`export.py`); that unchanged
LM implementation is not rewritten as part of this experiment.

Both HTP decoders are FP16 **bit-exact** for 1/3/256/1023-token probes.
Uniform K encoder AR1/AR128 scale maximum relative errors are
`0.11037% / 0.16041%`; Structured is `0.11062% / 0.17255%`, all below the
unchanged 0.2% limit. Float64-oracle index differences are adjacent-boundary
rounding differences: zero unexplained mismatches, not a claim of bit-exact
indices across FP64 and HTP. Every encoder operator is present in the HTP
profile. The common **V4** AR128 probe still fails its scale gate at
`0.24766%`; this is tracked separately and is **not the full-model V2 path**.

Legacy quality uses the unchanged four WikiText windows. New held-out
documents are **Hed PE**, **The Heart of Ezra Greer**, **The Hustler (film)**,
and **Josepha Petrick Kemarre**. Selection is frozen salted content-hash order,
excluding previously recorded titles and any document with prior 16-token
overlap, without looking at outcomes. Each document supplies one 1024-token
window and 1023 scored targets. Aggregate PPL is `exp(total_NLL/4092)`, not
the mean of window PPLs. This is independence from recorded experiments,
not a claim about the pretrained model's training data.

Unprofiled performance uses **897+128 tokens, three runs/group** in the
predeclared order **Uniform, Structured, Structured, Uniform, Uniform,
Structured**. Reports retain raw samples, mean, sample standard deviation
(`ddof=1`), minimum and maximum. Reset checks and codec diagnostics are separate
from throughput samples; PPL is scored once per window. Three observations
are not a statistical-superiority guarantee. Both models use the same new
Native library package; Structured source DLCs are reused and contexts are
rebuilt privately, never through old output-file symlinks.

All six paired graph audits pass: model I/O, non-K8 operations/constants,
activation encodings and quantized-DLC MatMul/FullyConnected contracts match.
The Uniform graph has 24 fewer quantized-DLC operators per K8 layer (336 across
14 layers for each AR shape); no K8 decoder LUT is present. These are graph
structure checks, not evidence of identical HTP scheduling or a proportional
latency improvement. Full details are in `reports/graph_audit.json`.

### Full-model results

Performance cells are **mean +/- sample SD [min, max]**, three crossed runs
per group, without profiling. Quality is one teacher-forced pass/window,
4092 targets/split/group. No data or quantizer parameters were changed after
seeing these results.

| Metric | Uniform K8 | Structured K8 |
|---|---:|---:|
| TTFT (ms) | 536.446 +/- 7.752 [530.278, 545.149] | 539.084 +/- 5.060 [533.821, 543.915] |
| Prefill (tok/s) | 1675.256 +/- 23.938 [1648.435, 1694.453] | 1666.834 +/- 15.706 [1651.927, 1683.232] |
| Decode (tok/s) | 37.1770 +/- 0.0241 [37.1558, 37.2032] | 37.1122 +/- 0.0568 [37.0502, 37.1619] |
| Legacy mean NLL | 3.16942902 | 3.16176300 |
| Legacy PPL | 23.79389464 | 23.61218745 |
| New held-out mean NLL | 3.62741375 | 3.50768736 |
| New held-out PPL | 37.61540782 | 33.37100346 |
| Allocated KV (MiB) | 28.875 | 28.875 |
| Runtime I/O buffers (MiB) | 96.9292 | 96.9292 |
| End VmRSS (MiB) | 150.992 +/- 0.140 | 151.159 +/- 0.052 |
| Process VmHWM (MiB) | 605.914 +/- 0.228 | 606.003 +/- 0.095 |
| Four context binaries, total (MiB) | 1620.8164 | 1621.3281 |

The Structured legacy NLL sums match the prior experiment exactly for all
four windows. The **new held-out set is different from previous experiments**;
its absolute PPL must not be compared with an earlier held-out PPL.

| New held-out document | Uniform mean NLL | Structured mean NLL | Uniform - Structured |
|---|---:|---:|---:|
| Hed PE | 3.03547252 | 3.04424901 | -0.00877649 |
| The Heart of Ezra Greer | 3.88743994 | 3.85577839 | +0.03166155 |
| The Hustler (film) | 3.96661961 | 3.58517121 | +0.38144840 |
| Josepha Petrick Kemarre | 3.62012292 | 3.54555084 | +0.07457208 |

Uniform changes aggregate legacy PPL by **+0.770%**, new held-out PPL by
**+12.719%**, mean TTFT by **-0.489%**, prefill by **+0.505%**, and decode by
**+0.175%**. Legacy windows split two better/two worse; held-out is one
better/three worse. The Hustler accounts for about **79.65% of the net held-out
NLL increase**. This is evidence on these frozen windows, not a claim that
every document or corpus will suffer a 12.7% PPL increase.

The one-spacing Uniform implementation is simpler, but this experiment does
**not** establish a meaningful end-to-end speed advantage: timing differences
are small, ranges overlap, and there are only three observations per group.
The worse scalar/K/QK MSE and aggregate PPL do not support replacing Structured
with Uniform on these conditions. Preserve Uniform as an opt-in research
baseline; do not promote it or change the default.

`comparison.json`, `comparison.csv` and `document_nll.csv` retain full precision,
raw timing samples, document scores and configuration/binary hashes.
`codec_costs.json` keeps named-op encoder cycle counters separate from physical
decoder HVX busy cycles and elapsed SDK execute times; none of these cycle
shares is an end-to-end TTFT percentage.

| Instrumented codec diagnostic | Uniform | Structured |
|---|---:|---:|
| AR1 scalar-index named-op cycles | 57,498 | 95,563 |
| AR128 scalar-index named-op cycles | 836,971 | 1,153,656 |
| AR128 centroid/norm-scale named-op cycles | 617,539 | 794,858 |
| AR128 direct-byte/output named-op cycles | 325,706 | 323,224 |
| AR128 encoder SDK execute (us) | 1,459 | 1,421 |
| Tile256 decoder physical HVX busy cycles | 354,100 | 356,227 |
| Tile256 decoder interval-union cycles | 59,805 | 60,529 |
| Tile256 decoder SDK execute (us) | 4,399 | 4,396 |
| Tile256 decoder execute excluding wait (us) | 648 | 678 |

These are separate single instrumented probes, not the three unprofiled
end-to-end samples. Uniform removes the region decision and reduces the
observed index/norm-scale work, but it does not remove byte-output traffic or
the decoder's final scale multiply. The measured decoder busy-cycle difference
is only about 0.6%; no decoder speed superiority is established. SDK execution
also includes scheduling/launch/wait effects, so lower named-op cycle totals
do not imply a lower elapsed encoder time.

### Reproduction and files

Use a new artifact root. Exclusive experiment logs/reports refuse overwriting
previous attempts. Native builds stay in that root; do not overwrite a prior
experiment's package. `benchmark_uniform_k8.py build` requires passed CPU and
HTP decoder/K-encoder checks before full-model compilation.

```bash
TQ_UNIFORM_ROOT=/mnt/d/ai-hub-models/binaries/turboquant/uniform_k8_baseline_repro
venv/bin/python scripts/llm/turboquant/optimize_uniform_k8.py --out "$TQ_UNIFORM_ROOT/codebook"
venv/bin/python scripts/llm/turboquant/native_decoder/build.py --out "$TQ_UNIFORM_ROOT/native" --test-hvx
venv/bin/python scripts/llm/turboquant/benchmark_uniform_k8.py prepare --root "$TQ_UNIFORM_ROOT" \
  --data-manifest /mnt/d/ai-hub-models/binaries/turboquant/uniform_k8_baseline_20261020/data_manifest.json
venv/bin/python scripts/llm/turboquant/verify_uniform_k8.py hvx --root "$TQ_UNIFORM_ROOT"
venv/bin/python scripts/llm/turboquant/verify_uniform_k8.py preservation --root "$TQ_UNIFORM_ROOT"
venv/bin/python -m pytest -q src/qai_hub_models/test/test_models/test_turboquant*.py \
  --junitxml="$TQ_UNIFORM_ROOT/tests/regression_final.xml"
venv/bin/python scripts/llm/turboquant/evaluate_uniform_k8.py --root "$TQ_UNIFORM_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_uniform_k8.py probe-build --root "$TQ_UNIFORM_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_uniform_k8.py probe-run --root "$TQ_UNIFORM_ROOT"
venv/bin/python scripts/llm/turboquant/report_uniform_k8.py probes --root "$TQ_UNIFORM_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_uniform_k8.py build --root "$TQ_UNIFORM_ROOT"
venv/bin/python scripts/llm/turboquant/report_uniform_k8.py audit --root "$TQ_UNIFORM_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_uniform_k8.py evaluate --root "$TQ_UNIFORM_ROOT"
venv/bin/python scripts/llm/turboquant/report_uniform_k8.py summary --root "$TQ_UNIFORM_ROOT"
venv/bin/python -m pytest -q src/qai_hub_models/test/test_models/test_turboquant*.py \
  --junitxml="$TQ_UNIFORM_ROOT/tests/regression_release.xml"
venv/bin/python scripts/llm/turboquant/verify_uniform_k8.py final --root "$TQ_UNIFORM_ROOT"
```

Core additions are `turboquant/uniform_k8.py`, opt-in config/reference/export
routing and `Decode8Uniform` in the Native package. Existing Structured/LM
defaults and constants are not edited. `test_turboquant_uniform_k8.py` adds
12 cases; the existing profile-identity test now recognizes Uniform's explicit
arithmetic identity. The five `*_uniform_k8.py` tools cover optimization,
protocol/build/measurement, offline errors, reporting and verification.
`protocol.json`, `data_manifest.json`, `source_identity.json`, `environment.json`
and `final_verification.json` retain parameters, data/binary/source hashes and
preservation checks.

## Lloyd–Max K8 convergence versus Structured K8 (opt-in, 2026-10-20)

Branch `exp/lloyd-max-k8-convergence`, based on
`exp/structured-k8-nll-allocation` commit `cd8a3d509`. No default is promoted,
no weights/activation grids are retrained or recalibrated, and no automatic
commit is made. Prior constants, binaries and experiment results are retained.
Artifacts: `/mnt/d/ai-hub-models/binaries/turboquant/lloyd_max_k8_convergence_20261020`.

The three full-model configurations have the same 14 odd K8 layers, 14 even
K4 layers and V2 on all 28 layers. Dense K42/V542, LM-tree K4/V2, current/past
compression, FP16 effective scales, norm correction, tile256, attention and
W4A16 parameters are fixed. Payload plus scales remains **28.875 MiB** at CL1024.

| Group | Profile | K8 encoder | K8 decoder |
|---|---|---|---|
| LM8-100 | `mixed_k6_v2_uniform` | unchanged frozen 100-iteration LM | Native LUT |
| LM8-Converged | `mixed_k6_v2_uniform_lm8_converged` | same tree/centroid lookup, new constants | same Native LUT |
| Structured K8 | `mixed_k6_v2_uniform_structured_k8` | unchanged two-region arithmetic | unchanged opt-in HVX arithmetic |

`KVCodecSpec.codebook_variant` and `TurboQuantConfig.converged_k8` are opt-in
identities. Their default values are omitted from serialization; all existing
profile hashes remain unchanged. K8 tables, encoder boundaries and norm
correction constants come from the same frozen codebook. Different codebooks
have incompatible cache hashes. `lm8_converged_v4_scaled` is an isolated K8
probe profile, **not** the full model's V2 allocation.

### FP64 convergence protocol and scalar results

`optimize_lloyd_max_k8.py` minimizes the analytically integrated Gaussian
scalar reconstruction MSE for `N(0,1/128)`. It does not fit Structured
centroids or use KV, attention, calibration NLL or test PPL for optimization.
Symmetric positive-half FP64 Lloyd updates use survival-function differences
to avoid upper-tail CDF cancellation. Four deterministic initializations are
fixed before evaluation: frozen LM8-100, Gaussian quantiles, uniform 4-sigma,
and high-rate Gaussian companding. There is no random sample seed.

Stop after **10 consecutive** iterations with maximum centroid movement
`<= 1e-10 sigma` **and** relative MSE change `<= 1e-11`; maximum `200000`
iterations. Choose the lowest final FP64 MSE among converged runs. No global
optimality claim is made. Each CSV retains every iteration's MSE, maximum
movement, relative MSE change and stability counter.

| Initialization | Additional iterations | Converged |
|---|---:|---|
| Frozen LM8-100 | 92,584 | yes |
| Gaussian quantiles | 92,684 | yes |
| Uniform 4-sigma | 93,288 | yes |
| High-rate companding | 62,495 | yes |

The selected run is `uniform_4sigma`. All four runs produce the **same FP16
table**; their tiny final FP64 MSE differences should not be interpreted as
meaningful initialization superiority. Curves are in
`figures/lloyd_convergence_nonnegative.{png,svg}`. The earlier plot with
automatic negative-axis padding is preserved separately.

| Scalar Gaussian MSE | LM8-100 | LM8-Converged | Structured K8 |
|---|---:|---:|---:|
| Nearest-centroid integrated MSE | 1.3054680e-6 | 3.2175846e-7 | 4.7935449e-7 |
| After FP16 centroid rounding, nearest rule | 1.3057813e-6 | 3.2192792e-7 | 4.7935449e-7 |

Convergence reduces the legacy scalar MSE by **75.35%**. Structured's
nearest-centroid MSE is about **49% higher** than converged LM; its deployed
arithmetic index rule and end-to-end quality are evaluated separately. The
old 100-iteration table must not be described as a fully converged LM8 optimum.
An independent piecewise numerical integral agrees with the analytic
converged MSE to `6.8e-12` relative error. Re-executing the pinned historical
100-step generator also reproduces its frozen FP64 and FP16 tables exactly.
The converged FP16 LUT SHA256 is
`c8d5945cbc5197a905e00ae1c70f5834d2a0122260ff9bb58e55a0fbcd44ae6b`;
FP64/FP32-boundary/FP16-boundary identities are retained alongside it in
`codebook/optimization.json` and the opt-in configuration metadata.

### Captured KV/QK and numerical validation

Offline errors use the same two historical W4A16 FP16-KV validation captures,
all 28 layers and 32 fixed causal query positions/document. Query already
contains attention scaling; GQA repeats each KV head twice. No captured
data selects coefficients. The emulator applies FP16 operation boundaries,
including norm correction, stored scale and restored product, but uses FP32
reductions; it is **not HTP bit-exact**.

| K8-layer metric only | LM8-100 | LM8-Converged | Structured K8 |
|---|---:|---:|---:|
| K reconstruction MSE, original coordinates | 0.0021052714 | 0.0006278402 | 0.0009215808 |
| QK score MSE, original FP16-KV reference | 0.0061150458 | 0.0016754761 | 0.0024291137 |

All-layer and rotated-reference results are retained in `offline_quality.json`.
Gaussian scalar MSE, norm-corrected vector MSE, QK score MSE and full-model NLL
are different objectives and are not substituted for each other.

An important existing rounding limitation is explicitly checked: the LM
encoder's affine-pair FP16 interpolation is not a literal LUT lookup. Of
256 indices, **128 / 116** differ from the direct FP16 decoder LUT for
LM8-100 / converged LM, with maximum absolute centroid differences
`0.00146484375 / 0.003173828125`. These are executed arithmetic-rounding
differences, **not mismatched source codebook hashes**. Structured's arithmetic
centroids match its LUT exactly. Changing the LM encoder algorithm to remove
this discrepancy would violate this experiment's constants-only comparison;
it is not silently fixed or claimed bit-exact here.

S26 Ultra isolated encoder checks retain the existing scale threshold `0.2%`
and boundary-distance threshold `0.001`. No unexplained index mismatch or
missing HTP-profiled operator is observed.

| Maximum scale relative error | AR1 | AR128 |
|---|---:|---:|
| LM8-100 K | 0.13264% (pass) | 0.25699% (fail) |
| Converged LM8 K | 0.11189% (pass) | 0.19189% (pass) |
| Structured K8 K | 0.11062% (pass) | 0.17255% (pass) |
| Common V4 standalone-probe control | 0.16533% (pass) | 0.24766% (fail) |

The common V4 probe is not the V2 full-model path. Its historical failure is
preserved and means **not all standalone encoder graphs pass**. All 16
isolated decoder probes (three requested decoders plus Structured LUT control,
each at 1/3/256/1023 tokens) are FP16 **bit-exact**, including all byte codes,
edge scales and partial tiles. The extra Structured LUT control has identical
inputs and outputs to the arithmetic decoder and isolates decoder cost without
introducing a fourth full-model group.

### Frozen evaluation data and measurement scope

Full-model scoring uses the unchanged four legacy WikiText windows plus four
new held-out test documents: **Hannah Dodd**, **Zrinski Battalion**,
**Temple Beth Israel (Eugene, Oregon)**, and **Oxaziridine**. Documents are
chosen by fixed salted content-hash order, not model outcomes. Previously
recorded experiment titles and any document sharing a 16-token span with
prior inputs are excluded. One 1024-token window per document, offset
`min(256, document_length-1024)`, yields 1023 scored targets. Each split's
overall PPL is `exp(total_NLL/4092)`, never the arithmetic mean of window PPLs.
This is independence from recorded experiments, not exclusion from original
model pretraining. Dataset revision, tokenizer, offsets and token hashes are
retained in `data_manifest.json` and `assets/assets.json`.

Performance is one unprofiled **897+128** session/group on S26 Ultra,
CL1024, burst policy. Execution order is Structured, LM8-100, converged LM
to reuse existing binaries during the new build. Reset diagnostics and
codec profiling are separate and are not repeated performance samples.
No variance, statistical superiority or counterbalanced-order claim is made.

### Full-model results

| Quality metric | LM8-100 | LM8-Converged | Structured K8 |
|---|---:|---:|---:|
| Legacy WikiText mean NLL | 3.17736437 | 3.14386415 | 3.16176300 |
| Legacy WikiText PPL | 23.98346 | **23.19332** | 23.61219 |
| New held-out mean NLL | 2.79712467 | 2.78602886 | 2.79758190 |
| New held-out PPL | 16.39743 | **16.21649** | 16.40493 |

| Single 897+128 session | LM8-100 | LM8-Converged | Structured K8 |
|---|---:|---:|---:|
| TTFT (ms, model load excluded) | 753.954 | 786.002 | 539.143 |
| Prefill (tok/s) | 1190.344 | 1142.646 | 1666.205 |
| Decode (tok/s) | 32.920 | 32.867 | 37.169 |
| Allocated host KV (MiB) | 28.875 | 28.875 | 28.875 |
| I/O buffers (MiB) | 96.929 | 96.929 | 96.929 |
| End VmRSS (MiB) | 153.867 | 154.223 | 151.070 |
| Process VmHWM (MiB) | 605.559 | 605.832 | 606.059 |
| Four context binaries (MiB) | 1635.695 | 1635.566 | 1621.328 |

**Interpretation:** converging LM8 lowers legacy PPL by **3.29%** and
new held-out PPL by **1.10%** versus LM8-100. Legacy NLL improves on all
four windows; held-out NLL improves on three of four documents and worsens
on Zrinski Battalion. This is an observed aggregate quality improvement,
not a universal per-document guarantee. Gaussian MSE's 75.35% reduction
does not translate to a comparable percentage PPL reduction.

Converged LM's measured TTFT is **4.25% higher**, with decode **0.16% lower**
than LM8-100. A constants-only change does not remove the expensive LM8
encoder structure. There is **no observed speed improvement** and one
observation does not establish a systematic slowdown.

Structured retains the latency advantage in this run: compared with
converged LM its TTFT is **31.41% lower** and decode **13.09% higher**, but
its legacy and held-out PPL are **1.81% / 1.16% higher**. The proper conclusion
is a quality/latency trade-off, not that Structured beats fully converged LM
in Gaussian MSE or overall quality. Converged LM wins three of four legacy
windows against Structured; on held-out documents they win two each.
These small test sets and single performance observations do not establish
statistical or broad-domain superiority. Defaults remain unchanged.

`comparison.json` / `.csv` contain the complete aggregate results, configuration
and binary hashes; `document_nll.csv` retains all 24 window scores.
`codec_costs.json` keeps named-op encoder cycle counters separate from physical
decoder HVX busy cycles and SDK execute times. Busy-cycle percentages are
**not TTFT or end-to-end latency reductions**. LM100 versus converged LM
isolates codebook constants. Structured versus LM changes both quantizer and
decoder; the separate same-codebook Structured decoder probes are necessary
for decoder-only attribution.

### Isolated codec costs

The following AR128 encoder measurements use the same captured layer-0 K
input. Stage columns are **named-op cycle counters**, not elapsed time and
not mutually exclusive physical HVX occupancy. The SDK execute column is
elapsed accelerator time from the separate instrumented probe.

| K8 encoder | Normalize cycles | Rotation cycles | Scalar-index cycles | Pack/output cycles | Norm/scale cycles | SDK execute (us) |
|---|---:|---:|---:|---:|---:|---:|
| LM8-100 | 1,214,365 | 34,111 | 16,218,775 | 34,027 | 12,813,320 | 3,203 |
| LM8-Converged | 1,157,135 | 35,570 | 15,666,367 | 42,710 | 12,195,259 | 3,187 |
| Structured K8 | 1,119,549 | 15,917 | 1,146,338 | 308,389 | 822,695 | 1,523 |

AR1 scalar-index / norm-scale counters are respectively
`752434 / 529539`, `774935 / 561896`, and `89799 / 45922`; SDK execute times
are `1480 / 1472 / 1328 us`. LM convergence changes constants, not tree depth
or centroid-lookup structure. Both LM variants therefore retain the expensive
8-bit index and norm-correction lookup. All groups store one UINT8 per K8
coordinate: a changed pack/output counter is not a changed packing algorithm.

For one **256-token** Native decoder invocation, using identical index and
scale inputs:

| Decoder | Physical HVX busy cycles | HVX interval-union cycles | SDK execute (us) | SDK execute excluding wait (us) |
|---|---:|---:|---:|---:|
| LM8-100 LUT | 536,340 | 90,389 | 4,318 | 678 |
| LM8-Converged LUT | 583,426 | 98,146 | 4,491 | 758 |
| Structured arithmetic | 358,972 | 60,605 | 4,429 | 653 |
| Structured LUT control | 593,957 | 100,201 | 4,469 | 769 |

Only the last two rows isolate the decoder algorithm with the **same
codebook and bit-exact outputs**. Arithmetic records 39.56% fewer HVX busy
cycles in this probe; this is not a 39.56% whole-model decode improvement.
Wait/launch costs, probe power policy, profiling overhead and a single sample
preclude converting these counters into a full-model speedup. No Native
kernel was changed in this convergence experiment.

### Reproduction

Use a **new output directory**. Existing reports, input data, performance
logs and codebook outputs are never replaced by the experiment harness.
The original failed decoder build is retained as
`decoder_lm8_100_failed_python_resolution`: no HTP measurement occurred in
that attempt. A probe-tool fix now preserves the SDK venv interpreter path
instead of resolving its symlink to `/usr/bin/python`; this does not change
the model or Native kernel.

`benchmark_lm8_convergence.py audit` compares all six attention-part
AR1/AR128 ONNX graphs and quantized DLC operator contracts: only the K8
boundary/centroid constants and LUT identifier may differ. It checks all
255 boundaries, both FP16 LUTs, unchanged weight/activation encodings,
I/O, MatMul shape/dtype and remaining operator connectivity. An SDK-generated
RMS node name is normalized, not its operation or tensors. This does not
claim that backend scheduling or measured execution time is identical.
`verify_lm8_convergence.py` then audits frozen input hashes, all three bundles,
24 scoring windows, three performance sessions, decoder/encoder checks and
the 691-test regression report without executing the device again.

One preservation issue is documented in
`runtime_manifest_preservation_note.json`: the first Structured push
regenerated an inherited runtime-manifest symlink with **identical bytes**,
changing its host timestamp. The prior device copy and host SHA256 match;
no prior binary or result content changed. The experiment now uses a private
runtime manifest, and the harness refuses inherited runtime-manifest symlinks.

Implementation changes are limited to opt-in config/reference/export/Native
table routing and the frozen `turboquant/lloyd_max_k8.py` constants. No existing
codebook constants or Native C++ kernels are edited. New experiment tools are
`optimize_lloyd_max_k8.py`, `benchmark_lm8_convergence.py`,
`evaluate_lm8_quality.py`, `probe_lm8_convergence.py`,
`summarize_lm8_convergence.py`, and `verify_lm8_convergence.py`; the isolated
probe CLIs accept the new profile/table, and
`test_turboquant_lm8_convergence.py` adds eight regression cases.
`final_verification.json` records the exact changed-file and report hashes.

```bash
TQ_LM_ROOT=/mnt/d/ai-hub-models/binaries/turboquant/lloyd_max_k8_convergence_repro
venv/bin/python scripts/llm/turboquant/optimize_lloyd_max_k8.py --out "$TQ_LM_ROOT/codebook"
venv/bin/python scripts/llm/turboquant/benchmark_lm8_convergence.py prepare --root "$TQ_LM_ROOT" \
  --data-manifest /mnt/d/ai-hub-models/binaries/turboquant/lloyd_max_k8_convergence_20261020/data_manifest.json
venv/bin/python scripts/llm/turboquant/evaluate_lm8_quality.py --root "$TQ_LM_ROOT"
venv/bin/python scripts/llm/turboquant/probe_lm8_convergence.py build --root "$TQ_LM_ROOT"
venv/bin/python scripts/llm/turboquant/probe_lm8_convergence.py run --root "$TQ_LM_ROOT"
venv/bin/python scripts/llm/turboquant/summarize_lm8_convergence.py probes --root "$TQ_LM_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_lm8_convergence.py build --root "$TQ_LM_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_lm8_convergence.py audit --root "$TQ_LM_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_lm8_convergence.py evaluate --root "$TQ_LM_ROOT"
venv/bin/python scripts/llm/turboquant/summarize_lm8_convergence.py results --root "$TQ_LM_ROOT"
venv/bin/python -m pytest -q src/qai_hub_models/test/test_models/test_turboquant*.py \
  --junitxml="$TQ_LM_ROOT/tests/regression_final.xml"
venv/bin/python scripts/llm/turboquant/verify_lm8_convergence.py --root "$TQ_LM_ROOT"
```

## Structured K8 NLL layer allocation (opt-in, 2026-10-13)

Branch `exp/structured-k8-nll-allocation`, based on
`exp/structured-k8-decoder` commit `fe744cddb`. This experiment does not
change any default, train weights, recalibrate activations, change a codebook,
or modify the Native package. No automatic commit.

The new `mixed_k6_v2_structured_k8_allocation` profile requires an explicit
layer-plan JSON. `structured_allocation_plan(k8_layers)` creates a canonical
plan containing the full configuration and its hash. Exactly 14 distinct
integer K8 layers in `[0,27]` are required; the other 14 use K4 and all 28 use
V2. Loading a plan with inconsistent bits, configuration, codebook or hash
fails. Cache state from a different allocation is rejected. K8 uses the
existing Structured quantizer and opt-in HVX arithmetic decoder; K4/V2 use
the existing LM tree and Native decoders. Dense K42/V542, tile256, GQA,
current/past compression, FP16 effective scales and W4A16 weights are unchanged.

At CL1024 the real host allocation is **30,277,632 bytes = 28.875 MiB**:
28 MiB of indices plus 0.875 MiB of FP16 scales. The average **index** budget
is K6/V2 (8 bits combined), not eight bits including scale metadata.

| Static allocation | K8 layers, zero-based |
|---|---|
| Uniform-Odd | 1,3,...,27 |
| Uniform-Even | 0,2,...,26 |
| Early | 0..13 |
| Late | 14..27 |
| Middle | 7..20 |
| Boundary | 0..6 and 21..27 |

### Frozen search and data protocol

Artifacts are kept outside the repository at
`/mnt/d/ai-hub-models/binaries/turboquant/structured_k8_nll_allocation_20261013`.
`protocol.json` and `data_manifest.json` freeze the inputs and hashes before
selection. Four allocation-calibration documents are reused from the previous
rotation experiment's validation manifest: Éva Gauthier, Die Another Day,
Battle of the Hongorai River, and Exit Through the Kwik-E-Mart. These are
**not activation calibration**. Final quality uses the unchanged legacy four
WikiText windows, excluded from this search. No contiguous 16-token span is
shared between the two input sets. Each window has 1024 tokens and 1023 scored
targets. Overall PPL is `exp(sum(NLL_sum) / 4092)`, not the mean of window PPLs.
The test windows have appeared in earlier experiments; they are not a newly
unseen generalization benchmark.

`allocation_offline_model.py` replays the actual exported W4A16 weights and
fixed activation/parameter quantization grids on CUDA. It does not substitute
an unquantized Hugging Face model. Full-model teacher-forced logits are
recomputed for each candidate. The KV codec reproduces the existing CPU
reference's LM/Structured index decisions, norm correction and FP16 rounding;
attention follows the AR128 mask/chunk geometry and tile256 accumulation.
However, GEMM/reduction arithmetic and quantized-grid emulation are **not
bit-exact HTP**. GPU use here is an offline search tool, not a CPU/GPU fallback
in the deployed HTP graph.

Search starts at Uniform-Odd. Each of at most **three iterations** evaluates
28 single flips, ranks four promotions and four demotions by full-model
calibration NLL, and evaluates their 16 swaps. Only an improvement larger
than `1e-6` mean NLL is accepted; ties use `(NLL, promotion, demotion)`.
Single flips temporarily use 13/15 K8 layers only for offline sensitivity;
every swap and exported profile retains exactly 14. Scores are cached by
mask and immutable evaluator identity. This is bounded greedy search, not a
proof of an optimal layer allocation. Attention MSE is not the objective.

The protocol also fixes an **HTP calibration acceptance gate**: evaluate the
final offline candidate and Uniform-Odd on the same four calibration inputs,
then retain the candidate only if actual HTP mean NLL improves by more than
`1e-6`. Otherwise retain Uniform-Odd. Freeze `selection.json` before any test
scoring. Static allocation test results cannot change that selection.

Offline search history (four calibration windows; these are **not HTP NLLs**):

| Step | Promote K4→K8 | Demote K8→K4 | Mean NLL |
|---|---:|---:|---:|
| Initial Uniform-Odd | — | — | 3.52375509 |
| 1 | 0 | 11 | 3.26225743 |
| 2 | 4 | 19 | 3.24126212 |
| 3 | 2 | 25 | 3.22837518 |

The proposed mask is `[0,1,2,3,4,5,7,9,13,15,17,21,23,27]`.
The offline mean NLLs of the six static masks are respectively
`3.52375509 / 3.24887410 / 3.23081409 / 3.62499426 / 3.58707253 / 3.23703912`.
The prior Attention-MSE mask `[10,15,16,17,18,19,20,21,22,23,24,25,26,27]`,
replayed with this experiment's Structured K8 codec, scores `3.60810823`
offline. This last replay is a mask comparison, not an eighth HTP benchmark.
The raw history, all single-flip/swapped-mask document NLLs, evaluator identity
and proposal are retained under `offline/` and `plans/nll_candidate.json`.
The exact evaluator sources used for selection are archived under
`offline/source_used/` and match `offline/identity.json`. After scoring finished,
only the standard repository license header was added to the working copies;
their parsed Python ASTs are identical. `source_archive_note.json` records this
non-semantic change. Resume checks remain fail-closed on source-byte changes;
use a fresh directory for reproduction with the final sources.

HTP calibration accepted this proposal **before test evaluation**:

| Calibration mean NLL, same four documents | Uniform-Odd | Uniform-Even diagnostic | NLL-selected |
|---|---:|---:|---:|
| Offline W4A16 surrogate | 3.52375509 | 3.24887410 | 3.22837518 |
| Actual full-model HTP | 3.60778174 | 3.27215053 | 3.23615315 |

The selected mask lowers HTP calibration mean NLL by `0.37162858` versus
Odd and by `0.03599738` versus Even. All four calibration documents improve
versus Odd; their mean NLL deltas are
`-0.92127229 / -0.28701896 / -0.19416345 / -0.08405963`.
The first document supplies about 62% of the total calibration gain, so it
would be inappropriate to describe this small selection set as broad
generalization evidence. The observed candidate improvement transfers to
HTP, but the emulator's error magnitude remains allocation-dependent.
`selection.json` and `plans/nll_selected.json` freeze the chosen mask; no
static test result can revise it.

Full configuration hashes (each JSON also embeds its codebook/rotation identity):

| Plan | SHA-256 |
|---|---|
| Uniform-Odd | `f7e7fa3ce4ea8e73547beb80f5a6a04ac78485a17c55832938e1cba026d985b5` |
| Uniform-Even | `01226307def7cd6780398969d690cb93cc6652268da474f139c9f4243f7e01db` |
| Early | `533a9bf507cc4bf6d650b42ab110afca84f2e739014c01680d81708e4883baea` |
| Late | `07d6a9f3ae1dd173eac74fea7e36dabcadebc972cb3b150e316f9bd5d74624ee` |
| Middle | `02545f3a7e8307291e782742327713cbbbb2f55bbf485f6603fdfcee982fa038` |
| Boundary | `c6e905f027b370548fba9ca92e51abd627d170e363573c9f767d473492039d60` |
| NLL proposal | `ccf1f776ed92e0e56239e94b1373ba35fc9a1e31a67fcce72749afbe8fdcfc42` |

### Fidelity and validation limits

The emulator's first-layer Q/K/V are close to an existing actual FP16-KV HTP
capture (relative MSE approximately `2.08e-6 / 7.86e-7 / 4.54e-5`), but this
does not establish full-model equivalence. On Uniform-Odd calibration, the
offline-minus-HTP document mean NLL ranges from **-0.553 to +0.139** and mean
`KL(HTP || offline)` from **0.187 to 0.518**. Therefore the offline sensitivities
are explicitly surrogate results, not measured HTP sensitivities. The final
candidate must pass the independent HTP acceptance gate described above.
Accumulation/fusion and error propagation can contribute to this discrepancy;
these checks do not isolate a unique cause.

CPU regression including the new allocation/search tests: **681 passed**,
two existing SWIG warnings. Tests cover exact count/range/type/hash checks,
cache incompatibility, storage accounting, greedy stopping/budget rules and
codec agreement with the existing reference. Export audits compare all
non-Key-codec ONNX nodes/constants and weight encodings, then inspect compiled
rotations, FP16 boundaries, per-layer decoder types and cache I/O. The old
encoder AR128 scale-gate failures remain separate known limitations; no
tolerance is relaxed and no new claim of encoder bit-exactness is made.
Ruff and `git diff --check` pass. Targeted mypy on the changed configuration
and two new test files passes with `--follow-imports=silent`. A dependency-aware
check also exposed existing tuple/list typing errors in unchanged
`export.py:585,593` (present in the base commit); this is not reported as a
clean repository-wide mypy run.

The runner only gains optional teacher-forced logit dumping for KL analysis.
The no-dump performance path executes the same graphs and is measured once
per unique configuration, unprofiled, with 897 input + 128 generated tokens.
Test KL is `KL(Uniform-Odd || candidate)`, including per-target P95/P99; it is
not a divergence from an FP16 baseline. Single observations do not establish
statistical speed superiority.

Each score dump is 621,722,112 bytes. To avoid filling device storage,
`release_allocation_device_logits.py` can remove this experiment's exact
device-side logit filenames **only after host/device SHA-256 agreement**.
All host dumps under `/mnt/d`, reports, receipts and model binaries remain.
The device copy can be restored with `adb push`. Run archival cleanup between
measurements, not concurrently with a timed performance run.

### Fixed test and single-run performance results

S26 Ultra (SM-S948N / SM8850), Qwen3-1.7B W4A16, CL1024. Quality is the
same four legacy WikiText windows (4092 scored targets). Performance is one
unprofiled 897-input / 128-generated-token run per configuration, in table
order. TTFT excludes model/context loading; decode throughput uses the
existing runner's 127 post-first-token steps. Score-mode logit dumping is
separate from the no-dump performance runs.

| Allocation | Test mean NLL | Test PPL | PPL reduction vs Odd | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| Uniform-Odd | 3.161763 | 23.612187 | 0.000% | 499.132 | 1798.643 | 37.272 |
| Uniform-Even | 3.043824 | 20.985342 | 11.125% | 493.144 | 1820.514 | 37.350 |
| Early | 3.010394 | 20.295398 | 14.047% | 497.889 | 1803.033 | 37.402 |
| Late | 3.321210 | 27.693837 | -17.286% | 496.150 | 1809.453 | 37.409 |
| Middle | 3.290510 | 26.856561 | -13.740% | 530.565 | 1693.198 | 37.327 |
| Boundary | 3.004137 | 20.168807 | 14.583% | 535.336 | 1678.634 | 37.243 |
| NLL-selected | 3.010138 | 20.290203 | 14.069% | 496.275 | 1809.032 | 37.281 |

**The NLL-selected mask improves over Uniform-Odd, but is not the best test
mask among these seven.** Its test mean NLL decreases by `0.15162482` and PPL
by 14.069%. Boundary has 0.60% lower PPL than the selected mask, and Early is
almost identical (20.295398 versus 20.290203). This supports transfer of the
selected candidate's gain over its initialization, not superiority of greedy
NLL selection over every static policy. No Boundary result was fed back into
selection and neither mask was promoted to a default.

The selected mask improves all four test windows over Odd:

| Window | Selected NLL | Selected PPL | Selected − Odd NLL | Selected − Boundary NLL |
|---|---:|---:|---:|---:|
| w0 | 2.485617 | 12.008527 | -0.058773 | -0.018375 |
| w1 | 3.236655 | 25.448456 | -0.191913 | +0.005243 |
| w2 | 3.002939 | 20.144649 | -0.178128 | +0.028282 |
| w3 | 3.315342 | 27.531807 | -0.177684 | +0.008854 |

The early-protecting masks outperform Odd in this test, while Late and Middle
worsen it. The initial offline single-flip diagnostic identifies layer 0 as
the largest promotion benefit (`-0.255798` mean NLL versus at most `-0.020932`
for the other initial promotions). These observations suggest that early K
protection matters here, but neither isolate layer 0's HTP causal contribution
nor establish a universal rule for other models or data. The three accepted
swaps promote 0/4/2 and demote 11/19/25; sensitivities are recomputed each round.

The selected run's TTFT changes by -0.57% and decode throughput by +0.024%
versus Odd: essentially unchanged **in these single observations**. Middle
and Boundary have higher measured TTFT; this is preserved in the table, not
normalized away. No repeat-based variance estimate, statistical speedup or
causal compiler/thermal explanation is claimed. Uniform-Odd's PPL exactly
reproduces the preceding arithmetic-decoder experiment.

Every configuration has **28.875 MiB actual host KV storage** and
**96.929 MiB runtime I/O buffers**. There is no per-token allocation metadata.
Other memory figures are distinct from KV storage:

| Allocation | End VmRSS (MiB) | Process VmHWM (MiB) | Four context binaries (MiB) |
|---|---:|---:|---:|
| Uniform-Odd | 150.125 | 604.863 | 1621.328 |
| Uniform-Even | 150.012 | 605.332 | 1621.301 |
| Early | 146.926 | 602.949 | 1621.160 |
| Late | 150.082 | 605.176 | 1621.234 |
| Middle | 150.816 | 605.734 | 1621.195 |
| Boundary | 150.613 | 605.387 | 1621.332 |
| NLL-selected | 150.156 | 605.285 | 1621.344 |

Logit divergence over all 4092 test targets, in nats:

| Allocation | Mean KL(Odd ∥ allocation) | P95 | P99 |
|---|---:|---:|---:|
| Uniform-Odd | 0 | 0 | 0 |
| Uniform-Even | 0.429715 | 1.678416 | 4.486211 |
| Early | 0.471868 | 1.911581 | 4.954686 |
| Late | 0.336313 | 1.295130 | 3.267350 |
| Middle | 0.311449 | 1.229007 | 3.235573 |
| Boundary | 0.458332 | 1.773342 | 5.044264 |
| NLL-selected | 0.451241 | 1.778218 | 5.214410 |

Odd is itself quantized, not a quality oracle: smaller KL to Odd does not
imply lower NLL. Late/Middle demonstrate this distinction. The saved logit
rows are independently checked against the exact teacher-forced target IDs;
their recomputed mean NLL must agree with the runner within `1e-6`.

Historical same-window quality references, **not new device measurements**:

| Earlier configuration | PPL |
|---|---:|
| Uniform LM K6/V2 | 19.844793 |
| Mixed K6/V2, Uniform LM K8 | 23.983459 |
| Mixed K6/V2, Attention-MSE-selected LM K8 | 27.241766 |
| This NLL-selected Structured K8 | 20.290203 |

The selected mask remains **0.445410 PPL / 2.24% above** the earlier uniform
K6/V2 result at the same cache size. The earlier MSE-selected allocation is
worse, but its codebook and selection protocol differ; this comparison does
not establish general superiority of NLL selection over Attention MSE.
The same prior mask is additionally compared with Structured K8 in the
offline calibration table above, without an unrequested eighth HTP run.

### Files and retained artifacts

- `config.py`: opt-in allocation factory, exact quota/config/codebook hash checks.
- `convert_parts.py`, `verify_kv_boundary.py`, `verify_rotated_attention.py`:
  external-plan help and audit reconstruction for the new opt-in profile.
- `qnn_runner/src/main.cpp`: optional score-mode logit dump; no-dump timing path unchanged.
- `allocation_offline_model.py`, `search_nll_allocation.py`: exported W4A16
  surrogate, full-model NLL sensitivity, immutable cached scores and bounded swaps.
- `benchmark_nll_allocation.py`: frozen protocol, isolated builds, graph audits,
  serial HTP quality/performance and pre-test selection gate.
- `summarize_nll_allocation.py`: paired NLL/PPL, logit-alignment audit, KL and figures.
- `release_allocation_device_logits.py`: hash-verified archival cleanup of only
  this experiment's device logit duplicates; host originals are retained.
- `test_turboquant_allocation.py`, `test_turboquant_allocation_search.py`:
  21 new allocation/cache/search/reference tests; README: this report and commands.

The artifact root contains `comparison.json/csv`, `document_nll.csv`,
`logit_kl.json`, `layer_nll_sensitivity.csv`, all seven plans, `selection.json`,
the complete `offline/` search history and source archive, raw `reports/`,
model/codebook/configuration/binary provenance, and validation logs.
`figures/layer_masks.svg` and `figures/initial_nll_sensitivity.svg` visualize
all masks and the initial per-layer NLL sensitivity; PNG counterparts are
included. All older experiment directories remain intact. Device cleanup
receipts identify identical retained host copies; no model binaries are deleted.

### Reproduction

Run from the repository root with the existing Android/QAIRT environment.
Use a **new output directory**; scripts refuse to overwrite completed results.
The source artifacts and Native package are pinned by `benchmark_nll_allocation.py`.

```bash
export PYTHONPATH=src:scripts/llm/turboquant
RUN=/mnt/d/ai-hub-models/binaries/turboquant/structured_k8_nll_allocation_NEW
venv/bin/python scripts/llm/turboquant/benchmark_nll_allocation.py prepare --root "$RUN"
venv/bin/python scripts/llm/turboquant/benchmark_nll_allocation.py build --root "$RUN"
venv/bin/python scripts/llm/turboquant/benchmark_nll_allocation.py push --root "$RUN" --groups uniform_odd uniform_even
venv/bin/python scripts/llm/turboquant/benchmark_nll_allocation.py calibration --root "$RUN" --groups uniform_odd uniform_even
venv/bin/python scripts/llm/turboquant/release_allocation_device_logits.py --root "$RUN"
venv/bin/python scripts/llm/turboquant/search_nll_allocation.py preflight --root "$RUN"
venv/bin/python scripts/llm/turboquant/search_nll_allocation.py search --root "$RUN"
venv/bin/python scripts/llm/turboquant/search_nll_allocation.py static-offline --root "$RUN"
# For a novel candidate; reuse the matching static bundle if masks coincide.
venv/bin/python scripts/llm/turboquant/benchmark_nll_allocation.py build --root "$RUN" --groups nll_candidate
venv/bin/python scripts/llm/turboquant/benchmark_nll_allocation.py push --root "$RUN" --groups nll_candidate
venv/bin/python scripts/llm/turboquant/benchmark_nll_allocation.py calibration --root "$RUN" --groups nll_candidate
venv/bin/python scripts/llm/turboquant/release_allocation_device_logits.py --root "$RUN"
venv/bin/python scripts/llm/turboquant/benchmark_nll_allocation.py finalize --root "$RUN"
venv/bin/python scripts/llm/turboquant/benchmark_nll_allocation.py push --root "$RUN" --groups early late middle boundary
for GROUP in uniform_odd uniform_even early late middle boundary; do
  venv/bin/python scripts/llm/turboquant/benchmark_nll_allocation.py evaluate --root "$RUN" --groups "$GROUP"
  venv/bin/python scripts/llm/turboquant/release_allocation_device_logits.py --root "$RUN"
done
# If selection.json selected a novel nll_candidate, evaluate it once as well.
venv/bin/python scripts/llm/turboquant/benchmark_nll_allocation.py evaluate --root "$RUN" --groups nll_candidate
venv/bin/python scripts/llm/turboquant/release_allocation_device_logits.py --root "$RUN"
venv/bin/python scripts/llm/turboquant/summarize_nll_allocation.py --root "$RUN"
```

An arbitrary allocation can be generated without editing defaults:

```python
import json
from pathlib import Path
from qai_hub_models.models.templates.llm.turboquant.config import structured_allocation_plan

plan = structured_allocation_plan(list(range(0, 28, 2)))
with Path("my_allocation.json").open("x") as stream:
    json.dump(plan, stream, indent=2)
```

Pass it to `convert_parts.py --profile mixed_k6_v2_structured_k8_allocation
--layer-plan my_allocation.json --structured-k8-decoder arithmetic` alongside
the ordinary split/output/Native-package options. The allocation profile has
no implicit mask; `get_profile()` rejects it without its frozen JSON.

## Structured K8 arithmetic decoder (opt-in, 2026-10-13)

Branch `exp/structured-k8-decoder`, based on `exp/structured-k8-codebook`
commit `246a53417be071ecfc97b7a8b4f23e4c3013ce28`. No automatic commit or
default promotion. **The default remains the existing HVX LUT decoder.**

`--structured-k8-decoder arithmetic` is an export-only option on
`convert_parts.py`. It does not change the profile, quantizer, codebook, cache
format or configuration hash. Both groups use
`mixed_k6_v2_uniform_structured_k8`: K8 on odd layers 1,3,...,27, K4 on the
other 14 layers, V2 throughout, Dense K42/V542, current/past compression,
FP16 effective scales, unchanged W4A16 weights/calibration and tile256 attention.
K4/V2 decoders and every encoder/attention node remain unchanged.

### Decode8Structured

The new QHPI op takes packed UINT8 indices, FP16 scales, and **four FP16
constants** `[offset0, step0, offset1, step1]` from `structured_k8.py` (8 bytes,
not a 256-entry centroid LUT). Existing `Decode8` remains available.

```text
positive = byte >= 128
magnitude = positive ? byte - 128 : 127 - byte
region = magnitude >> 6
local = magnitude & 63
centroid = FP16(FP16(local * step[region]) + offset[region])
signed_centroid = positive ? centroid : -centroid
output = FP16(signed_centroid * stored_effective_scale)
```

Negative region/local fields are complemented in the existing ascending-byte
ABI; they must not be interpreted as unreflected magnitude bits. On HVX,
unsigned halfword conversion represents local0..63 exactly, followed by
separate FP16-result multiply/add intrinsics, sign-bit XOR and FP16 scale
multiply. The generated V81 instructions use qfloat arithmetic with explicit
FP16 conversion at each required rounding boundary; these are not fused FMA.
No FMA, LUT lookup, 256-entry table staging or scalar coordinate loop. A row
reads 128 index bytes and one scale, and writes two 128-byte FP16 vectors.
Coefficient splats are outside the row loop. QHPI slices are multithreaded on
HVX; no CPU inference fallback. The host library's scalar implementation is
for SDK preparation/reference only.

The export pass replaces only the current/past Structured K8 Native nodes,
checks the old LUT identity, and swaps that initializer for the four constants.
Arithmetic export rejects ordinary LM K8 and incompatible Native packages.
Reusing an output bundle with a different decoder selection is rejected even
though bit-exact cache compatibility allows both to retain the same config hash.

### Numerical verification

- CPU regression: **660 passed**, two existing SWIG warnings.
- Hexagon libnative: old 2..8-bit tests pass; arithmetic and LUT agree for
  all byte codes, region/sign endpoints, signed zero, subnormal/normal scale
  edges, negative finite scales, maximum finite scales, unaligned I/O,
  zero/odd slice lengths, and guard pages.
- S26 Ultra HTP: both decoders pass **1/3/256/1023-token** probes, eight KV
  heads x 128 coordinates. All eight outputs match the independent NumPy
  FP16 oracle bit-for-bit, and the paired device output files are identical.
  Input/scale hashes also match. No numeric tolerance was relaxed.
- Current/past KV, GQA and attention oracle tests cover the arithmetic path.
  Strict ONNX arithmetic reference tests use `ReferenceEvaluator`: the
  initial ORT CPU test showed one-ULP differences, whereas literal ONNX
  evaluation and the actual HTP kernel match the required rounding contract.
  This CPU test-backend issue did not change the deployed kernel or encoder.
- All six KV graphs pass the decoder-only audit: unchanged ONNX nodes and
  constants except the K8 decoder/LUT replacement, identical encodings and I/O.
  Converted DLC op counts, non-decoder op signatures, tensor dtypes/dimensions
  and parameters also match. Only row IDs and SDK-generated random RMS node
  labels are ignored. Non-KV part1 is shared byte-for-byte.

The encoder is unchanged. This work does not resolve or waive the historical
LM K8/companion V4 AR128 scale-gate failures recorded in the preceding experiment.

### Isolated HTP decoder diagnostic

One optrace execution per decoder/shape, identical synthetic indices and edge
scales, same Native package. Arithmetic ran first, then LUT.
HVX counts below are actual physical worker busy cycles for the decoder op,
not wall-clock microseconds. SDK execute times include graph/runtime overhead;
both total and excluding-wait values are retained. Probe power policy is
qnn-net-run's default, unlike the full-model burst policy.

| Tokens | LUT HVX cycles | Arithmetic HVX cycles | LUT execute (us) | Arithmetic execute (us) | LUT excluding wait (us) | Arithmetic excluding wait (us) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 365,470 | 108,968 | 4,340 | 4,823 | 608 | 514 |
| 3 | 378,288 | 97,616 | 4,299 | 4,258 | 586 | 479 |
| 256 | 588,263 | 330,065 | 4,482 | 4,374 | 748 | 662 |
| 1023 | 1,199,805 | 1,077,009 | 4,913 | 4,945 | 1,244 | 1,190 |

Physical work falls **43.9% at tile256**, but only **10.2% at 1023 tokens**.
The old LUT's table preparation/symmetry checks are per invocation; arithmetic
removes that setup while retaining the same input/output traffic. The smaller
large-shape gain is consistent with amortized setup and remaining arithmetic/
memory work, not proof of a uniquely identified memory bottleneck. Total SDK
execute time is worse at 1 and 1023 tokens despite lower HVX work, illustrating
why busy-cycle savings must not be equated with end-to-end latency savings.
The disassembly also shows that converting integer local indices to FP16
requires multiple vector instructions. Arithmetic removes table preparation
and lookup, not all decoder work. These are single observations without
variance or statistical claims.

### Full-model results

Qwen3-1.7B W4A16, S26 Ultra / SM8850 V81, QAIRT 2.48.0.260626, CL1024,
HTP burst. **One unprofiled 897-token prompt + 128-token generation per group**,
LUT first then arithmetic. The 127 decode steps use CL1024. Both measurements
are fresh and use the same new Native library, SHA256
`3e69cc4df58568307cee841f13c78092c6a490dc2ab57fc0111805550cd271a0`.
PPL uses the unchanged four WikiText windows, 4092 scored targets in total:
`exp(sum(window_nll_sum) / 4092)`, not the average of window PPLs.

| Metric | Structured K8 + LUT | Structured K8 + arithmetic | Observed change |
|---|---:|---:|---:|
| TTFT (ms) | 549.783 | 498.262 | -9.37% |
| Prefill (tok/s) | 1,634.550 | 1,801.763 | +10.23% |
| Decode (tok/s) | 35.533 | 37.284 | +4.93% |
| Decode (ms/token) | 28.143 | 26.821 | -4.70% |
| Mean decode QNN execute (ms) | 26.390 | 25.085 | -4.94% |
| Mean decode host KV prepare (ms) | 1.151 | 1.138 | -1.14% |
| PPL | 23.61218745 | 23.61218745 | identical |
| Host KV cache (MiB) | 28.875 | 28.875 | unchanged |
| I/O buffers (MiB) | 96.929 | 96.929 | unchanged |
| End VmRSS (MiB) | 151.309 | 150.191 | single observation |
| Process VmHWM (MiB) | 606.020 | 605.066 | single observation |
| Four context binaries (bytes) | 1,700,085,760 | 1,700,085,760 | unchanged |

All 128 generated IDs and all four reported NLL sums are identical between
groups. Each group passes the separate two-session reset check. Cache storage
is exactly **30,277,632 bytes = 28 MiB indices + 0.875 MiB FP16 scales**;
the decoder introduces no token metadata or new I/O buffers. Replacing the
512-byte centroid initializer with eight coefficient bytes does not change
the final context file sizes at the SDK's allocation granularity.

The observations support a decoder work reduction and a promising end-to-end
gain without measured quality loss, **not statistical superiority**. The
1.305 ms reduction in mean QNN time accounts for most of the observed 1.322 ms
decode-step reduction; it is not a claim that all of that difference is
causally attributable to this kernel. Device state and fixed execution order
are uncontrolled single-run limitations. Tile256 busy cycles fall 43.9%, but
only the K8 decoder in 14 layers changes: encoder, K4/V2 decode and attention
remain. That percentage is neither the TTFT reduction nor a full-model speedup.
The default remains LUT; arithmetic is preserved as an explicit opt-in.

### Reproduction and artifacts

Use a **fresh** artifact directory. Both full-model contexts use the exact same
new Native package: LUT reuses immutable old DLCs, while arithmetic rebuilds
parts2/3/4 and shares the non-KV part1 with LUT. This controls the package version
without replacing libraries in old bundles. The arithmetic parts use up to
three concurrent converters; device runs are sequential.

```bash
export PYTHONPATH=src
TQ_AD_ROOT=/mnt/d/ai-hub-models/binaries/turboquant/structured_k8_decoder_NEW
venv/bin/python scripts/llm/turboquant/native_decoder/build.py --out "$TQ_AD_ROOT/native" --test-hvx
for decoder in lut arithmetic; do
  TQ_DECODER_ARGS=()
  if [ "$decoder" = arithmetic ]; then TQ_DECODER_ARGS+=(--structured-arithmetic); fi
  venv/bin/python scripts/llm/turboquant/validate_native_decoder.py all \
    --work-dir "$TQ_AD_ROOT/probe_$decoder" --package "$TQ_AD_ROOT/native" \
    --bits 8 --lut structured_k8 --tokens 1 3 256 1023 \
    --scale-edges --strict-bit-exact --optrace "${TQ_DECODER_ARGS[@]}" \
    --device-dir "/data/local/tmp/qaihm_turboquant/${TQ_AD_ROOT##*/}_$decoder" || break
done
venv/bin/python scripts/llm/turboquant/benchmark_structured_decoder_once.py prepare --root "$TQ_AD_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_structured_decoder_once.py build-lut --root "$TQ_AD_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_structured_decoder_once.py build-arithmetic --root "$TQ_AD_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_structured_decoder_once.py assemble --root "$TQ_AD_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_structured_decoder_once.py evaluate --root "$TQ_AD_ROOT"
venv/bin/python scripts/llm/turboquant/benchmark_structured_decoder_once.py summarize --root "$TQ_AD_ROOT"
venv/bin/python scripts/llm/turboquant/profile_structured_decoder_probes.py --root "$TQ_AD_ROOT"
venv/bin/python -m pytest -q src/qai_hub_models/test/test_models/test_turboquant*.py
```

Actual root: `/mnt/d/ai-hub-models/binaries/turboquant/structured_k8_decoder_20261013`.
`protocol.json` freezes inputs/configuration/package/conditions;
`decoder_comparison.json`/`.csv` and `probe_profiles/` retain paired numerical
and physical probe results. `comparison.json`/`.csv`, `reports/`, and
`tests/cpu_regression.xml` retain full-model and regression results.
`reports/decoder_only_audit.json` checks the source and converted graphs;
`native/structured8_disassembly.txt` records the SIMD instructions.
`final_verification.json` records final source/artifact hashes and preservation
checks, including unchanged codebook/configuration and bit-exact paired output.

`schematic_recovery.json` records an SDK output-directory issue: initial
schematics landed in the caller's working directory. Arithmetic schematics
were regenerated on the host from unchanged inputs; all four regenerated
context binaries have **identical SHA256 to the preserved measured binaries**.
No additional device execution was used. Probe compilation now sets its working
directory explicitly so schematics cannot collide across decoder variants.

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
