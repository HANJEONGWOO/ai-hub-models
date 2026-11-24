# Relative attention output error for K/V allocation

This study evaluates four-state K/V allocation on Qwen3-1.7B W4A16 on an RTX 3090. The context is 1,024 tokens. Every reported configuration uses the deployed K4/V4 Lloyd–Max formats, K8 structured arithmetic format and V2 Lloyd–Max format as prescribed by its layer plan. Calibration, selection and test rules were [frozen before measurement](protocol.json).

## Allocation and validation results

| Allocation (mean K5/V3) | Sum of relative calibration errors | Validation NLL | Validation PPL |
| --- | ---: | ---: | ---: |
| Early | 1.729318 | 2.802460 | 16.4852 |
| Prior absolute-MSE allocation | 3.053707 | 3.111947 | 22.4647 |
| Four-state relative-error allocation | 1.584191 | 2.808654 | 16.5876 |

The new allocation reduced validation NLL by 0.303293 nats/token and PPL by 26.16% against the previous absolute-error mask. All eight validation articles improved. Against Early, four improved and four worsened, with pooled NLL 0.006194 higher. The absolute-error plan had placed all seven K8 layers at or after layer 20. The relative-error plan moved all seven to the first eight layers. It selected K8 at layers **0–5 and 7**, and V4 at **0–7, 11, 20, 23, 25–27**. Early places K8 at 0–6 and V4 at 0–13. All layer numbers are zero-based.

The validation result met the prespecified condition to explore all 15 budgets. Across their best masks, the local objective reached its minimum at seven K8 layers and fourteen V4 layers (1.584191), narrowly ahead of eight K8 and twelve V4 (1.585143). The seven-layer candidate was identical to the fixed candidate, so its existing validation measurement served both purposes. The fixed and expanded allocations produce the same deployment hash, `c115c763359abe6835ff67bedfd0815644883c26a1003da0a4a1ba43906ae75f`. Full-model NLL was measured for the three distinct validation masks.

## Independent test and resource cost

| Frozen allocation | Mean test NLL (nats/token) | Test PPL | Packed cache at CL1024 |
| --- | ---: | ---: | ---: |
| Early | 3.032509 | 20.7492 | 28.875 MiB |
| Four-state relative-error allocation | 3.039832 | 20.9017 | 28.875 MiB |

Four of eight test articles improved and four worsened. Relative allocation minus Early was **+0.007323 nats/token**, with a document-paired bootstrap 95% interval of **[−0.005412, +0.020269]** (10,000 replicates). Pooled PPL was 0.73% higher. The improved calibration objective recovered most of the prior validation deficit, but the independent test provides insufficient evidence of better quality than Early or of formal equivalence. The previous absolute-MSE allocation was a validation control. Each allocated cache contained 29,360,128 index bytes and 917,504 bytes of FP16 scales.

| Measured stage on RTX 3090 | Time |
| --- | ---: |
| Eight FP16-KV reference forwards including all four local replays | 28.49 s |
| Calibration invocation including setup, controls and result I/O | 107.12 s |
| Fixed DP, then all 15 quotas (includes plan serialization) | 0.059 s + 0.055 s |
| Validation scoring, cache checks and allocation decision (three masks × eight windows) | 71.10 s |
| Source checks, model load, planning and validation combined | 144.45 s |
| Independent test and summary (two masks × eight windows) | 45.79 s |

Calibration plus selection took 251.56 s across the two invocations. Including test gives 297.35 s. Data preparation and report export are excluded. Rows that include setup or other stages overlap. All fifteen quotas reused the same calibration table and required zero additional full-model forwards during DP.

The [layer scores](layer_scores.csv), [15 quota optima](quota_plans.csv), [comparison table](comparison.csv), [per-document NLL/PPL](document_nll.csv) and [selected deployment mask](selected_plan.json) reproduce the reported metrics. The exporter checks them against per-document squared sums, token hashes, raw NLLs, the frozen validation decision, bootstrap differences and the real packed cache allocation.

## Method and data

Each calibration document runs through the original pre-recalibration W4A16 FP16-KV reference model once. At each of its 28 layers, the local codec replay receives identical Q/K/V inputs for K4/V2, K4/V4, K8/V2 and K8/V4 and shares the attention probabilities for a given K format. The unchanged reference output continues to the next layer. We pool the squared differences over the eight documents, all 1,024 causal queries, 16 heads and 128 output coordinates, and divide by the pooled squared reference output plus 1e-12, once per layer. Both sums use FP64. A dynamic program minimizes the sum of these measured per-layer distortions subject to exact layer quotas. It directly accounts for K/V interaction within a layer and approximates the effects across layers by addition.

All three fixed candidates use seven K8 layers, fourteen V4 layers, and otherwise K4 and V2. Early comes from the historical K5/V3 prefix plan. The absolute-error control is the prior ranked k=7 mask; its validation NLL is remeasured on the same fresh articles as the new mask. Budget expansion requires a strictly smaller pooled validation NLL for the new fixed candidate than the absolute-error control. If the gate opens, we optimize over all 15 feasible quotas and evaluate the locally best plan. The best new candidate by validation NLL is frozen and compared with Early on test. The eight test windows score 8,184 teacher-forced next-token targets per model configuration.

The calibration token IDs match [the preceding study](../local_kv_scores/README.md) byte for byte. Fresh validation and test each use eight WikiText-2 articles distinct from each other and previous evaluation documents. Exclusions span 16 local data manifests and 554 token files, plus full-article 16-token spans from the previous study. The official validation/test splits had respectively zero and three eligible articles, so both fresh sets use separate deterministic partitions of the official training split. Test independence refers to these recorded experiments. The raw data manifest records every title, offset and token hash.

## Numerical checks

Recomputing the old three absolute errors from the new squared-error sums reproduced the previous calibration MSE to within 1.04 × 10⁻¹². On the archived HTP control for one layer, the FP16-KV reference replay had relative L2 error 0.002617 against the device capture. For the same input, all four local quantized outputs matched the corresponding offline full-model attention implementation exactly. The validation and test NLL measurements use the RTX 3090 replay.

Reference output energy varied by a factor of 7,121 between layers on calibration. Dividing by that energy changes the influence of high-amplitude layers on the objective. The largest absolute within-layer interaction, measured as D(K8,V4) − D(K8,V2) − D(K4,V4) + D(K4,V2), was 0.01385 at layer 0. This experiment changes both normalization and joint K/V scoring, so their individual contributions to downstream NLL remain unresolved.

## Reproduction

The archived run is under `/mnt/d/ai-hub-models/binaries/turboquant/relative_kv_20261124`. Seed a fresh directory with the experiment protocol and original frozen data manifest so that historical exclusions match. Models and previous study artifacts must be present under the same archive root.

The archived `data_manifest.json` SHA-256 is `db726e9ff1b32bc182e48dfc1e3c6cd6d92cb0f4edc833e768d794cff3b90810`. The calibration scores hash is `f8d5334ff79c4322a6c9e859c1e8bc5d9a607b37441780d89e0bfdd9c5cb91c3`. The frozen validation decision hash is `53d6f07b99552a39dc11ca9caba07e8ff06b476827d10f3330dd8fff465b9e23`. The raw test result hash is `6a4ff377f7788571d7f631b702d57f83d0c010c35af28fc5189f3e35a2a185ff`.

```bash
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=src
archive=/mnt/d/ai-hub-models/binaries/turboquant
original="$archive/relative_kv_20261124"
run_root="$archive/relative_kv_reproduction"
tools_dir=scripts/llm/turboquant
report_dir="$tools_dir/experiments/relative_kv"
mkdir -p "$run_root"
cp --no-clobber "$report_dir/protocol.json" "$original/data_manifest.json" "$run_root/"
venv/bin/python "$tools_dir/relative_kv_data.py" --root "$run_root"
venv/bin/python "$tools_dir/relative_kv_attention.py" --root "$run_root"
venv/bin/python "$tools_dir/evaluate_relative_kv.py" --root "$run_root"
venv/bin/python "$tools_dir/report_relative_kv.py" --root "$run_root" --out "$run_root/report"
```
