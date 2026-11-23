# Qwen3-1.7B layerwise K/V allocation

Qwen3-1.7B W4A16, CL1024, offline replay on an RTX 3090. The [frozen protocol](protocol.json) defines the score, feasible layer counts, tie rules and separate validation/test evaluations. Results are specific to the 2026-11-23 study.

## Outcome

Validation selected **K5/V3 Early**. All three ranked plans had higher NLL than Early on each of the eight validation documents. The largest predicted reduction in local attention MSE occurred at k=8, with K8 on layers 20–27 and V4 on layers 16–27. The k=7 plan has the same K/V quotas as Early; placement alone increased validation NLL by 0.5560 nats/target.

| Plan | K8 / V4 layers (counts) | Sum of local gains | Validation NLL | Validation PPL | NLL change from Early |
|---|---:|---:|---:|---:|---:|
| K5/V3 Early | 7 / 14 | 0.13784 | **2.74407** | **15.55021** | 0 |
| Ranked k=7 | 7 / 14 | 12.23216 | 3.30006 | 27.11438 | +0.55599 |
| Ranked k=8 | 8 / 12 | **12.24354** | 3.26703 | 26.23333 | +0.52296 |
| Ranked k=9 | 9 / 10 | 12.13608 | 3.26443 | 26.16528 | +0.52036 |

The frozen winner and the control were the same configuration. Its single test evaluation across eight independent documents yielded **NLL 2.98581, PPL 19.80249**. The test difference is zero by identity; the test supplies no independent estimate of the quality gap between a ranked mask and Early. Every evaluated plan allocated **30,277,632 bytes (28.875 MiB)** in the packed host cache implementation, comprising 28 MiB of indices and 0.875 MiB of FP16 scales at CL1024.

The final eight layers accounted for 89.0% of all K gains and 90.9% of all V gains. The absolute MSE proxy therefore concentrated precision late. The worse end-to-end NLL at matched quotas establishes a placement effect on these validation documents. The experiment does not identify whether scale differences between layers, K/V interaction, or propagation through the remaining network explains the discrepancy.

[Layer errors and ranks](layer_scores.csv) · [All 15 budget points and masks](budget_curve.csv) · [Validation and test scores](shortlist.csv) · [Per-document NLL](document_nll.csv) · [Selected layer plan](selected_plan.json)

## Time and numerical checks

| Measured stage | Seconds |
|---|---:|
| FP16 graph/source preparation, hashes and loading | 73.75 |
| Reference and historical replay check | 0.53 |
| Eight single-pass calibration forwards including all local attention outputs | 27.08 |
| Entire calibration invocation including setup, control and writes | 101.42 |
| Rank and freeze all 15 plans | 0.073 |
| End-to-end scoring model load | 45.70 |
| Validation of four unique plans over eight documents | 92.03 |
| Entire ranking/selection invocation through decision, including identity and cache checks | 161.98 |
| Test and summary, separately | 22.88 |

The two selection stages took **263.40 s** in total. Including independent test, their combined wall time was **286.28 s**. These times exclude dataset preparation, development, and the gaps between invocations. There was no expensive search arm in this experiment, so its speedup over full-model search was not measured.

On a historical HTP capture (layer 0, all 1,024 queries), offline FP16 reference AV had relative L2 error 0.002617. At identical inputs all three local quantized paths had maximum absolute difference 0 against the existing offline NLL attention implementation. This comparison checks the offline arithmetic; no new HTP inference was run. The ranking calculation uses the same FP16 Q/K/V for all locally compared states.

Raw files reside under `/mnt/d/ai-hub-models/binaries/turboquant/local_kv_scores_20261123`. Data manifest SHA-256 `a9adf42b02130989167452614b1a96d7a275022f075fcc57179dd9efdc61fa4d`, local-score summary `717ac042cd4909f1a6780a439b543dd733b97d47f8bad31ec5fdef60fa75fcb4`, test result `a2988dcd3143fd32e26138657a6dfa5f785c1177fb596597fab460301a657456`.

## Method

One pre-recalibration W4A16 FP16-KV forward per calibration document supplies the identical post-RoPE query and key and the value for each local comparison. Only the reference attention output continues to the next layer. Local quantized paths reproduce the existing dense rotations, structured arithmetic K8, Lloyd–Max K4/V2/V4, norm correction, FP16 effective scales, current-token quantization and causal AR128 attention with 256-token AV tiles.

Every layer receives a mean squared error for K4/V2, K8/V2 and K4/V4. Errors are averaged over all 1,024 query positions, 16 query heads, 128 head coordinates and eight equally weighted documents. The difference from K4/V2 gives the independent K and V promotion gains. Both lists are sorted in descending order. For each integer k from 0 to 14 the plan uses the top k K layers and top 28−2k V layers. The largest sum of gains nominates a plan and its feasible neighbors. Equal scores favor the lower layer ID and an equal gain sum favors the smaller k.

The end-to-end scorer uses the previous study's W4A16 weights, activation grids and TurboQuant codec implementation. It replays the model on the GPU with explicit FP16 rounding. The historical K5/V3 Early anchor has K8 on layers 0–6 and V4 on layers 0–13. Both selection and test use pooled teacher-forced next-token NLL over 8,184 targets per configuration. Only the validation winner and Early enter test, with identical masks scored once. Final-test data are excluded from mask selection.

## Data

The three cohorts use 24 distinct WikiText-2 articles. Previous local studies left zero eligible official validation articles and three eligible official test articles under the full-article overlap rule. Before scoring, the dataset builder used three disjoint salted hash partitions of the official training split for the study cohorts. Each has eight documents, screened against 15 previous manifests and 530 archived token files by title, hash and 16-token spans. Independence is relative to recorded local studies. The raw manifest records the article, offset, tokenizer and hash of each window. Model pretraining corpus membership is unknown.

## Reproduction

Run at the repository root using the archived graph and tokenizer locations. To repeat the experiment in a fresh directory, seed it with the published data manifest so the prior-study exclusion list is identical. Large binaries and raw measurements live under the archive root.

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
archive=/mnt/d/ai-hub-models/binaries/turboquant
original="$archive/local_kv_scores_20261123"
run_root="$archive/local_kv_scores_reproduction"
tool_dir=scripts/llm/turboquant
report_dir="$tool_dir/experiments/local_kv_scores"
mkdir -p "$run_root"
cp --no-clobber "$report_dir/protocol.json" "$run_root/protocol.json"
cp --no-clobber "$original/data_manifest.json" "$run_root/data_manifest.json"
venv/bin/python "$tool_dir/local_kv_data.py" --root "$run_root"
venv/bin/python "$tool_dir/local_kv_attention_scores.py" --root "$run_root"
venv/bin/python "$tool_dir/evaluate_local_kv_scores.py" --root "$run_root"
venv/bin/python "$tool_dir/report_local_kv_scores.py" --root "$run_root" --out "$run_root/report"
```
