# Same-memory K/V refinement on Qwen3

Offline W4A16 replay compared local K/V allocations on WikiText-2. Each model used 4 documents (4,092 targets) for calibration and 8 separate documents (8,184 targets) for held-out evaluation, with one 1,024-token window per document. The historical Early anchors use K5/V3 for Qwen3-1.7B and Qwen3-4B and K6/V2 for Qwen3-8B. [Historical placement comparison](../cross_model_pu_budget/offline_report.md)

## Held-out performance of the calibration choice

NLL is mean next-token negative log-likelihood and PPL is its exponential. Each difference is the chosen allocation minus its model's anchor on the same targets. Negative differences favor the chosen allocation. Document counts give lower, equal and higher NLL in that order.

| Model | Chosen plan | Anchor PPL | Chosen PPL | Δ PPL | Δ NLL (nats/target) | Document Δ NLL, 95% interval | Lower / equal / higher |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | `kplus1_vminus2` | 20.96537 | 21.15232 | +0.18695 | +0.008878 | [-0.005684, +0.022831] | 1 / 0 / 7 |
| Qwen3-4B | `anchor__k8_last` | 16.84847 | 16.58070 | -0.26777 | -0.016020 | [-0.041090, +0.008980] | 6 / 0 / 2 |
| Qwen3-8B | `kminus1_vplus2__k8_adjacent` | 11.88824 | 12.04581 | +0.15757 | +0.013167 | [-0.003839, +0.030232] | 2 / 0 / 6 |

- Qwen3-1.7B had higher held-out NLL with the chosen allocation. The document interval includes zero.
- Qwen3-4B had lower held-out NLL with the chosen allocation. The document interval includes zero.
- Qwen3-8B had higher held-out NLL with the chosen allocation. The document interval includes zero.

Intervals summarize variation across these WikiText-2 documents. Each of 10,000 bootstrap replicates resamples complete anchor/candidate document pairs with replacement and averages their NLL differences with equal document weight (seed 20261120, 2.5th and 97.5th percentiles). The small sample limits precision. The comparisons address the measured local moves, model checkpoints and corpus. A budget exchange changes both K and V, so it measures their joint effect.

## Chosen allocations

Layer indices start at zero and ranges include both endpoints. Listed key layers use piecewise-uniform K8 and all other key layers use Lloyd-Max K4. Listed value layers use Lloyd-Max V4 and all other value layers use Lloyd-Max V2. All rows average eight payload bits per K/V coordinate pair within the model. The CL1024 format size includes the packed indices and one FP16 scale for each K and V head vector.

| Model | Mean K / V bits | CL1024 MiB | K8 layers | V4 layers | Change to K8 from anchor | Change to V4 from anchor |
|---|---:|---:|---|---|---|---|
| Qwen3-1.7B | 5.143 / 2.857 | 28.875 | 0-7 | 0-11 | add 7 | remove 12-13 |
| Qwen3-4B | 5.000 / 3.000 | 37.125 | 0-7, 35 | 0-17 | add 35, remove 8 | same |
| Qwen3-8B | 5.889 / 2.111 | 37.125 | 0-15, 17 | 0-1 | remove 16 | add 0-1 |

## Held-out controls

The best budget-only candidate and the best placement-only candidate on calibration were also frozen for each model. Their held-out comparisons with the anchor are exploratory. A candidate that also won overall appears in both tables.

| Model | Control family | Frozen plan | Δ NLL | Δ PPL | Lower / equal / higher |
|---|---|---|---:|---:|---:|
| Qwen3-1.7B | Placement only | `anchor__k8_adjacent` | +0.012055 | +0.25428 | 3 / 0 / 5 |
| Qwen3-1.7B | Budget only | `kplus1_vminus2` | +0.008878 | +0.18695 | 1 / 0 / 7 |
| Qwen3-4B | Placement only | `anchor__k8_last` | -0.016020 | -0.26777 | 6 / 0 / 2 |
| Qwen3-4B | Budget only | `kplus1_vminus2` | -0.014966 | -0.25028 | 6 / 0 / 2 |
| Qwen3-8B | Placement only | `anchor__k8_last` | +0.008907 | +0.10636 | 2 / 0 / 6 |
| Qwen3-8B | Budget only | `kminus1_vplus2` | +0.010205 | +0.12194 | 3 / 0 / 5 |

## Calibration search

A budget move exchanges one K8 layer for two V4 layers or vice versa. A placement swap moves the higher precision between layers at a fixed K or V count. Each swap moves the last higher-precision layer in an Early prefix to the adjacent, middle or final lower-precision layer. Joint candidates vary both allocation counts and placement. The 8B anchor has zero V4 layers, so V placement first becomes testable after one budget exchange. The lowest calibration NLL fixes the model's choice before held-out scoring. Ties favor the anchor, then the lexicographically first plan ID.

| Model | Candidate family | Plans scored | Lowest calibration NLL plan | Δ NLL vs anchor | Budget contribution | Placement contribution |
|---|---|---:|---|---:|---:|---:|
| Qwen3-1.7B | Early anchor | 1 | `anchor` | +0.000000 | +0.000000 | +0.000000 |
| Qwen3-1.7B | Budget and placement (move K8) | 2 | `kplus1_vminus2__k8_adjacent` | -0.011721 | -0.016357 | +0.004636 |
| Qwen3-1.7B | Budget and placement (move V4) | 2 | `kminus1_vplus2__v4_adjacent` | +0.007299 | +0.021883 | -0.014584 |
| Qwen3-1.7B | Budget only | 2 | `kplus1_vminus2` | -0.016357 | -0.016357 | +0.000000 |
| Qwen3-1.7B | Placement only (move K8) | 3 | `anchor__k8_adjacent` | -0.002880 | +0.000000 | -0.002880 |
| Qwen3-1.7B | Placement only (move V4) | 3 | `anchor__v4_midpoint` | +0.012451 | +0.000000 | +0.012451 |
| Qwen3-4B | Early anchor | 1 | `anchor` | +0.000000 | +0.000000 | +0.000000 |
| Qwen3-4B | Budget and placement (move K8) | 2 | `kplus1_vminus2__k8_adjacent` | -0.015256 | -0.012643 | -0.002613 |
| Qwen3-4B | Budget and placement (move V4) | 2 | `kplus1_vminus2__v4_adjacent` | -0.010374 | -0.012643 | +0.002268 |
| Qwen3-4B | Budget only | 2 | `kplus1_vminus2` | -0.012643 | -0.012643 | +0.000000 |
| Qwen3-4B | Placement only (move K8) | 3 | `anchor__k8_last` | -0.021050 | +0.000000 | -0.021050 |
| Qwen3-4B | Placement only (move V4) | 3 | `anchor__v4_midpoint` | -0.015856 | +0.000000 | -0.015856 |
| Qwen3-8B | Early anchor | 1 | `anchor` | +0.000000 | +0.000000 | +0.000000 |
| Qwen3-8B | Budget and placement (move K8) | 1 | `kminus1_vplus2__k8_adjacent` | -0.013237 | -0.009585 | -0.003652 |
| Qwen3-8B | Budget and placement (move V4) | 3 | `kminus1_vplus2__v4_adjacent` | +0.000501 | -0.009585 | +0.010086 |
| Qwen3-8B | Budget only | 1 | `kminus1_vplus2` | -0.009585 | -0.009585 | +0.000000 |
| Qwen3-8B | Placement only (move K8) | 3 | `anchor__k8_last` | +0.004378 | +0.000000 | +0.004378 |

The budget contribution compares the Early prefix at the candidate's quotas with the anchor. The placement contribution compares the candidate with that same prefix. Their sum is the candidate's calibration Δ NLL. The placement effect can depend on the K/V quotas and the layers exchanged.

The CSV contains all 35 calibration scores and 10 held-out scores, including absolute NLL, PPL, signed differences from the corresponding anchor, per-document scores, exact layer lists and configuration hashes. [Download all scores](scores.csv)

## Evaluation documents

The corpus is `Salesforce/wikitext/wikitext-2-raw-v1` at revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Calibration uses the official training split. The held-out pool combines validation and test. The dataset builder excluded archived article titles and content hashes, screened for shared 16-token sequences against inventoried tokens and previously selected articles, and checked that the three tokenizers agree. These checks cover recorded local experiments. Model pretraining and original activation-calibration data membership are unknown.

| Role | Official split | Window ID in CSV | Article | Token offset |
|---|---|---|---|---:|
| calibration | train | `calibration_00.bin` | WASP @-@ 13b | 256 |
| calibration | train | `calibration_01.bin` | Cadmium | 10 |
| calibration | train | `calibration_02.bin` | M @-@ 6 ( Michigan highway ) | 256 |
| calibration | train | `calibration_03.bin` | There 's Got to Be a Way | 104 |
| heldout | validation | `heldout_00.bin` | Raid on Manila ( 1798 ) | 256 |
| heldout | test | `heldout_01.bin` | Nerva | 256 |
| heldout | validation | `heldout_02.bin` | 1985 Rajneeshee assassination plot | 256 |
| heldout | test | `heldout_03.bin` | Commonwealth War Graves Commission | 256 |
| heldout | validation | `heldout_04.bin` | Mount Elbert | 256 |
| heldout | test | `heldout_05.bin` | American Beauty ( 1999 film ) | 256 |
| heldout | test | `heldout_06.bin` | View of the World from 9th Avenue | 256 |
| heldout | test | `heldout_07.bin` | Shackleton ( crater ) | 256 |

Dataset manifest SHA-256 `4f1f746a1b416fbae4ceca91c6a6e2e58d1b49ff94631884f3ab40e3ad459559`. Candidate protocol SHA-256 `d1afeb3952b2fee1263a9315e8a1569364e58969daf8689888680fc61e23c777`. Evaluation protocol SHA-256 `4ecfe8cc209de992172decd634054d1ce1e83336f0b3058f47e1b7ab8bd7222a`.

## Reproduce the experiment

Run from the repository root with the archived source corpus, checkpoints, split graphs and historical scores available. The commands freeze candidates and documents, run W4A16 replay, verify raw scores and regenerate the report.

```bash
run_root=/mnt/d/ai-hub-models/binaries/turboquant/cross_model_pu_refinement_20261120
export PYTHONPATH=src
venv/bin/python scripts/llm/turboquant/refinement_plans.py --out "$run_root"
venv/bin/python scripts/llm/turboquant/refinement_data.py --root "$run_root"
venv/bin/python scripts/llm/turboquant/run_kv_refinement.py --root "$run_root"
venv/bin/python scripts/llm/turboquant/verify_kv_refinement.py --root "$run_root"
python3 scripts/llm/turboquant/report_kv_refinement.py --root "$run_root"
```

Input `summary.json` SHA-256 `f196fb6b85e1bca0bb7322def45e0091c95c40622e3d3c34c9d6f2de4b42d61f`.


Calibration-selected allocation files: [Qwen3-1.7B](selected_plans/qwen3_1_7b.json) · [Qwen3-4B](selected_plans/qwen3_4b.json) · [Qwen3-8B](selected_plans/qwen3_8b.json).
