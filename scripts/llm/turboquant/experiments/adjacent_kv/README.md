# Qwen3-1.7B adjacent K/V budgets

Six static allocations were specified for Qwen3-1.7B W4A16 with a context capacity of 1,024 tokens. Each uses the existing Structured K8 arithmetic and Lloyd–Max K4/V2/V4 formats. For each K8 quota (6, 7, 8), the V4 quota is respectively 16, 14, 12. The Early policy gives the higher precision to the first layers. The DP policy uses the corresponding frozen mask from the [previous calibration](../relative_kv/quota_plans.csv).

## Validation result

**Early with seven K8 layers and fourteen V4 layers ranked first.** K8 covers layers 0–6 and V4 covers 0–13 (zero-based). All other K and V entries use K4 and V2, respectively. Its mean payload is K5/V3 and its allocated cache is 28.875 MiB. The selected configuration hash is `47a050d4c0849404e6b2e8dc4a07d0b6d19b0c20b2b15d26b91b85cc3c1f3f4d`.

| Rank | Plan | K8 / V4 layer counts | Mean token NLL | PPL | NLL minus Early 07 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | early_07 | 7 / 14 | 2.802460 | 16.48515 | 0.000000 |
| 2 | early_08 | 8 / 12 | 2.803604 | 16.50402 | +0.001144 |
| 3 | dp_06 | 6 / 16 | 2.807909 | 16.57522 | +0.005449 |
| 4 | dp_07 | 7 / 14 | 2.808654 | 16.58758 | +0.006194 |
| 5 | early_06 | 6 / 16 | 2.810144 | 16.61232 | +0.007684 |
| 6 | dp_08 | 8 / 12 | 2.811237 | 16.63048 | +0.008777 |

Early 08 had lower NLL on five of eight articles, but its losses on the other three yielded a pooled PPL 0.01887 higher than Early 07. At matched quotas the DP mask was better at six K8 layers, and Early was better at seven and eight. These ranks describe the eight validation windows used to select the plan.

The 48 document scores comprise 16 qualifying historical records (Early 07 and DP 07) and 32 new full-model forwards. New forwards took 89.50 s. The evaluation invocation took 161.33 s, including 22.80 s to construct and verify the plans, sources and inputs, and 48.30 s to load the model. The time excludes the prior calibration and the subsequent report export. The raw selection record has SHA-256 `7f23ea55f9ccac1e8e3206f8f0c14a08e7c210d2d4d0a0ff5fb1dccd30a51f16`.

## Evaluation

The [selection protocol](protocol.json) fixes the candidate set, eight validation windows, token-weighted ranking and tie rule. Every mask is scored on the same 8,184 next-token targets using the archived W4A16 full-model offline replay on an RTX 3090. The windows come from the earlier experiment's validation set of WikiText-2 articles drawn from the official training split. The lowest unrounded mean NLL wins. An exact tie favors `early_07`, then lexical plan ID.

Validation records from the [prior experiment](../relative_kv/README.md) can be reused when the evaluator, model, runtime, mask and token hashes match, and the SHA-256 of the archived raw JSON matches the previous per-document report. The other masks are scored with the same model and token windows. The six plans are allocated individually using the project cache implementation. Each occupies 29,360,128 packed-index bytes and 917,504 FP16-scale bytes.

## Reproduction

From the repository root with the previous experiment, weights and frozen token assets installed:

```bash
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=src
scripts_dir=scripts/llm/turboquant
archive=/mnt/d/ai-hub-models/binaries/turboquant
run_root="$archive/adjacent_kv_reproduction"
venv/bin/python "$scripts_dir/evaluate_adjacent_kv.py" --root "$run_root" --prior "$archive/relative_kv_20261124"
venv/bin/python "$scripts_dir/report_adjacent_kv.py" --root "$run_root" --out "$run_root/report"
```

The original archive is `/mnt/d/ai-hub-models/binaries/turboquant/adjacent_kv_20261124` and contains the protocol, evaluation identity, all six plans, raw document scores, validation decision, timing and execution log. The [pooled comparison](comparison.csv), [document scores](document_nll.csv) and [selected layer mask](selected_plan.json) are also available here.
