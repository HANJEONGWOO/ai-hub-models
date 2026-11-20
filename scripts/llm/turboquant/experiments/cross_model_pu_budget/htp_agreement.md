# Agreement with archived Galaxy S26 Ultra measurements

The offline replay and the Snapdragon SM8850 HTP give the same direction for **20 of 21** pairwise comparisons across the 12 configurations measured on both. Each pair compares two distinct masks or budgets within the same model. The quality metric is mean next-token NLL across the same four WikiText windows and 4,092 targets. Sorting by perplexity gives the same order.

| Model | Matched masks | Pairs | Same order | Reversed | Tied in either evaluation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 6/11 | 15 | 14 | 1 | 0 |
| Qwen3-4B | 3/11 | 3 | 3 | 0 | 0 |
| Qwen3-8B | 3/11 | 3 | 3 | 0 | 0 |

## Reversals and ties

Each difference is the mean NLL of plan A minus that of plan B, in nats per token. A negative value favors A.

| Model | Plan A | Plan B | Offline difference | HTP difference |
| --- | --- | --- | ---: | ---: |
| Qwen3-1.7B | `k6_v2_early` | `k6_v2_boundary` | -0.012145 | +0.006257 |

## Scope

The archive covers K4/V4 on all three models and all five PU K6/V2 placements on 1.7B. Its 4B and 8B PU K6/V2 measurements cover boundary and interleaved placement. PU K5/V3 has no matching archived device score. Earlier K5/V3 measurements use different key codecs.

The generator matches the W4A16 split, AR128 attention encodings, active codebooks, rotations, layer widths and token hashes. The 4B and 8B splits are specific to their respective checkpoints. All device timings in the reference are single archived 897-token prefill and 128-token generation runs. The replay approximates HTP arithmetic. The four fixed windows and overlapping within-model pairs support descriptive comparisons only.

The reference CSV records each raw score and timing report with its SHA-256. The offline summary SHA-256 is `395090c39d533342bf564d34456e9216bd343b98f5a69f64c371ba8af335d0f5`. [HTP reference](htp_reference.csv) · [Offline scores](offline_scores.csv)
