<!--
Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
SPDX-License-Identifier: BSD-3-Clause
-->

# TurboQuant KV-cache tools

Developer tools for the opt-in TurboQuant KV-cache codec in
`qai_hub_models.models.templates.llm.turboquant`. The ABI, numeric tolerances
and current results are in
[`tutorials/llm/turboquant_design.md`](../../../tutorials/llm/turboquant_design.md).

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

Run from the repo root with `PYTHONPATH=src` so the repo sources shadow any
installed `qai_hub_models` wheel. None of these tools submit AI Hub jobs.

For restore throughput, benchmark the **past-cache length**, not only the
number of newly generated tokens. An AR=1, context=1024 graph restores 1023
tokens per KV tensor:

```bash
PYTHONPATH=src python scripts/llm/turboquant/htp_codec_validation.py all \
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
    --context-length 1024 --profile k4_v4 --attention-tile 256
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
