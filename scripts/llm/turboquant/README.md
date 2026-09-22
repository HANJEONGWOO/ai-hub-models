<!--
Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
SPDX-License-Identifier: BSD-3-Clause
-->

# TurboQuant KV-cache tools

Developer tools for the opt-in TurboQuant KV-cache codec in
`qai_hub_models.models.templates.llm.turboquant`. The ABI, numeric tolerances
and current results are in
[`tutorials/llm/turboquant_design.md`](../../../tutorials/llm/turboquant_design.md).

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
