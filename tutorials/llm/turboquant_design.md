# Qwen3 TurboQuant KV-cache — 설계(ABI·수치 계약)와 P0–P3 결과

> §11의 ladder 복원 결과 이후 성능 수정과 재측정은 §12에 기록한다.

작성일: 2026-09-15 · 작업 브랜치: `turboquant-kv-cache` (기준 commit `2a895603e`)

작업 명세: [turboquant_npu_implementation_spec.md](turboquant_npu_implementation_spec.md). 이 문서는 명세 P0의 `design.md` 산출물이며, P1(참조 구현), P2(최소 HTP 실행 검증), P3(Qwen3-1.7B 실기기 통합, context 1024)의 실제 결과를 포함한다. codec 입력 경계(KV 전용 int8 이전 값)의 정의·검증과 FP16 KV 비교군은 §11에 있다. P4의 체계적 벤치마크(여러 context, 응답 품질 평가, HTP 측 메모리 계측)와 P5(다른 모델 크기)는 수행하지 않았다.

## 1. 완료 상태 (명세 10.2 기준)

| 상태 | 판정 | 근거 |
|---|---|---|
| 참조 구현 완료 | **완료** | 고정 commit 원본 대비 index 불일치 0, FWHT 복원 오차 0.0, packed byte 동일 (§4.4). 단위 테스트 144개 통과 |
| 최소 codec HTP 실행 (P2) | **완료** | S26(SM8850) HTP에서 encode/decode 8개 그래프가 fp16 허용오차로 oracle과 일치, 모든 op이 accelerator 프로파일에 기록됨 (§6) |
| NPU 기능 검증 완료 (1.7B 전체 생성 루프) | **완료, 단 메모리 경로 조건 미충족** | `k4_v4`가 S26에서 prefill + 128토큰 decode, EOS 처리, 세션 reset, context 경계(1024) 통과. 최종 수정 번들의 codec op이 28개 layer 전부 HTP detailed profile에 기록됨 (§12.3). KV는 호출 사이에 host에서 packed로 보관되지만, 그래프 안에서 과거 KV 전체를 매 스텝 복원하므로 명세 5.3의 "최종 메모리 경로"는 아님 |
| 압축 효과 입증 | 부분 | host KV 저장소(56.0→28.9 MiB)와 프로세스 RSS(221.8→138.6 MiB, 1회 측정) 감소를 측정. HTP 측 intermediate·scratch·shared memory는 측정하지 않음 |
| 품질 평가 완료 | 부분 | 실기기 WikiText teacher-forced PPL(4 window) 측정. 응답 품질 평가(Grace 등)와 retrieval 시험은 수행하지 않음 |
| 성능 개선 입증 | **기존 codec 대비 개선, baseline 대비 미달** | §12 수정으로 decode 1.95→10.76 tok/s(5.52배), TTFT 1270→118ms. 동일 장치 int8 baseline은 41.79 tok/s로 여전히 3.88배 빠름 |
| 타 모델 검증 완료 | 미착수 | 0.6B/4B/8B는 config·shape 테스트만 통과 |
| codec 입력 경계 검증 | **완료** | codec이 KV 전용 int8 양자화 이전의 16-bit 값을 읽음을 ONNX encodings·dlc-info·onnxruntime에서 확인(§11.4). 같은 16-bit 값을 변환 없이 그대로 저장하는 비압축 비교군 `baseline_int16_kv`를 별도 artifact로 생성·실행 |

## 2. 환경 manifest

| 항목 | 값 |
|---|---|
| ai-hub-models | commit `2a895603e`, 브랜치 `turboquant-kv-cache` |
| 알고리즘 기준 | turboquant_plus `ba52ad107d1fdd02bc9be8fd85308226b75c905b` (Apache-2.0) |
| host Python | 3.12.3, numpy 2.4.4, scipy 1.17.1, torch 2.10.0+cu128, transformers 5.7.0, onnx 1.18.0, onnxruntime-gpu 1.23.2 |
| QAIRT | 2.48.0.260626 (x86 converter는 `~/qnn-venv`, onnx 1.18.0) |
| Hexagon SDK / NDK | 6.6.0.0(이번 단계에서 미사용, custom op 불필요) / r26c(`libc++_shared.so`만 사용) |
| 장치 | Galaxy S26 Ultra `SM-S948N`, `ro.soc.model=SM8850`(QNN soc_model 87, HTP V81), `samsung/m3qksx/m3q:16/BP4A.251205.006/S948NKSS3AZF1_OKR3AZF1:user/release-keys` |
| 모델 | `Qwen/Qwen3-1.7B` HF revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| 평가 데이터 | `Salesforce/wikitext` `wikitext-2-raw-v1` test parquet (snapshot `b08601e0`) |

## 3. 코드 구성

| 위치 | 내용 |
|---|---|
| `src/qai_hub_models/models/templates/llm/turboquant/config.py` | 프로파일(`baseline_int8`, `baseline_int16_kv`, `k4_v4`, `k8_v3`, `k4_v3`), format version, `config_hash()`, 모델 shape 검사 |
| `.../turboquant/constants.py` | 생성된 고정 상수: codebook(float64 hex), FWHT sign, sha256 |
| `.../turboquant/reference.py` | float64 CPU oracle: `FWHTRotation`, `DenseQRRotation`, `PolarQuantReference` |
| `.../turboquant/packing.py` | MSB-first bit packing, norm 저장 dtype 정책 |
| `.../turboquant/cache.py` | host packed cache(append/reset/overflow/직렬화/메모리 산정) |
| `.../turboquant/export.py` | HTP 호환 built-in op만 쓰는 ONNX encode/decode 그래프 |
| `.../turboquant/numerics.py` | backend별 허용오차와 oracle 비교 |
| `src/qai_hub_models/test/test_models/test_turboquant_{codec,cache,export}.py` | 단위 테스트(`test_qaihm` CI 수집 경로), golden fixture `turboquant_golden_v1.json` |
| `scripts/llm/turboquant/generate_constants.py` | 고정 commit 참조를 실행해 `constants.py` 생성/`--check` |
| `scripts/llm/turboquant/verify_reference.py` | oracle ↔ 고정 commit 원본 대조, golden fixture 생성 |
| `scripts/llm/turboquant/evaluate_qwen3_kv.py` | PC 참조 평가(PPL/KL/KV 통계)와 실제 KV snapshot 저장 |
| `scripts/llm/turboquant/htp_codec_validation.py` | P2: 변환 → context binary → adb 실행 → 비교·프로파일 증거 |
| `.../turboquant/graph_surgery.py` | P3: split part ONNX의 `past_*_in`/`past_*_out`에 codec 서브그래프 삽입, encodings 재사용. cache 분기 재구성(tap 재격자화, guard, 복제, §11.3)과 비압축 int16 KV 프로파일, encoding 편집 리포트 |
| `src/qai_hub_models/test/test_models/test_turboquant_graph_surgery.py` | 합성 delta-KV part로 surgery 배선을 onnxruntime에서 검증 |
| `scripts/llm/turboquant/split_checkpoint.py` | 배포 AIMET checkpoint를 repo split 코드로 part별 번들로 로컬 분할 |
| `scripts/llm/turboquant/convert_parts.py` | part별 surgery(프로파일) → `qairt-converter --quantization_overrides` → `qairt-quantizer --enable_float_fallback` → prompt/token 그래프 weight-shared context binary |
| `scripts/llm/turboquant/qnn_runner/` | 공개 QNN C API로 작성한 Android runner(`qnn-llm-runner`): 이름 기반 I/O 역할, pyref KV 배치, generate/score, 세션 반복, detailed profile, JSON 리포트 |
| `scripts/llm/turboquant/run_device_llm.py` | RoPE 표·토큰 asset 생성, sha256 기반 push, 장치 실행과 리포트 수집 |
| `scripts/llm/turboquant/verify_kv_boundary.py` | §11: 변환된 그래프의 `qairt-dlc-info` op 표에서 KV 양자화 경계(write chain dtype, codec 입력, packed/norm encoding 부재, attention Concat의 int8 변환 위치)를 검사하고 JSON 증거를 남김 |

기존 모델, export 경로, 다른 LLM 코드는 수정하지 않았다. TurboQuant는 opt-in 스크립트 경로로만 적용되므로 기본 export 동작에는 영향이 없다.

## 4. 알고리즘·수치 계약 (format version 1)

### 4.1 경로 구분

| 이름 | 회전 | QJL | 용도 |
|---|---|---|---|
| `Rotation.FWHT` (기본) | `R = D2·H·D1`, 역방향 `D1·H·D2`, `H`는 Sylvester 순서·`1/sqrt(128)` 정규화 | 없음 | 실사용 codec, NPU 그래프 |
| `Rotation.DENSE_QR` | 참조 `PolarQuant`와 같은 Haar QR(`default_rng(seed)`, sign·det 보정) | 없음 | 참조 클래스와의 대조 oracle 전용 |
| QJL | — | — | 1차 범위 밖. `get_profile("qjl_reference")`는 `NotImplementedError` |

참조 Python `PolarQuant`는 dense QR만 사용하며 FWHT 구성은 참조 repo에 클래스로 존재하지 않는다. FWHT codec은 참조 `rotation.random_rotation_fast`/`apply_fast_rotation*` 조각을 조합한 것이고, 이 조합을 참조 조각과 직접 대조했다(§4.4).

### 4.2 상수와 seed

- codebook: 참조 `codebook.optimal_centroids(bits, 128)`(N(0, 1/d) Gaussian 근사 Lloyd, quantile 초기화, **정확히 100회 반복**). 4-bit 최외곽 centroid는 0.24021009724443104이다. `turbo4-resurrection.md`의 표(0.1739)는 반복 0회 초기값이고, 수렴 Lloyd-Max(0.241529)와도 다르다. 둘 다 사용하지 않는다.
- FWHT sign: `random_rotation_fast(128, np.random.default_rng(seed))`에서 `signs1`을 먼저, `signs2`를 나중에 뽑는다. seed 정책은 참조 `KVCacheCompressor`를 따라 K=42, V=542이다.
- 모든 상수는 float64 hex로 고정하고 sha256을 기록한다. runtime에 RNG나 scipy를 다시 실행하지 않는다.

| 상수 | sha256 |
|---|---|
| 4-bit codebook (d=128, float64 LE) | `2260ac9861ecc11872763ac359673e54e76be620463948dfaa105f97c1a07740` |
| 3-bit codebook | `5d03648f22b4824350be74b68224644c757385477d07eae8dfa3afe98cb2e455` |
| FWHT sign seed 42 (`int8(s1)+int8(s2)`) | `49847ad511f6a5ef144adc61803db50cbc7c1e6d8ea9b15678cc4d4da7ccdaf3` |
| FWHT sign seed 542 | `c41c124d07d9dc3b561daa9984b1ec9de6a2edf681db090e0f12fe5c985dbedb` |

참조 문서에는 llama.cpp/Metal의 실사용 sign 값이 없으므로 기존 llama.cpp binary cache와의 호환은 주장하지 않는다.

### 4.3 encode/decode 정의

- encode: `norm = ||x||₂`, `y = R·(x / norm)`(norm이 0이면 1로 나눔), `idx = searchsorted(midpoints(c), y, side="left")`. 경계값과 정확히 같으면 **낮은 index**가 된다.
- decode: `ŷ = c[idx]`, `norm_correction=true`이면 `ŷ /= ||ŷ||`(guard `>1e-10`, d=128에서는 도달 불가), `x̂ = Rᵀ·ŷ·norm`.
- 영벡터: norm 0, 복원 결과 정확히 0이다. 4-bit 중앙 경계가 −3.6e-17이라 float64 oracle은 index 8을 내지만 fp16에서는 경계가 −0이 되어 7이 나온다. 회전 좌표가 정확히 0인 경우의 index는 backend마다 다를 수 있다. 영벡터는 복원값이 그대로 0이다. 영벡터가 아니면 `c7 = −c8`이므로 오차 크기는 같지만 복원 좌표의 부호가 달라진다.
- NaN/Inf 입력: host oracle은 `ValueError`로 거부한다. HTP 그래프는 검사하지 않으므로 호출자가 finite 입력을 보장해야 한다.
- block: `head_dim == block_size == 128`만 지원한다. 2의 거듭제곱이 아니거나 128과 다르면 `ValueError`이며, 조용한 truncate나 padding은 없다.
- norm 저장: FP16(첫 후보). 저장 전에 finite·음수 아님을 검사한다. 65504를 넘으면 `OverflowError`, 0이 아니면서 FP16 최소 정규값(6.1e-5)보다 작으면 `ValueError`를 낸다(0으로 반올림되거나 거친 subnormal이 되는 것을 막음). 정확히 0은 허용한다. `norm_dtype="float32"`로 바꾸면 `config_hash`가 달라진다. 실제 Qwen3-1.7B의 norm은 K 11.4~399.6, V 0.33~852.7로 두 한계에서 충분히 떨어져 있다(snapshot layer 0/13/27과 §7.2).

### 4.4 참조 대조 결과 (`verify_reference.py`)

2,054개 벡터(Gaussian과 영벡터, 1e-30, 1e4, one-hot, 상수 행 포함)에서 확인했다.

- codebook 3/4-bit: 참조와 byte 동일
- FWHT forward/inverse(seed 42, 542): 최대 오차 **0.0**. dense QR 행렬: 0.0
- 참조 `PolarQuant`(dense QR, 3/4-bit, seed 42/542): index 불일치 **0**, 복원 최대 상대오차 ≤2.2e-16
- FWHT 조합 경로: index 불일치 **0**, 복원 오차 0.0. `pack_indices` 결과가 참조 `utils.pack_indices`와 동일

## 5. Cache ABI v1

### 5.1 선택한 cache 경로

Qwen3는 현재 delta-cache I/O(`genie_input_ids`)만 구현되어 있다. 그래프는 새 토큰의 K/V만 출력하고, host가 전체 cache를 보관한다. native KV(ScatterElements)는 Qwen3에 없다. 따라서 1차 지원 대상은 **delta-cache**로 정했고, 다른 I/O 조합은 P3에서 명시적으로 거부한다.

### 5.2 형식

| 항목 | 값 |
|---|---|
| format | `qaihm-turboquant-kv`, version 1 |
| config 식별 | `TurboQuantConfig.config_hash()` = 정렬된 JSON(프로파일, K/V codec·bits·seed, block size, rotation, norm correction, norm dtype, bit order, `qjl=false`, 참조 commit, codebook/sign sha256)의 sha256 |
| packed index | `uint8 (kv_heads, batch, token, bytes)`, K와 V 모두 같은 축 순서. 4-bit는 128개 index → 64 byte, 3-bit는 48 byte |
| bit 순서 | block 안에서 연속 MSB-first bit stream(참조 `pack_indices`와 동일). 4-bit는 `byte = idx[2k] << 4 \| idx[2k+1]`. tail은 0으로 채우며, unpack은 0이 아닌 padding bit를 거부 |
| norm | `float16 (kv_heads, batch, token, 1)`, 원래 L2 norm(미리 나누지 않음) |
| hub layout 변환 | 입력 K `(kv_heads, batch, head_dim, new)`는 token 축을 −2로 바꿔 저장. V `(kv_heads, batch, new, head_dim)`는 그대로. decode 결과는 원래 hub layout으로 반환. `new == head_dim(128)`이면 shape만으로 hub K와 HF K를 구분할 수 없으므로, 통합 시에는 graph I/O 이름으로 텐서를 매핑해야 한다 |
| 길이 | 논리 길이(`get_seq_length`)와 할당 capacity(`context_length`)를 구분하며 buffer는 capacity만큼 선할당 |
| append | 모든 layer가 같은 새 토큰 수를 가져야 한다. 새 토큰만 encode하고 과거 토큰은 재양자화하지 않는다. capacity를 넘으면 쓰기 전에 `ValueError`를 내며 sliding은 없다 |
| 세션 | `reset()`이 buffer와 길이를 0으로 만든다. `batch_size != 1`은 `NotImplementedError` |
| 직렬화 | `state_dict()`는 buffer 복사본과 format/version/config hash/shape/length를 담은 snapshot이다. `load_state_dict()`는 layer 수·buffer 이름·shape·dtype을 모두 검증한 뒤에만 쓰므로, 잘못된 state가 cache를 일부만 덮어쓰지 않는다 |
| `BASELINE` 텐서 | host는 기존 generator처럼 float32로 보관하고, int8은 배포 그래프 내부의 기존 경로가 담당 |

### 5.3 메모리 산정 검산 (명세 9.4)

`TurboQuantKVCache.memory_report()` 단위 테스트로 고정했다(Qwen3-1.7B, 28 layers, 8 KV heads, 4096 tokens).

| 프로파일 | packed payload | FP16 norm | host float32 K |
|---|---:|---:|---:|
| `k4_v4` | 112 MiB | 3.5 MiB | 0 |

이 값은 산술 검산이며 실측 allocation이 아니다. 정렬, QNN intermediate, scratch, host mirror는 P3/P4에서 따로 측정해야 한다.

### 5.4 수치 허용오차 (P1에서 고정)

| backend | 허용되는 index 차이 | norm 상대오차 | decode 상대오차 |
|---|---|---:|---:|
| byte packing | 없음(bit-exact) | — | — |
| float32 그래프 실행(ORT CPU) | ±1, oracle 회전 좌표가 두 centroid 사이 경계에서 ≤1e-5 | ≤1e-5 | ≤1e-5 |
| HTP FP16 | ±1, 두 centroid 사이 경계에서 ≤1e-3 | ≤2e-3 | ≤5e-3 |

- index 허용 판정은 반드시 **실제로 바뀐 두 centroid 사이의 경계**까지 거리로 한다. 아무 경계나 가까우면 허용하는 방식은 반대 방향 kernel 오류를 통과시킨다.
- decode 오차는 벡터별 `||x̂_dev − decode_oracle(같은 packed, 같은 norm)||₂ / norm`이다. 최대 원소 기준을 쓰면 회전이 오차를 `1/sqrt(128)`로 퍼뜨려 centroid 1개 오류(약 2.2e-2)를 놓칠 수 있다.
- 영벡터는 norm과 복원값이 정확히 0이어야 하고, `unexplained_index_mismatches`(허용 범위 밖 index 차이)는 0이어야 한다.
- 이 판정들이 실제로 오류를 잡는지는 단위 테스트로 확인했다(반대 방향 flip, centroid 1개 오류 주입).

## 6. P2: S26 HTP 최소 codec 실행 결과

### 6.1 실행 경로 결정 (명세 7.1)

- **built-in op만으로 충분하다.** custom op package는 만들지 않았다.
- 사용 op: `Abs, ReduceMax, Greater, Where, Mul, Div, ReduceSum, Sqrt, MatMul(→FullyConnected), Reshape(→Transpose), Cast, Slice(→StridedSlice), Add, Gather`. 모든 텐서는 rank ≤4.
- QAIRT 2.48 HTP 제약과 대응은 다음과 같다(문서 확인, 일부 실측):
  - bitwise op이 없고 UINT_8 산술도 없다 → index는 `Greater` + INT32 `ReduceSum`, pack은 INT32 `hi*16+lo` 후 `Cast→UINT8`
  - UINT_8 Concat/Gather/Transpose와 fp16→uint8 Cast는 prepare에서 거부된다 → unpack은 `Cast(UINT8→FLOAT)` 뒤 정확한 fp16 산술(`b/16`의 `Floor`, 나머지)로 nibble을 풀고(마지막 축 64인 packed 레이아웃에서 계산; 마지막 축이 1인 텐서에서 하면 HTP가 약 3배 느림), centroid는 `y = c_0 + Σ_k (c_k − c_{k−1})·[idx ≥ k]`로 복원한다: 임계값 15개와의 브로드캐스트 `Greater` 1회 → `Cast` → K=15 `MatMul`(반올림 1회) → `Add`. 처음 구현은 `Cast(UINT8→INT32)` + `Gather([256,2] fp16 LUT)`였는데 SM8850 HTP의 `Gather`는 index 하나당 약 35 cycle(scalar 경로)이라 복원 cycle의 80%를 차지했다.
  - 8 head × 1023 token 복원 그래프(unpack + centroid + norm 보정 + 회전) 단독 accelerator 시간(S26, 3회 실행 중 마지막): `Gather` 9.7 ms, 16-entry `Gather`(nibble index) 20.9 ms, 2-byte(65536×4) LUT `Gather` 5.6 ms, `Greater`+`Where` 이진 트리 1.6 ms(leaf를 텐서 산술로 바꾼 profiling 호환판 1.8 ms), ladder `MatMul` 4.2 ms, ladder `Mul`+`ReduceSum` 22.7 ms. 모두 oracle과 HTP 허용오차 안(max rel 1.6e-3). Where 트리가 가장 빠르지만 KV 텐서당 op이 약 50개라 LLM part(codec 20개) 컨텍스트 컴파일이 그래프당 25분·host RAM 약 10 GB로 늘어 OOM으로 실패했고(값 입력이 둘 다 상수인 `Where`는 `--profiling_level detailed`에서 실행 실패), op 6개로 컴파일 부담이 작은 ladder `MatMul`을 채택했다. norm 보정 op을 encode 쪽으로 접는 것은 7%(5.48→5.11 ms) 이득이라 포맷을 바꾸지 않았다.
  - fp16에서 `sum(x²)`는 norm 약 256부터 overflow → max|x|로 먼저 나눈 뒤 제곱한다
  - **실측(SM8850)**: HTP FP16 `Div`는 나누는 값이 2^14를 넘으면 부정확했다(20000에서 norm 오차 22%, 36000 이상은 NaN). 그래서 max|x| > 256인 행은 먼저 정확한 2^-8을 곱해 나누는 값이 항상 ≤256이 되게 했다. 수정 후 max|x| 1e-2~6.5e4 전 범위에서 통과했다.
- 그래프 입력 계약: finite, `||x|| ≤ 65504`(fp16 norm 저장 한계).
- 실행기: `qnn-net-run` + `libQnnHtp.so` + offline context binary(soc_model 87). 이 조합에는 CPU partitioning 옵션이 없어 HTP가 못 돌리는 op이 있으면 prepare가 실패한다. `qairt-net-run --backend_order`는 사용하지 않았다.

### 6.2 결과

입력은 그래프마다 3케이스다. 실제 Qwen3-1.7B KV(layer 0, layer 27. T=128은 token 0–127, T=1은 token 128)와 범위 케이스(영벡터, one-hot, Gaussian, 상수 행, max|x| 1e-2~6.5e4, norm ≤6.5e4)를 넣었다. decode 입력은 oracle이 encode한 packed/norm이다. 비교 대상은 장치가 실제로 받은 fp16 반올림 입력이다.

| graph | 판정 | 최대 index 불일치율 | 설명 불가 불일치 | 최대 norm 상대오차 | 최대 decode 상대오차 | accelerator 실행 µs (warm median) | QNN 실행 µs |
|---|---|---:|---:|---:|---:|---:|---:|
| `tq_encode_key_t128` | 통과 | 4.063% | 0 | 1.58e-3 | – | 3890 | 4138 |
| `tq_decode_key_t128` | 통과 | – | – | – | 1.55e-3 | 1242 | 1454 |
| `tq_encode_key_t1` | 통과 | 15.527% | 0 | 8.99e-4 | – | 379 | 524 |
| `tq_decode_key_t1` | 통과 | – | – | – | 1.00e-3 | 345 | 520 |
| `tq_encode_value_t128` | 통과 | 0.172% | 0 | 1.69e-3 | – | 3925 | 4180 |
| `tq_decode_value_t128` | 통과 | – | – | – | 1.31e-3 | 1225 | 1378 |
| `tq_encode_value_t1` | 통과 | 12.695% | 0 | 7.19e-4 | – | 380 | 572 |
| `tq_decode_value_t1` | 통과 | – | – | – | 1.00e-3 | 349 | 530 |

- 실제 KV 케이스의 index 불일치율은 0~0.19%이고, 모두 경계 ±3e-4 이내의 인접 index였다. 장치 index로 복원한 rel MSE는 oracle과 소수 넷째 자리까지 같다(예: K layer0 0.00749 대 0.00749).
- 표의 높은 불일치율은 range 케이스의 영벡터와 교대 부호 상수 행 때문이다. 이 행들은 회전 좌표가 정확히 0이 되어 중앙 경계(−3.6e-17, fp16에서 −0)에서 동률이 생긴다(§4.3). 상수 행에서는 해당 좌표 부호가 oracle과 반대로 복원되지만 오차 크기는 같고, 장치 index로 복원한 rel MSE는 oracle과 같다(K t128 range 0.00532 대 0.00532).
- 실행 증거:
  - context metadata `dspArch 81 / socModel 87`, 16회 실행 모두 exit code 0
  - `--profiling_level detailed`에서 DLC의 **모든 op**이 `Accelerator (execute) time (cycles)` 아래에 기록됨. HVX thread 8
  - FullyConnected(MatMul)의 cycle은 0으로 표시되는데, HMX 실행으로 추정한다(미확인)
- 시간 수치는 프로파일링을 켠 단일 세션 관측값이다. 첫 실행(전원·VTCM 획득 포함)은 제외했으며 성능 벤치마크가 아니다. encode T=128에서는 `Greater`(15-way 비교)가 cycle의 약 58%, INT32 `ReduceSum`이 약 29%를 차지해 이후 최적화 대상이다.
- artifact 해시(ONNX/DLC/context/입력)는 작업 디렉터리의 `manifest.json`에 기록된다.

## 7. PC 참조 평가 (장치 결과 아님)

`evaluate_qwen3_kv.py`로 HF float32 모델에 float64 codec oracle을 붙여 측정했다. WikiText-2 test 비중첩 1024-token window 4개(4,092개 토큰 채점), 128-token chunk teacher forcing이다. 배포 delta-cache 그래프와 같은 계약을 따른다: attention은 이전 chunk의 복원 KV와 현재 chunk의 exact KV를 보고, 새 chunk만 encode한다. `k8_*` 행의 K는 float이다(배포 int8 K는 양자화 모델이 필요).

### 7.1 teacher-forced 품질

| 프로파일 | PPL | float KV 대비 | 평균 KL | top-1 일치 |
|---|---:|---:|---:|---:|
| float KV (기준) | 18.556 | — | — | — |
| `k8_v3` (K float) | 18.392 | −0.88% | 0.00370 | 97.8% |
| `k4_v4` | 24.154 | **+30.2%** | 0.185 | 86.1% |
| `k4_v3` | 23.945 | **+29.0%** | 0.192 | 86.0% |

### 7.2 KV 통계 (window 1, 28 layers)

| | K (4-bit) | V (4-bit) | V (3-bit) |
|---|---:|---:|---:|
| 평균 rel MSE | 0.0084 | 0.0091 | 0.0336 |
| 평균 / 최소 cosine | 0.9958 / 0.9779 | 0.9954 / 0.9743 | 0.9832 / 0.9516 |
| 최대 norm | 399.6 | 852.7 | 852.7 |
| HTP 경계 허용창(1e-3) 안의 좌표 비율 | 7.6% | 7.7% | 3.9% |

해석(PC proxy 한정):

- V는 4-bit와 3-bit 모두 이 규모의 측정에서 품질 저하가 드러나지 않았다.
- K 4-bit는 벡터 복원 오차가 V와 비슷한데도 PPL이 약 30% 나빠졌다. Q·K 내적 오차가 softmax에서 증폭되는 것으로 보인다.
- 참조 repo가 실사용 기본값으로 비대칭 K=q8_0 / V=turbo 구성을 두는 것과 방향이 같다.
- `k4_v4` 채택 여부는 K outlier 처리나 K bit 상향 같은 추가 실험과 합의가 필요하다. 합격 임계값은 사용자가 정하지 않았으므로 판정하지 않는다.

## 8. P3: Qwen3-1.7B 실기기 통합 (context 1024)

### 8.1 선택한 경로

- **재양자화 없음.** 배포된 w4a16 checkpoint(`qwen3_1_7b` asset v2, SpinQuant R1+R3 → AdaScale → Calibration)를 repo split 코드로 4개 part로 나눈다. 기존 weight와 activation encodings를 그대로 쓰고 KV 저장 방식만 바꿔 baseline과 비교한다.
- **ONNX surgery**(`graph_surgery.py`):
  - `past_{kind}_{L}_in`을 `tq_{kind}_{L}_{packed,norm}_in`으로 교체하고, 복원 서브그래프가 기존 Slice→Concat 소비자로 들어간다. K는 복원 뒤 token 축을 hub layout으로 되돌린다.
  - `past_{kind}_{L}_out`은 내부 텐서로 남기고, encode 서브그래프가 `tq_*_out`을 만든다.
  - codec 입력은 KV 전용 int8 양자화 이전의 값이다: tap encoding을 16-bit로 재격자화하고 cache 분기를 guard·복제로 다시 만든다(§11.3).
- **로컬 변환**(`convert_parts.py`): `qairt-converter --quantization_overrides` 뒤 `qairt-quantizer --enable_float_fallback --float_bitwidth 16 --act_bitwidth 16`.
  - encodings가 없는 codec 서브그래프는 fp16으로 남는다. QAIRT가 int8↔fp16 경계에 `Convert` op을 자동으로 넣고, codec 상수(회전·경계·LUT)는 fp16으로 유지됨을 dlc-info로 확인했다.
  - 둘째 part부터 prompt(AR=128)와 token(AR=1) 그래프를 weight-sharing context binary 하나로 묶었다. 그래프 변환 1개당 약 2분이 걸렸다.
- **전용 runner**(`qnn_runner/`):
  - SampleApp·Genie 소스는 QAIRT 라이선스상 소스 재배포가 허용되지 않아, 헤더의 공개 API만으로 새로 작성했다.
  - tensor 이름으로 역할을 정하므로 codec 스트림도 int8 KV와 같은 경로로 처리한다.
  - KV 배치는 ai-hub-models `HubCompatibleGenerator`와 같은 pyref 방식(오른쪽 정렬 past, 앞쪽 pad)이다.
  - RoPE는 Python에서 만든 float32 표를 쓰고, 양자화 규칙은 Python과 같다(round-half-even).
  - host는 KV를 packed 상태로 보관하고, 매 스텝 그래프 입력 버퍼로 복사한다. RAW client buffer를 쓴다.
- **codec 입력.** 배포 w4a16 그래프에서 K/V를 만드는 연산(K: SpinQuant R3 MatMul, V: v_proj Conv)의 출력은 16-bit 정수 activation이고 그 뒤 Convert에서 KV 전용 int8이 시작한다. codec은 그 int8 이전 값을 압축한다(§11.2). PC 참조 평가(§7)는 HF float K/V를 압축했으므로 입력이 다르다.

### 8.2 baseline 재현 확인

| 번들 | 생성 결과(35토큰 프롬프트) | WikiText window 0 PPL |
|---|---|---:|
| AI Hub 빌드 번들(QAIRT 2.45, 10 graph/part) | "Gravity is the force that pulls objects toward Earth." | 11.586 |
| 로컬 변환 baseline(QAIRT 2.48, 2 graph/part) | 같은 첫 문장, EOS 뒤 continuation은 다름 | 11.651 (+0.55%) |

같은 runner로 AI Hub 번들과 로컬 번들이 모두 동작했고, 로컬 변환이 baseline 품질을 재현했다. EOS 뒤 차이는 QAIRT 버전 간 수치 차이로 보이며, 원인은 검증하지 않았다.

### 8.3 명세 대비 남은 조건과 다음 단계

1. **메모리 경로(명세 5.3):** 복원이 그래프 안에서 전체 past에 대해 일어난다. 복원 결과(fp16)와 int8 변환 텐서가 HTP intermediate로 잡히므로, packed 저장만으로는 실제 peak 메모리 이득을 주장할 수 없다. tile 단위 복원이나 packed 소비형 attention 융합이 필요하다.
2. **속도:** 복원 서브그래프의 LUT `Gather`가 병목이었고, §6.1의 ladder `MatMul` 복원으로 바꿨다(§11.5에 전후 측정). Where 트리는 단독 그래프에서 2배 더 빠르지만 LLM 규모 컴파일이 실패해 보류했다. 그 뒤에도 매 스텝 과거 전체를 복원하는 구조는 같다. 남은 후보:
   - context 길이 bucket(예: 256/512/1024 token 그래프)으로 실제 길이만큼만 복원(runner가 그래프를 고르고 cache를 옮겨야 함)
   - Q를 회전 좌표계로 보내 K 복원의 회전 MatMul을 생략하고, AV 출력에서 역회전(attention 경로 encodings 재검토 필요)
   - custom HVX op(`TQDecodeTile`) 또는 attention과 융합
3. **runner:** shared buffer(rpcmem)와 smartmask KV 배치를 쓰면 baseline을 Genie 수준에 가깝게 비교할 수 있다. 명세의 "동일 runner 비교" 원칙은 현재도 지켜진다.
4. **P4:** context 512/2048/4096, 응답 품질 평가, HTP 측 메모리(`dumpsys meminfo`, QNN 메모리 이벤트), 전력 조건 기록.
5. **P5:** 0.6B/4B/8B는 같은 surgery·변환·runner로 적용 가능한 구조지만 실행하지 않았다.

## 9. 재현 명령

`src/`를 설치된 wheel보다 앞에 두기 위해 `PYTHONPATH=src`가 필요하다. 로컬 개발 환경에서는 venv에 `pytest`, `pre-commit`, `mypy`를 개발 의존성 버전으로 설치했고, gitignore된 `src/qai_hub_models/_version.py`가 있어야 `src/` import가 된다.

```bash
# 상수 생성/검사 (고정 commit의 참조 코드를 실행)
python scripts/llm/turboquant/generate_constants.py --reference-repo ~/git/turboquant_plus --check

# 참조 대조와 golden fixture
PYTHONPATH=src python scripts/llm/turboquant/verify_reference.py \
    --reference-repo ~/git/turboquant_plus --report /tmp/claude/turboquant/verify_reference.json

# 단위 테스트
python -m pytest src/qai_hub_models/test/test_models/test_turboquant_codec.py \
    src/qai_hub_models/test/test_models/test_turboquant_cache.py \
    src/qai_hub_models/test/test_models/test_turboquant_export.py

# PC 참조 평가와 KV snapshot (GPU 사용, HF 캐시 필요)
HF_HUB_OFFLINE=1 PYTHONPATH=src python scripts/llm/turboquant/evaluate_qwen3_kv.py \
    --model Qwen/Qwen3-1.7B --num-windows 4 \
    --report /tmp/claude/turboquant/qwen3_1_7b_eval.json \
    --snapshot ~/.qaihm/tmp/turboquant/qwen3_1_7b_kv_snapshot.npz

# P2: 빌드 → S26 실행 → 비교 (adb 장치, QAIRT 2.48 필요)
PYTHONPATH=src python scripts/llm/turboquant/htp_codec_validation.py all \
    --work-dir ~/.qaihm/tmp/turboquant/p2 \
    --snapshot ~/.qaihm/tmp/turboquant/qwen3_1_7b_kv_snapshot.npz
```

P3 (context 1024; 번들 이름과 경로는 예시):

```bash
# 배포 checkpoint 로컬 분할 (Workbench 사용 안 함)
HF_HUB_OFFLINE=1 PYTHONPATH=src python scripts/llm/turboquant/split_checkpoint.py \
    --model-id qwen3_1_7b --out ~/.qaihm/tmp/turboquant/qwen3_1_7b_w4a16_split

# 프로파일별 변환 (baseline_int8 | baseline_int16_kv | k4_v4); part 한 그래프당 약 2분
PYTHONPATH=src python scripts/llm/turboquant/convert_parts.py \
    --split-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_w4a16_split \
    --out ~/.qaihm/tmp/turboquant/qwen3_1_7b_k4_v4_cl1024 --context-length 1024 --profile k4_v4

# runner 빌드 (NDK r26c, QAIRT 헤더)
bash scripts/llm/turboquant/qnn_runner/build_android.sh

# 장치 asset, push, 실행
PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py assets \
    --checkpoint-dir ~/.qaihm/qai-hub-models/models/qwen3_1_7b/v2/qwen3_1_7b_w4a16 \
    --context-length 1024 --out ~/.qaihm/tmp/turboquant/device_assets_cl1024
PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py push \
    --bundle-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_k4_v4_cl1024 --name k4_v4_cl1024
PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py run --name k4_v4_cl1024 \
    --assets ~/.qaihm/tmp/turboquant/device_assets_cl1024 --mode generate --n-gen 128 --sessions 4 \
    --report ~/.qaihm/tmp/turboquant/reports/perf_k4_v4.json
PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py run --name k4_v4_cl1024 \
    --assets ~/.qaihm/tmp/turboquant/device_assets_cl1024 --mode score --tokens wikitext_w0.bin \
    --report ~/.qaihm/tmp/turboquant/reports/score_k4_v4_w0.json
# 추가 옵션: --stop-on-eos, --tokens boundary_prompt_cl1024.bin, --profile-decode-step N, --profile-prefill
```

`baseline_int8` 번들은 그래프 이름 수정 전에 변환되어 `--graph-suffix _float`로 실행했다. 이후 변환부터는 접미사가 붙지 않는다.

경계 검증과 push:

```bash
for p in baseline_int16_kv k4_v4; do
  PYTHONPATH=src python scripts/llm/turboquant/convert_parts.py \
      --split-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_w4a16_split \
      --out ~/.qaihm/tmp/turboquant/qwen3_1_7b_${p}_cl1024 --context-length 1024 --profile $p
  # KV 양자화 경계 검증 (dlc-info 기반, 위반 시 종료 코드 1)
  PYTHONPATH=src python scripts/llm/turboquant/verify_kv_boundary.py \
      --bundle ~/.qaihm/tmp/turboquant/qwen3_1_7b_${p}_cl1024 \
      --baseline-bundle ~/.qaihm/tmp/turboquant/qwen3_1_7b_baseline_int8_cl1024 \
      --report ~/.qaihm/tmp/turboquant/reports/boundary_${p}.json
  PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py push \
      --bundle-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_${p}_cl1024 --name ${p}_cl1024
done
# 장치 실행은 P3와 같은 인자(perf: --n-gen 128 --sessions 4, EOS: --stop-on-eos --sessions 2,
# 경계: --tokens boundary_prompt_cl1024.bin --sessions 2, 채점: --mode score --tokens wikitext_w{0..3}.bin,
# 프로파일: --n-gen 8 --profile-decode-step 3 --profile-prefill)
```

장치 파일은 `/data/local/tmp/qaihm_turboquant/` 아래에만 쓴다. 클라우드 작업은 제출하지 않는다.

## 10. 미지원 항목과 잔여 리스크

- 미지원: QJL, 3-bit NPU 그래프(3-bit는 host oracle만), batch>1, head_dim≠128, 128보다 긴 prefill chunk 그래프, native KV, 0.7B(현재 repo에 해당 checkpoint가 없음. 0.6B는 shape 테스트만 수행).
- HTP fp16 index는 경계 근처에서 oracle과 다를 수 있다(실제 KV에서 측정 ≤0.19%). byte 단위 재현이 필요한 용도에는 fp32 host 경로만 bit-exact를 보장한다.
- host cache(`cache.py`)는 `new == 128`일 때 shape로 K layout 실수를 잡지 못한다. 장치 runner는 I/O 이름으로 역할을 정하므로 이 문제가 없다.
- Hexagon SDK 6.6.0.0(tools 19.0.07)은 QAIRT 문서가 V81용으로 명시한 6.4.0/19.0.04와 다르다. custom op이 필요해지면 ABI 호환부터 확인해야 한다.
- 성능은 baseline보다 크게 느리고, HTP 측 peak 메모리는 측정하지 않았다(§11.5, §8.3).
- 변환한 context는 1024 하나다. 다른 context 길이, 3-bit 프로파일, 128 초과 prefill 청크는 실행하지 않았다.
- QAIRT 2.45(AI Hub)와 2.48(로컬) 사이 수치 차이로 EOS 뒤 생성이 달라진다. 비교는 반드시 같은 변환 경로의 번들끼리 해야 한다.
- §11의 codec cache 분기는 QAIRT 2.48의 encoding 전파·no-op 제거 동작에 맞춘 guard(`Max(x, x)`)에 의존한다. 다른 QAIRT 버전에서는 `verify_kv_boundary.py`로 경계를 다시 확인해야 한다. 이 경로는 배포 encodings를 파생시킨 것이며 재보정하지 않았다.
- encode 서브그래프의 `ReduceMax`는 part의 opset(18)에 맞춰 `axes`를 입력으로 넣는다. 초기 버전은 속성 형태였고 QAIRT는 받아들였지만 onnxruntime은 거부했다. P2 codec 그래프(opset 17)는 속성 형태가 맞다.

## 11. codec 입력 경계(int8 이전 KV)와 비압축 int16 KV 비교군 (context 1024)

배포 w4a16 그래프는 KV cache를 int8로 저장한다. 이 절은 codec이 **KV 전용 int8 양자화 이전의 K/V**를 읽도록 만든 경로를 정의·검증하고, 같은 값을 변환 없이 그대로 저장하는 **비압축 int16 KV** 비교군과 같은 w4a16 모델·runner·평가 조건에서 비교한다. decode 속도 최적화는 범위에 넣지 않았다.

### 11.1 비교군 정의

| 프로파일 | K 저장 | V 저장 | codec 입력 | 비고 |
|---|---|---|---|---|
| `baseline_int8` | 배포 affine int8 | 배포 affine int8 | — | 배포 경로 그대로. 성능·메모리 기준 |
| `baseline_int16_kv` | uFxp_16 (K 생성 연산의 16-bit grid 그대로) | uFxp_16 | — | KV cache 저장과 graph I/O만 바꿈. 값 변환 없음. 가중치·tokenizer·분할·비-KV activation encodings는 동일 |
| `k4_v4` | PolarQuant 4-bit | PolarQuant 4-bit | int8 이전 K/V | K는 k_norm·RoPE·SpinQuant R3 뒤 cache 좌표계에서 압축, V에는 RoPE 없음 |

"int8 이전 K/V"는 HF 전체 FP16 모델의 KV가 아니다. 같은 w4a16 그래프에서 KV 전용 int8 encoding이 붙기 직전 텐서이며, 이 텐서는 이미 16-bit 정수 activation(K: SpinQuant R3 MatMul 출력, V: v_proj Conv 출력)이며, `baseline_int16_kv`는 그 값을 그대로 저장한다. HF FP16 모델 값은 §7의 PC 참조 평가와 §11.6의 host 참고값으로만 쓴다. 모든 프로파일은 `config_hash()`가 다르고, 번들 경로는 `~/.qaihm/tmp/turboquant/qwen3_1_7b_{profile}_cl1024`로 분리된다. 배포 checkpoint는 덮어쓰지 않았다.

### 11.2 K/V 생성 → encoding → cache I/O → attention 소비 지도 (Qwen3-1.7B w4a16, Part2 layer 0)

배포 checkpoint의 사전 생성 ONNX·AIMET encodings(v1.0.0)와 QAIRT 변환 결과를 직접 추적했다(ONNX의 논리 dtype은 모두 FLOAT이므로 encodings와 dlc-info로 판단).

ONNX + encodings (head 0 기준, 8 KV head 동일 구조):

| 단계 | K | V |
|---|---|---|
| projection | `conv2d_16` k_proj Conv, **bw16** | `conv2d_24` v_proj Conv, **bw8** (scale 0.00769, per-head) |
| norm / RoPE | k_norm(`mul_2350`, bw16) → RoPE(`sub_881`/`add_2702`) → `cat_16` **bw16** | (없음) |
| tap (KV 전용 int8 시작) | `spinquant_block0_k_R3` MatMul(int4 Hadamard param) 출력 `spinquant_block0_k_R3_out` **bw8** (0.2765, per-head) | `conv2d_24` 자체 |
| 공유 Transpose | `transpose_1` bw8 (0.2765) | `permute_24` bw8 (0.00769) |
| cache 쓰기 | Concat → `past_key_0_out` bw8 (**0.4101**, head 최대값 공유) | Concat → `past_value_0_out` bw8 (**0.0218**) |
| cache 읽기 | `past_key_0_in` bw8 (0.4101) → Slice `slice_1..8` (동일 grid) | `past_value_0_in` bw8 (0.0218) → `slice_9..16` |
| attention 소비 | Concat(`slice_i`, `transpose_i`) → `cat_24..31` bw8 (예: 0.3115) → QK MatMul 2번째 입력 (16×8) | Concat(`slice_i`, `permute_i`) → `cat_32..39` bw8 → AV MatMul 2번째 입력 (16×8) |

이 int8은 `_apply_int8_kv_cache_tying_and_lm_head`가 KV I/O를 8-bit symmetric으로 묶고, `_set_matmul_second_input_to_8b`가 attention MatMul 2번째 입력을 8-bit로 만들며 Concat/Transpose/Slice를 거슬러 8-bit를 전파한 결과다. 전파가 멈추는 첫 연산 출력이 tap(K: R3 MatMul, V: v_proj Conv)이며, 그 입력(`cat_16`, layernorm 출력)은 16-bit다. 즉 `past_*_out` encoding만 지우면 per-head int8(tap·Transpose)이 그대로 남아 원본 값이 복원되지 않는다.

QAIRT 변환 결과(`baseline_int8`, token 그래프, dlc-info):

- K: `cat_16`(uFxp_16) → FullyConnected → `spinquant_block0_k_R3_out_fc` **uFxp_16** → `Convert → uFxp_8(0.2765)` → Reshape → Transpose(uFxp_8) → Concat `past_key_0_out`(uFxp_8, 0.4101) 및 Concat `cat_24`(uFxp_8, 0.3115).
- V: layernorm 출력(uFxp_16) → Conv2d(Transpose 접힘) → `permute_24` **uFxp_16** → `Convert → uFxp_8(0.00769)` → Concat `past_value_0_out`(0.0218) 및 `cat_32`(0.00769).

즉 장치에서도 tap 연산 자체는 16-bit로 계산되고, KV 전용 int8은 그 뒤 Convert 한 개에서 시작한다.

### 11.3 적용한 변경 (재보정 없음)

AIMET 재보정 대신 배포 encodings를 파생시켜 비-KV activation encodings와 param encodings(860개)를 bit 단위로 유지했다. `graph_surgery.apply_kv_profile`이 codec 프로파일(`k4_v4`)에 대해 part당 다음을 수행한다(part2 token 그래프, 10 layer 기준 수치):

| 편집 | 대상 | 수 | 이유 |
|---|---|---:|---|
| regrid 8→16 bit | tap `spinquant_block{i}_k_R3_out`(80), `conv2d_*`(80) | 160 | 같은 calibrated min을 유지한 16-bit grid(scale/256, offset×256). 연산은 정수로 유지되고 가중치는 그대로 |
| guard 삽입 | tap마다 `Max(x, x)` → `{tap}_tq_cache` | 160 | encoding 없는 정확한 no-op. float fallback으로 cache 분기를 float16으로 시작 |
| Transpose 복제 | `transpose_i`/`permute_i` → `*_tq_cache` | 160 | 원본 Transpose는 int8 encoding을 유지해 새 토큰의 attention 경로를 baseline과 동일하게 둠 |
| encoding 제거 | `past_*_in`, `past_*_out` | 40 | cache I/O. `past_*_out` Concat은 복제 체인만 읽도록 재배선 |
| 유지 | `slice_*`(cache 입력 tied grid), `cat_24..39`(attention per-head int8), 나머지 4534개 activation encodings | — | Slice는 float 입력을 받으면 QAIRT가 float16으로 두고 attention Concat 입력에서 int8로 변환(P3에서 검증). attention MatMul은 16×8 유지 |

`baseline_int16_kv`(비압축)는 codec 없이 저장 경로를 **무손실**로 만든다. float16 변환도 하지 않는다.

| 편집 | 대상 | 수 | 이유 |
|---|---|---:|---|
| regrid 8→16 bit (공유 grid) | tap 160개 | 160 | layer·kind마다 8개 head의 16-bit grid를 합집합(가장 넓은 head의 scale, offset −32768)으로 통일. head마다 grid가 다르면 cache Concat에서 재격자화가 생기므로 하나의 grid로 맞춰 저장을 정확히 만든다 |
| Transpose 복제 + encoding | `transpose_i`/`permute_i` → `*_tq_cache` | 160 | 복제본에 같은 공유 grid encoding을 명시. 원본은 int8 encoding을 유지해 attention 경로 불변 |
| encoding 설정 | `past_*_out`(20), `past_*_in`(20), cache 입력 Slice 출력(160) | 200 | 모두 같은 공유 grid. cache I/O가 uFxp_16이 되고 입력·출력 encoding이 동일하다 |
| 유지 | `cat_24..39`(attention per-head int8), 나머지 activation encodings | — | int8 변환은 attention Concat 입력 한 곳뿐 |

guard는 넣지 않는다(모든 텐서에 encoding이 명시되어 QAIRT의 전파·no-op 제거에 영향을 받지 않는다). 처음에는 이 비교군을 float16 cache로 만들었으나(QAIRT가 FLOAT encoding을 버려 converter `--config`로 출력 dtype을 강제), uFxp_16→Float_16 변환이 큰 값에서 int16 grid보다 거친 반올림(상대 ≤2^-11)을 넣으므로 무손실 int16 저장으로 바꿨다. 편집 목록은 번들의 `{graph}.kv_edits.json`에 tensor·이유·전후 encoding으로 기록된다.

시도했다가 버린 방법과 이유:

1. 공유 Transpose의 int8 encoding만 제거: QAIRT가 tap의 16-bit encoding을 pass-through op에 전파해 `past_*_out`이 uFxp_16이 되고, attention Concat `cat_24`는 두 입력이 모두 비-int8이 되자 float16으로 바뀌어 QK MatMul이 **16×16**으로 변했다(attention 소비 경로 이탈).
2. AIMET FLOAT encoding 명시: QAIRT 2.48 `convert_encodings.py`는 `dtype: FLOAT` encoding을 버린다("Discarding the float encodings").
3. guard 없이 Transpose만 복제: token 그래프에서 V의 Transpose는 no-op(`[1,1,1,128]`)이라 QAIRT가 제거하면서 cache Concat을 attention 쪽 int8 Convert에 연결했고, Concat 출력 encoding이 "bitwidth 8 + 16-bit grid"로 뒤섞였다(수치 파괴). `Max(x, x)` guard는 QAIRT의 identity 제거 규칙(상수 곱/나눗셈)에 걸리지 않아 유지된다.

### 11.4 변환 결과의 경계 검증 (`verify_kv_boundary.py`, dlc-info)

프로파일마다 prompt(AR=128)·token(AR=1) × part 2/3/4 = 6개 그래프, 그래프당 KV 항목 20/20/16개를 검사했다(리포트: `reports/boundary_{profile}.json`). 아래는 token 그래프 part2 layer 0의 최종 QNN op 열이다(다른 layer·part·prompt 그래프도 같은 구조, 위반 0).

`k4_v4` (codec):

- **cache 쓰기(K):** `cat_16`(uFxp_16) → FullyConnected `spinquant_block0_k_R3` → Reshape → `spinquant_block0_k_R3_out` **uFxp_16** (encoding `bitwidth 16, min -39.429, scale 0.001203, offset -32768`; baseline 번들의 같은 텐서는 `bitwidth 8, min -39.429, scale 0.308, offset -128` — min 동일, scale 1/256) → `Convert → Float_16` → guard `..._tq_cache_guard`(Eltwise_Binary, Float_16) → `node_transpose_1_tq_cache`(Float_16) → Concat → `past_key_0_out`(Float_16) → codec 첫 op(`tq_key_0_present_tokens_last` Transpose)이 Convert 없이 직접 읽음.
- **cache 쓰기(V):** layernorm 출력(uFxp_16) → Conv2d → `conv2d_24` **uFxp_16** → `Convert → Float_16` → guard → Concat → `past_value_0_out`(Float_16) → codec. token 그래프에서는 복제 Transpose가 no-op이라 QAIRT가 제거했지만 guard가 남아 float16 분기가 유지된다.
- **attention 새 토큰(K):** 같은 `spinquant_block0_k_R3_out`(uFxp_16) → 원본 `node_transpose_1` → `Convert → uFxp_8`(0.2765, baseline과 같은 per-head grid) → `cat_24`(**uFxp_8**, 0.3115) → QK MatMul(2번째 입력 8-bit, 16×8 유지). V도 `conv2d_24 → Convert uFxp_8(0.00769) → cat_32`로 동일.
- **cache 읽기:** codec 복원 텐서(Float_16) → StridedSlice(Float_16) → `Convert → uFxp_8` → `cat_24`(입력 두 개 모두 uFxp_8). 그래프당 160개(8 head × K/V × 10 layer)의 Convert가 모두 attention Concat 입력에 있다.
- `tq_*_packed_*`는 Uint_8, `tq_*_norm_*`는 Float_16이며 모두 "No encoding info"로 재양자화되지 않는다.

`baseline_int16_kv` (비압축):

- **cache 쓰기(K):** FullyConnected → Reshape → `spinquant_block0_k_R3_out` **uFxp_16** (공유 grid `bitwidth 16, min -48.754, scale 0.001488, offset -32768` — layer 0 K에서 가장 넓은 head의 grid) → `node_transpose_1_tq_cache`(uFxp_16, 같은 encoding) → Concat → `past_key_0_out` **uFxp_16, APP_READ, 같은 encoding**. 변환 op이 하나도 없다.
- **cache 쓰기(V):** Conv2d → `conv2d_24` **uFxp_16**(공유 grid `min -2.538, scale 7.744e-5`) → Concat `node_cat_1149` → `past_value_0_out` **uFxp_16**, 같은 encoding. no-op 복제 Transpose는 제거됐지만 tap 자체가 공유 grid라 결과는 같다.
- **cache 읽기:** `past_key_0_in` **uFxp_16, APP_WRITE, 출력과 동일한 encoding** → StridedSlice `slice_1`(uFxp_16, 같은 grid) → `cat_24`. 여기서 QAIRT는 입력이 모두 16-bit가 되자 `cat_24`를 **uFxp_16**(범위 ±39.87, baseline `cat_24`와 같은 범위)으로 승격하고, `cat_24 → Convert → uFxp_8(0.3115)` 뒤에 QK MatMul을 둔다. 즉 int8 변환이 Concat 입력에서 Concat 출력으로 옮겨졌을 뿐 MatMul은 16×8이고 int8 grid도 baseline과 같다. 새 토큰 K는 원본 `transpose_1`이 16-bit(per-head 범위)로 승격되어 `cat_24`로 들어가므로, baseline의 "per-head int8(0.2765) → cat_24 int8(0.3115)" 두 번 반올림이 한 번(0.3115)으로 줄어든다. V(`cat_32`)도 같다.
- I/O 표: `past_*_in`/`past_*_out` 모두 uFxp_16이고 encoding 문자열이 동일하다(runner의 in/out 일치 검사 통과). runner가 보고한 KV stream dtype은 `ufxp16`.

`baseline_int8`과 비교하면 새 토큰 경로에서 tap 연산 출력이 uFxp_16인 점은 같고, 차이는 (1) cache 저장이 int8(공유 scale)이 아니라 tap의 16-bit grid 그대로, (2) attention 입력의 int8 변환이 한 번인 점이다.

- **실행 증거:** context binary metadata `dspArch 81 / socModel 87`, HTP backend(CPU 분할 옵션 없음). runner detailed profile에서 `baseline_int16_kv` decode 1스텝 accelerator 시간은 part2/3/4 = 6.18/5.32/10.14 ms(baseline_int8 5.55/5.77/10.06 ms)로 모든 op이 accelerator에서 실행됐다.

ONNX 쪽 증거는 `{graph}.kv_edits.json`(tap·복제·guard·set 목록)과 단위 테스트(`test_turboquant_graph_surgery.py`: 합성 part에서 attention Concat 입력 배선·encoding 유지·복제 체인·값 동일성·int16 grid 공유 검사)다.

### 11.5 실기기 결과 (S26, CL=1024, 각 1회)

조건: 같은 w4a16 가중치·tokenizer·4-part 분할, QAIRT 2.48 로컬 변환, 같은 runner(RAW buffer, pyref 전체 복사)와 burst 전력 설정, 35토큰 chat prompt(`enable_thinking=False`), 128토큰 greedy, WikiText 4 window/4,092 채점 토큰. 속도·메모리는 **프로파일당 1세션 1회**(warmup 없음) 측정이라 P3의 "warmup 뒤 3회 median"보다 편차가 크다(예: `baseline_int8` decode는 3회 median 38.97 tok/s였고 이번 1회는 41.75 tok/s). 리포트는 `reports/once/`. TTFT는 prefill 시작부터 첫 argmax까지이며 모델 로딩은 제외한다. RSS는 host 프로세스 기준이고 HTP·DMA-BUF 측 메모리를 포함하지 않는다. runner는 RAW client buffer와 pyref 전체 복사를 쓰므로 baseline 수치도 Genie 수준의 최적 경로가 아니다.

| 지표 | `baseline_int8` | `baseline_int16_kv` | `k4_v4` |
|---|---:|---:|---:|
| KV 저장 형식 | uFxp_8 (공유 scale) | uFxp_16 (tap grid 그대로, 무손실) | 4-bit packed + fp16 norm |
| TTFT | 54 ms | 40 ms | 1263 ms |
| prefill (35토큰) | 648 tok/s | 877 tok/s | 27.7 tok/s |
| decode | 41.75 tok/s (24.0 ms/tok) | 34.72 tok/s (28.8 ms/tok) | 1.96 tok/s (512 ms/tok) |
| host KV 저장소 (1024토큰) | 56.0 MiB | 112.0 MiB | 28.9 MiB |
| runner I/O 버퍼 | 151.2 MiB | 263.2 MiB | 96.9 MiB |
| 프로세스 VmRSS (종료 시) | 221.8 MiB | 391.7 MiB | 144.0 MiB |
| WikiText PPL (4 window 통합) | 19.903 | 20.108 | 20.415 |
| PPL vs `baseline_int8` | — | +1.03% | +2.57% |
| PPL vs `baseline_int16_kv` | −1.02% | — | +1.53% |
| window별 PPL | 11.651 / 23.729 / 22.017 / 25.781 | 11.545 / 24.098 / 21.856 / 26.888 | 12.637 / 24.182 / 21.309 / 26.674 |

`k4_v4`는 §6.1의 ladder `MatMul` 복원으로 다시 변환·측정한 번들이다(이전 `Gather` 복원 번들은 decode 1.43 tok/s, TTFT 729 ms였다). PPL은 teacher-forced라 세션 반복과 무관하다. `baseline_int8`의 PPL은 번들 불변이라 이전 값을 쓴다. 세 프로파일 모두 128토큰 생성 완료, `--stop-on-eos`에서 11번째 토큰 `<|im_end|>` 정지, 897토큰 프롬프트 + 128토큰으로 KV 1024/1024 채움을 1세션에서 확인했다. 고정 프롬프트의 첫 문장은 모두 "Gravity is the force that pulls objects toward Earth."로 같았고 EOS 뒤 continuation은 다르다. 이것으로 품질 동등을 주장하지 않는다. HTP backend에는 CPU 분할이 없다.

관찰:

- `baseline_int16_kv`(무손실 16-bit 저장)는 int8 KV baseline보다 PPL이 **1.03% 나쁘다.** 이전에 float16으로 저장했을 때(+1.35%; §11.3)보다 차이가 줄었지만 부호는 같다. 즉 int8 KV 단계를 제거하는 것 자체가 이 모델에서는 PPL을 낮추지 않는다. 원인은 검증하지 않았다. 후보: 배포 checkpoint의 AdaScale·calibration이 int8 KV quantizer를 켠 채로 수행되어 하류 encodings가 그 분포에 맞춰져 있음, 그리고 attention 입력 int8 변환이 baseline과 다른 위치(Concat 출력)에서 한 번만 일어남(§11.4). "16-bit KV가 곧 상한"이라는 가정이 이 모델·양자화 경로에서는 성립하지 않으므로 비교는 두 baseline 모두에 대해 제시한다.
- `k4_v4`(K·V 4-bit)는 `baseline_int8` 대비 +2.57%, `baseline_int16_kv` 대비 +1.53%다. window 0(12.637)에서 가장 나쁘고 window 2에서는 두 baseline보다 낮다. 이전 `Gather` 번들(+1.97%)보다 약간 커졌는데, ladder `MatMul` centroid 복원이 반올림을 1회 추가하기 때문이다(§6.1). 단일 실행·4 window이므로 유의성을 주장하지 않고, 응답 품질 평가는 수행하지 않았다.
- decode 속도: ladder `MatMul` 복원으로 `k4_v4` decode가 **1.43 → 1.96 tok/s(698 → 512 ms/tok)**로 빨라졌다(§6.1 단독 복원 그래프 9.7 → 4.2 ms). 여전히 `baseline_int8`의 약 1/21이며, 매 스텝 과거 1023토큰 전체를 복원하는 구조가 남은 병목이다(§8.3). 이번 1회 실행에서 TTFT(729 → 1263 ms)와 prefill(48 → 27.7 tok/s)은 오히려 나빠졌다: ladder는 임계값 축(15)을 브로드캐스트한 `[heads, past, 128, 15]` 중간 텐서를 만들어 128토큰 청크의 prefill에서 메모리·대역폭 비용이 크다. decode(새 토큰 1개, 과거 복원)에서는 이 비용보다 `Gather` 제거 이득이 커서 순이득이다. 이 prefill 회귀는 단일 실행이라 재확인이 필요하며, 필요하면 prefill 그래프만 `Gather`를 유지하는 혼합도 가능하다.
- `baseline_int16_kv`는 accelerator 시간이 baseline과 비슷하지만(§11.4) host KV 복사량이 2배(run당 1,379 vs 690 MiB)라 end-to-end decode는 느리다.
- 메모리: host 값만 측정했다. HTP intermediate·scratch·DMA-BUF는 **미측정**이다. int16 KV의 이론 payload(112 MiB)와 실측 NPU 메모리는 다른 값이다.

### 11.6 host 참고값 (장치 결과 아님)

HF Qwen3-1.7B의 float K/V snapshot(layer 0/13/27, 8 head × 256 token)에 4-bit codec을 적용한 상대 L2 재구성 오차(`reports/host_snapshot_input_grid.json`). int8 grid는 텐서 자체 max로 잡은 유리한 per-tensor symmetric grid다.

| 입력 | K (layer 0/13/27) | V (layer 0/13/27) |
|---|---|---|
| int8만 | 0.050 / 0.029 / 0.034 | 0.042 / 0.096 / 0.031 |
| codec, float 입력 | 0.085 / 0.094 / 0.090 | 0.094 / 0.095 / 0.096 |
| codec, int8 입력 (float 대비) | 0.098 / 0.098 / 0.096 | 0.103 / 0.135 / 0.100 |

codec 자체 오차(≈9%)에 int8 오차가 대략 제곱합으로 더해진다. 장치 K는 R3 회전 뒤 값이고 int8 grid도 다르므로 이 표는 경향 참고용이다.

## 12. 2026-09-17 성능 수정: centroid 복원·비교 축·head 배치

### 12.1 원인과 변경

`k4_v4`의 KV 저장 용량 감소는 attention 계산량 감소를 뜻하지 않는다.
AR=1/context=1024에서도 `decode_subgraph`는 유효 캐시 길이와 무관하게
각 K/V의 과거 1023토큰을 복원한다. 28 layer × K/V × 8 heads × 1023 × 128
= 58,662,912개 값을 매 생성 토큰마다 복원하고, norm 보정과 dense inverse
rotation을 수행한 다음 attention의 int8 경계로 넘긴다. host에 저장된
packed KV는 작아도 그래프 내부에는 전체 FP16 KV가 생긴다. WSL RAM 증설은
변환 작업의 OOM에는 도움이 되지만 이 장치 실행 비용을 없애지는 않는다.

동일 장치에서 기존 ladder 번들을 다시 실행한 마지막 세션의 decode step
평균은 host prepare 1.488ms, commit 0.063ms, 4개 QNN 호출 합계 508.067ms였다
(`perf_k4_v4_before_affine.json`). host KV 복사보다 장치 그래프 실행이 지배적이다.
logits 처리까지 포함한 decode는 3회 세션 중앙값 1.94895 tok/s,
4개 window 합산 PPL은 20.414824로 기존 보고를 재현했다.

추가 병목은 ONNX 연산의 배치였다.

- **Encode:** 기존 비교 텐서 `[heads, tokens, 128, 15]`를
  `[heads, tokens, 15, 128]`로 변경했다. 마지막 축에 128개의 head 성분을
  유지해 HTP의 벡터 연산이 가능하도록 하고, INT32 합산 축만 함께 옮겼다.
- **Decode:** 15개 임계값을 브로드캐스트하는 ladder MatMul을 제거했다.
  대칭 codebook의 index를 `abs(index - 7.5) - 0.5`로 0..7에 접고,
  인접 centroid 두 개씩을 잇는 네 개 선형식 중 하나를 선택한 뒤 부호를
  복원한다. 정수 index 0..15에서는 원래 codebook lookup과 같은 수식이다.
  FP16 연산 순서는 달라지므로 bit-exact 주장이 아니라 기존 수치 tolerance로
  검증한다. 15-fold 중간 텐서가 없어져 1023토큰, 8 head 복원에서 최대 단일
  FP16 중간 텐서는 약 30MiB에서 2MiB로 줄었다(전체 peak allocation 수치는 아님).
- packed byte 순서, codebook, 회전 seed/행렬, norm 보정, cache ABI와
  `config_hash()`는 유지했다. 새 byte 포맷이나 host CPU codec은 추가하지 않았다.
- **실제 모델 배치:** standalone `[1, heads, tokens, dim]`과 달리 모델 ABI는
  `[heads, 1, tokens, dim]`이다. 같은 affine decoder도 후자에서는 HTP가
  heads를 batch로 처리하여 1023토큰 K 복원이 1.557ms에서 7.477ms로 느려졌다.
  codec 내부를 `[1, heads, tokens, dim]`으로 정규화하고 경계에서만 element
  순서를 보존하는 Reshape를 넣었다. packed 입력은 UINT8 Transpose 제약을
  피하도록 float Cast 후 reshape한다. 입출력 ABI와 runner는 그대로다.
  실제 모델 ABI의 K/V 복원 시간이 1.875/1.900ms로 줄었고, 수치 검증을
  통과했다. batch 변경은 codec 내부에만 적용하며 attention 배치는 바꾸지 않는다.

### 12.2 단독 codec 검증

동일 S26/SM8850, QAIRT 2.48, HTP backend, burst 설정. 아래는 standalone
`qnn-net-run` basic profile의 마지막 실행 accelerator 시간이며, 전체 LLM
속도로 환산하지 않는다. 이 표는 `[1, heads, tokens, dim]` standalone
배치의 진단 결과이고, 실제 ABI 결과는 아래에 별도로 기록한다.

| 경로 (8 heads) | 기존 ladder | 수정 후 |
|---|---:|---:|
| K encode, 128토큰 | 3.910ms | 0.622ms |
| V encode, 128토큰 | 3.916ms | 0.608ms |
| K decode, 128토큰 | 0.849ms | 0.467ms |
| V decode, 128토큰 | 0.852ms | 0.457ms |
| K decode, 1023토큰 | 약 4.2ms (§6.1) | 1.557ms |
| V decode, 1023토큰 | — | 1.553ms |

1/128토큰의 encode/decode 8개 그래프와 1023토큰 decode 2개 그래프 모두
실제 KV 및 full-FP16-range 입력에서 기존 tolerance를 통과했다. 1023토큰의
최대 decode 상대 오차는 K=0.001521, V=0.001409(기준 0.005). basic와 detailed
실행 및 DLC op의 HTP profile 확인도 통과했다. 긴 codec 입력은 기록된
256토큰 snapshot의 벡터를 반복한 것이며, 모델의 긴 문맥 품질 검증과 다르다.

근거: `~/.qaihm/tmp/turboquant/{p2_ladder,p2_affine,p2_affine_1023}/p2_report.json`.
실제 모델 ABI로 최종 소스를 재검증한
`{p2_optimized_hub,p2_optimized_hub_1023}/p2_report.json`에서는
K/V encode 128토큰 0.621/0.609ms, decode 128토큰 0.478/0.470ms,
decode 1023토큰 1.875/1.900ms였다. 이 마지막 검증의 decode 상대 오차는
K=0.00152047, V=0.00140818로 기준 0.005 이내다.

전체 byte 값, K/V, norm correction on/off, zero/큰 norm, decoder 중간 텐서
크기, encode 비교 축, 실제 Hub I/O와 내부 batch 배치를 포함한 관련 테스트
158개를 통과했다. `htp_codec_validation.py --head-major`로 모델과 같은 I/O를
검증할 수 있다. `[1, heads]` 단독 수치를 모델의 `[heads, 1]` 경로에 그대로
대입하면 이 배치 병목을 놓치게 된다.

### 12.3 전체 모델 실측

S26/SM8850, CL=1024, 같은 가중치·runner·burst 설정, 35토큰 prompt,
128토큰 greedy, **각 번들 3개 연속 세션의 중앙값**이다. 프로파일링은 끄고
측정했고 모델 로딩은 TTFT에서 제외한다. 별도 warmup 제외나 온도 통제·
실행 순서 교차는 하지 않았으므로 체계적인 P4 결과가 아니라 이번 수정의
동일 장치 재측정이다. baseline 두 종류와 수정 전 ladder 번들도 재실행했다.
PPL은 같은 WikiText 4 window의 NLL 합계 / 4,092토큰으로 산출했다.

| 지표 | int8 baseline | int16 KV baseline | 수정 전 `k4_v4` | 수정 후 `k4_v4` |
|---|---:|---:|---:|---:|
| TTFT | 49.4ms | 35.5ms | 1269.5ms | **117.8ms** |
| prefill | 710.7 tok/s | 990.0 tok/s | 27.59 tok/s | **299.12 tok/s** |
| decode | 41.795 tok/s | 32.425 tok/s | 1.949 tok/s | **10.759 tok/s** |
| decode 범위 (3세션) | 41.776–42.152 | 32.230–32.458 | 1.946–1.952 | 10.742–10.965 |
| host KV 저장소 | 56.0MiB | 112.0MiB | 28.875MiB | 28.875MiB |
| 프로세스 종료 VmRSS | 223.2MiB | 391.4MiB | 143.9MiB | 144.2MiB |
| PPL (4 window) | 19.903103 | 20.108306 | 20.414824 | 20.463505 |

수정 전 대비 decode **5.52배**, prefill **10.84배**, TTFT **10.78배** 개선했다.
PPL은 **+0.238%**로 약간 나빠졌다(int8 baseline 대비 +2.816%, int16 대비
+1.766%). window별 PPL은 12.591734 / 24.004772 / 21.468385 / 27.023249다.
codebook·ABI는 같지만 FP16 실행 순서·배치가 달라졌으므로 품질이 완전히
같다고 주장하지 않는다. 4 window만으로 응답 품질이나 통계적 유의성을
판정하지 않는다. VmRSS는 host 값이며 HTP/DMA-BUF peak memory는 아니다.

변경을 분리한 전체 모델 중간 측정도 수행했다. centroid 복원과 비교 축만
수정한 번들은 decode 2.809 tok/s, TTFT 364.1ms였다. 실제 모델의 head 배치를
추가로 정규화한 뒤 10.759 tok/s, 117.8ms가 됐다. 따라서 standalone codec의
빠른 결과만으로 모델 성능을 예측하면 실제 배치 문제를 놓친다.

최종 번들의 마지막 세션에서 decode step 평균 QNN 호출 합계는
**89.257ms**(수정 전 508.067ms), host prepare/commit은 1.185/0.043ms였다.
여전히 int8 baseline보다 decode가 **3.88배 느리다.** 과거 1023토큰 전체의
dequantization·norm 보정·dense inverse rotation이 매 layer/step에 남아 있기
때문이다. 압축은 현재 **호출 사이의 host KV 저장량**을 줄이는 기능이지
packed KV를 직접 소비하는 attention 구현이 아니다. baseline 수준의 속도를
목표로 한다면 다음 단계는 유효 길이에 맞춘 그래프 버킷 또는 packed decode와
attention의 fused/tiled HTP 구현이며, 그 성능은 이번 결과로 입증하지 않았다.

기능 검증:

- 3세션 각각 128토큰 생성 완료, reset 뒤 생성 token ID 완전 일치.
- EOS 처리 2세션 모두 11번째 토큰 `<|im_end|>`에서 정지, token ID 일치.
- 897토큰 prompt + 128토큰 생성으로 cache 1024/1024 도달, 2세션 token ID 일치.
- prompt/token × part 2/3/4의 KV 양자화 경계 검사 통과: packed UINT8와
  FP16 norm ABI, codec 이전 16-bit 경계와 attention int8 경계를 유지한다.
- 별도 detailed profile의 prefill/decode 양쪽에서 28개 layer 전체의 codec
  op이 HTP에 기록됨. decode part 2/3/4의 accelerator execute 시간은
  28.114/27.940/28.729ms이며, codec 이름의 op cycle 합계는 각 part 전체
  op cycle 합계의 82.2/82.2/77.9%였다(벽시계 시간의 비중은 아님).
  detailed profiling은 큰 계측 오버헤드가 있으므로 그 실행의 TTFT·tok/s를
  위 성능 표에 섞지 않았다.
- 관련 단위 테스트 158개 및 변경 파일의 pre-commit 검사 통과.

산출물은 `~/.qaihm/tmp/turboquant/` 아래에 있으며, 원래 번들은 보존했다.

- 최종 번들: `qwen3_1_7b_k4_v4_optimized_cl1024/`;
  device bundle: `k4_v4_optimized_cl1024`.
- `reports/perf_{baseline_int8_affine_control,baseline_int16_affine_control,k4_v4_before_affine,k4_v4_optimized}.json`:
  세션별 측정, step별 host/QNN 시간, host 메모리.
- 같은 이름의 `quality_*.json`: window별 NLL과 통합 PPL.
- `reports/generation_{eos,boundary}_k4_v4_optimized.json`: reset/EOS/context 경계.
- `reports/boundary_k4_v4_optimized.json`: 변환된 그래프의 양자화 경계 검사.
- `reports/profile_k4_v4_optimized.json`: 별도 prefill/decode HTP op profile.

### 12.4 재현

기존 번들을 덮어쓰지 않고 수정된 소스로 재변환해야 한다.

```bash
PYTHONPATH=src python scripts/llm/turboquant/convert_parts.py \
    --split-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_w4a16_split \
    --out ~/.qaihm/tmp/turboquant/qwen3_1_7b_k4_v4_optimized_cl1024 \
    --context-length 1024 --profile k4_v4
PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py push \
    --bundle-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_k4_v4_optimized_cl1024 \
    --name k4_v4_optimized_cl1024
PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py run \
    --name k4_v4_optimized_cl1024 \
    --assets ~/.qaihm/tmp/turboquant/device_assets_cl1024 \
    --mode generate --n-gen 128 --sessions 3 \
    --report ~/.qaihm/tmp/turboquant/reports/perf_k4_v4_optimized.json
```

## 13. 출처

turboquant_plus(Copyright 2026 Tom Turney, Apache-2.0, https://github.com/TheTom/turboquant_plus, commit `ba52ad1`). 이 구현은 참조 코드를 복사하지 않고 알고리즘을 재구현했다. 참조를 실행해 얻은 codebook·sign 상수와 golden fixture에는 출처와 commit을 기록했다.
