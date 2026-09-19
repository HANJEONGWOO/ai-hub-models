# Qwen3 TurboQuant KV-cache — 설계(ABI·수치 계약)와 P0–P3 결과

> 현재 기본 회전은 dense QR이며 QJL은 off다(§16). §15까지의 실측은 WHT 경로의 이력이다.

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
| `Rotation.FWHT` (명시적 선택) | `R = D2·H·D1`, 역방향 `D1·H·D2`, `H`는 Sylvester 순서·`1/sqrt(128)` 정규화 | 없음 | 이전 실측 재현, `--rotation fwht` |
| `Rotation.DENSE_QR` (기본) | 참조 `PolarQuant`와 같은 Haar QR(`default_rng(seed)`, sign·det 보정) | 없음 | 실사용 codec·NPU 그래프·Python oracle |
| QJL | — | — | 1차 범위 밖. `get_profile("qjl_reference")`는 `NotImplementedError` |

참조 Python `PolarQuant`는 dense QR만 사용하며 FWHT 구성은 참조 repo에 클래스로 존재하지 않는다. FWHT codec은 참조 `rotation.random_rotation_fast`/`apply_fast_rotation*` 조각을 조합한 것이고, 이 조합을 참조 조각과 직접 대조했다(§4.4).

### 4.2 상수와 seed

- codebook: 참조 `codebook.optimal_centroids(bits, 128)`(N(0, 1/d) Gaussian 근사 Lloyd, quantile 초기화, **정확히 100회 반복**). 4-bit 최외곽 centroid는 0.24021009724443104이다. `turbo4-resurrection.md`의 표(0.1739)는 반복 0회 초기값이고, 수렴 Lloyd-Max(0.241529)와도 다르다. 둘 다 사용하지 않는다.
- FWHT sign: `random_rotation_fast(128, np.random.default_rng(seed))`에서 `signs1`을 먼저, `signs2`를 나중에 뽑는다. seed 정책은 참조 `KVCacheCompressor`를 따라 K=42, V=542이다.
- codebook/FWHT 상수는 float64 hex로 고정하고 sha256을 기록한다. Dense QR은 host에서 생성·캐시하고 실제 export float32 행렬의 sha256을 기록한다. NPU runtime에서는 QR/RNG를 실행하지 않는다.

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
    --rotation fwht \
    --model Qwen/Qwen3-1.7B --num-windows 4 \
    --report /tmp/claude/turboquant/qwen3_1_7b_eval.json \
    --snapshot ~/.qaihm/tmp/turboquant/qwen3_1_7b_kv_snapshot.npz

# P2: 빌드 → S26 실행 → 비교 (adb 장치, QAIRT 2.48 필요)
PYTHONPATH=src python scripts/llm/turboquant/htp_codec_validation.py all \
    --rotation fwht \
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
    --out ~/.qaihm/tmp/turboquant/qwen3_1_7b_k4_v4_cl1024 --context-length 1024 --profile k4_v4 --rotation fwht

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
      --out ~/.qaihm/tmp/turboquant/qwen3_1_7b_${p}_cl1024 --context-length 1024 --profile $p --rotation fwht
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
    --context-length 1024 --profile k4_v4 --rotation fwht
PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py push \
    --bundle-dir ~/.qaihm/tmp/turboquant/qwen3_1_7b_k4_v4_optimized_cl1024 \
    --name k4_v4_optimized_cl1024
PYTHONPATH=src python scripts/llm/turboquant/run_device_llm.py run \
    --name k4_v4_optimized_cl1024 \
    --assets ~/.qaihm/tmp/turboquant/device_assets_cl1024 \
    --mode generate --n-gen 128 --sessions 3 \
    --report ~/.qaihm/tmp/turboquant/reports/perf_k4_v4_optimized.json
```

## 13. 2026-09-17 tiled KV 복원과 attention

### 13.1 구현 범위

`tiled_attention.py`는 codec 삽입 후 Qwen3의 head별 attention을 검사하고
다음 **two-pass tiled 그래프**로 바꾼다. `convert_parts.py --attention-tile 256`으로
명시적으로 켜며, 기본값 0은 §12의 전체 복원 경로를 유지한다.

1. packed K와 norm을 token 축에서 최대 256토큰씩 Slice한다. 해당 블록만
   복원하고 head별 QK를 계산한다. 복원 K를 전체 캐시로 합치지 않는다.
2. 작은 QK score 블록만 context 축으로 합쳐 기존 mask와 **전역 softmax**를
   그대로 적용한다. block별 softmax를 따로 정규화하는 잘못된 계산은 하지 않는다.
3. packed V와 norm도 같은 크기로 Slice한다. 각 V 블록을 복원해 해당 확률
   블록과 곱한 부분 AV를 만들고 누적한다. 복원 V도 전체 캐시로 합치지 않는다.

이것은 **그래프 수준 tiling**이며, 하나의 HTP custom kernel로 decode·attention을
fusion하거나 online softmax까지 구현한 FlashAttention은 아니다. 중간 텐서의
실제 배치·생존 기간·VTCM/DDR 이동은 QAIRT 스케줄러가 결정한다. 따라서
"전체 FP16 KV 텐서 제거"와 "peak NPU memory/latency 감소"를 구분해 검증한다.
고정 context의 모든 past slot은 여전히 처리하며 유효 길이 버킷은 추가하지 않았다.

캐시 ABI, MSB-first packed bytes, FP16 norm, encoder, codebook, 회전 행렬,
가중치와 param encodings는 바꾸지 않는다. tile 크기는 cache format이 아닌
컴파일 옵션이다. 인식하지 못한 attention 구조·누락된 encoding은 오류로
거부하며 일부 head만 바꾸고 조용히 계속하지 않는다.

### 13.2 양자화 경계와 수치 차이

각 tile의 per-head KV Slice와 attention Concat에는 원본 Slice/Concat의
encoding을 모두 복사한다. **Concat encoding만 복사한 초기 시도에서는
QAIRT가 attention을 16×16으로 승격했다.** Slice encoding을 함께 유지한 뒤
실제 모델에서 추출한 attention 그래프의 모든 QK·AV가 16×8임을 확인했다.
새 토큰은 기존 attention 입력에서 가져오며 캐시 encoder 경로는 그대로다.

HTP의 혼합 dtype Concat 경계를 유지하기 위해 각 KV tile 뒤에 현재 토큰을
붙인다. 마지막이 아닌 K tile에서는 현재 토큰의 score를 제외하고, V tile에서는
현재 토큰에 대응하는 확률을 0으로 둔다. 현재 토큰의 기여는 마지막 tile에서만
포함된다. mask와 softmax의 입력 순서는 기존 past → current 순서다.

QK tile과 부분 AV는 원래 MatMul의 calibrated output grid를 재사용한다.
부분 AV의 재양자화와 누적 순서 때문에 FP16/정수 반올림은 원래 한 번의 AV와
다르다. float32 ONNX의 수학적 동등성을 실기기 품질 동등성으로 간주하지 않고,
최종 바이너리의 WikiText PPL을 별도로 측정한다.

`verify_tiled_attention.py`는 최종 DLC op 표에서 각 tile의 KV 쓰기/읽기 경계,
packed/norm I/O, 모든 attention MatMul의 16×8 dtype, 전체 복원 텐서의 부재와
복원 중간 텐서의 크기를 검사한다. 빈/미완성 DLC op 표는 실패로 판정한다.
CPU 검증은 빈/부분/전체 캐시, causal mask, GQA, 불완전한 마지막 tile,
cache I/O 유지, 미지원 구조 거부를 포함한다. 기존 테스트와 합쳐 185개가 통과했다.

### 13.3 재현 및 측정 규칙

명령은 `scripts/llm/turboquant/README.md`의 "Tiled KV restore and attention"에
있다. 이 변경부터 성능 수치는 사용자 요청에 따라 **구성당 128토큰 생성 1회**로
측정한다. median/반복 범위로 표시하지 않는다. 품질·EOS·context 경계·세션 reset
검증 및 detailed profiling은 성능 측정과 분리한다.

### 13.4 실기기 결과: 중간 텐서 제거, 속도 개선 없음

S26/SM8850, CL=1024, 같은 가중치·runner·burst 설정, 35토큰 prompt,
128토큰 greedy 생성이다. **아래 네 구성 모두 이번 작업에서 성능을 1회씩
측정했다.** 프로파일링은 껐고 모델 로딩은 TTFT에서 제외했다. 별도 warmup 제외,
온도 통제, 실행 순서 교차는 하지 않았다. 약 0.5–2% 차이의 유의성은 판단할 수 없다.

| 지표 | int8 baseline | int16 KV baseline | 기존 최적화 `k4_v4` | tiled-256 `k4_v4` |
|---|---:|---:|---:|---:|
| TTFT | 52.4ms | 40.3ms | 116.7ms | 114.4ms |
| prefill | 670.62 tok/s | 873.38 tok/s | 300.96 tok/s | 306.96 tok/s |
| decode | 42.245 tok/s | 35.144 tok/s | 10.867 tok/s | 10.813 tok/s |
| host KV 저장소 | 56.0MiB | 112.0MiB | 28.875MiB | 28.875MiB |
| 프로세스 종료 VmRSS | 223.3MiB | 391.4MiB | 144.0MiB | 151.9MiB |
| PPL (4 window) | 19.903103* | 20.108306* | 20.463505* | 20.369400 |

`*` PPL은 변경하지 않은 동일 바이너리의 §12 측정값을 재사용했다. tiled PPL은
이번에 같은 WikiText 4 window, 총 4,092토큰으로 새로 측정했다. window별 PPL은
12.700146 / 24.147175 / 21.198048 / 26.481465다. 기존 최적화 대비 -0.460%,
int8 baseline 대비 +2.343%다. 4 window만으로 전반적인 품질 향상을 주장하지 않는다.

decode는 기존 대비 **-0.49%**로 사실상 동일하며 int8 baseline이 여전히
**3.91배 빠르다.** prefill +1.99%, TTFT -1.95% 역시 단회 측정의 관찰값이다.
평균 decode step의 QNN 호출 합계는 88.666 → 89.890ms, host prepare는
1.093 → 0.981ms, commit은 0.034 → 0.032ms다. 따라서 host 복사 시간을 줄이는
것만으로 남은 차이를 해결할 수 없다. 기본 경로는 기존 untiled 구현으로 유지한다.

메모리 결과는 구분해서 해석해야 한다.

- 최종 prompt/token × part 2/3/4 모두 전체 FP16 K/V 복원 텐서가 없고,
  최대 단일 FP16 codec 중간 텐서는 **512KiB**다. decode의 기존 1023토큰
  텐서 2,095,104 bytes(약 2MiB) 대비 약 75% 작다. 전체 중간 텐서 합계나
  NPU peak memory가 75% 감소했다는 뜻은 아니다.
- 컴파일러 context metadata의 decode `spillFillBufferSize`는 part 2/3/4에서
  2,555,904 / 2,686,976 / 2,424,832 bytes → **모두 0**으로 줄었다.
  prefill은 6,815,744 / 6,815,744 / 5,767,168 →
  5,308,416 / 5,308,416 / 4,325,376 bytes다. 이는 컴파일러가 보고한 buffer
  요구량이지 실측 DDR 트래픽이나 전체 HTP peak allocation이 아니다.
- 반대로 decode `opDataSize`는 part 2/3/4에서
  12,167,168 / 12,185,088 / 11,122,432 →
  19,411,968 / 19,501,568 / 17,041,920 bytes로 늘었다. 타일별 연산과
  스케줄 자료가 증가했고, host VmRSS도 약 7.9MiB 증가했다.

타일링은 전체 복원 텐서를 없애지만 **총 1023토큰의 unpack·centroid 복원·norm
보정·dense inverse rotation 연산량은 줄이지 않는다.** 또한 tile마다 Slice,
Concat, QK/AV, AV 누적이 추가된다. 이 결과는 그래프 tiling만으로 baseline과의
속도 차이를 줄이지 못했음을 보여준다. 다음 속도 개선 후보는 실제 packed KV를
소비하는 HTP custom fusion 또는 유효 cache 길이별 그래프 버킷이며, 이번에는
구현하거나 성능을 입증하지 않았다.

검증과 별도 detailed profile:

- EOS는 11번째 토큰에서 정상 정지했다. 897토큰 prompt + 128토큰 생성으로
  cache 1024/1024에 도달했고, 별도 reset 검증의 두 짧은 8토큰 세션은 token ID가
  완전히 같았다. 이 기능 검증들의 시간 수치는 위 성능 표에 포함하지 않았다.
- prefill과 decode 양쪽에서 28개 레이어의 tile 0/256/512/768 연산이 HTP
  profile에 기록됐다. decode part 2/3/4의 accelerator execute 시간은
  28.861 / 28.685 / 28.365ms다. 단일 custom fused kernel이 실행됐다는 뜻은 아니다.
- decode op cycle 합계에서 KV 복원 경로가 차지하는 비중은 part별
  77.2 / 77.2 / 73.8%로 여전히 크다. 여기서 복원 경로는
  `tq_{key,value}_<layer>_tile<start>_` 아래 `dec_*`, `restored*`, packed/norm
  Slice이며 encoder와 `tq_attn_*`는 제외했다. 이는 **op cycle 합계의 비중**이지
  벽시계 시간의 비중이 아니다. 기존 runner의 `codec_op_cycles_sum`은 `tq_*`를
  모두 세므로 새 `tq_attn_*`까지 포함한다. 이를 이전 codec 비중과 직접 비교하면
  안 된다. detailed profile 실행의 큰 계측 오버헤드도 성능 수치에 섞지 않았다.
- 관련 단위 테스트 **185개**와 변경 파일의 pre-commit 검사(mypy 포함)를 통과했다.

산출물은 `~/.qaihm/tmp/turboquant/` 아래에 보존했다.

- 최종 번들: `qwen3_1_7b_k4_v4_tiled256_int8_cl1024/`;
  device bundle: `k4_v4_tiled256_cl1024`. part 1은 기존 최적화 번들과 동일하며
  part 3/4의 별도 변환 디렉터리는 최종 번들에서 symlink로 참조한다.
- `reports/perf_{baseline_int8_tiled_control_once,baseline_int16_tiled_control_once,k4_v4_optimized_tiled_control_once,k4_v4_tiled256_once}.json`:
  구성당 1회의 원본 성능 결과.
- `reports/quality_k4_v4_tiled256.json`, `reports/score_k4_v4_tiled256_w{0,1,2,3}.json`:
  통합 PPL과 window별 NLL.
- `reports/boundary_k4_v4_tiled256.json`: 모든 tile의 dtype·경계·복원 크기 검사 통과.
- `reports/tiled256_int8_grid_comparison.json`: 6개 그래프의 KV Concat·QK·부분 AV
  10,752개 encoding 비교가 모두 기존 calibrated grid와 일치한다.
- `reports/invariants_k4_v4_tiled256.json`,
  `reports/generation_{eos,boundary,reset}_k4_v4_tiled256.json`: 기능 검증 결과.
- `reports/profile_k4_v4_tiled256.json`: 별도 prefill/decode HTP op profile.

## 14. 2026-09-17 유효 길이 버킷·회전 이동·scale 사전 계산

### 14.1 선택형 구현과 호환성

새 `k4_v4_scaled` profile, `--rotated-attention --attention-tile 256` 및
`--context-buckets 128 256 512 1024`를 함께 사용한다. 기존 `k4_v4`, format 1,
단일 CL1024 runner 동작은 유지한다. 명령은 도구 README에 있다.

- **유효 길이 버킷:** runner가 `cached <= C - AR`인 가장 작은 C를 고른다.
  AR1의 past 용량은 127/255/511/1023, AR128은 CL256/512/1024에서
  128/384/896이다. AR128/CL128은 past 용량이 0이므로 생성하지 않는다.
  따라서 선택한 버킷 내부의 padding은 여전히 계산한다. 정확히 n토큰만
  처리하는 동적 커널이나 모델의 context window 축소가 아니다.
- host cache는 1024토큰 용량과 절대 위치를 유지한다. graph I/O만 선택한
  길이로 줄이고 유효 KV를 오른쪽에 정렬한다. 버킷이 바뀔 때 cache row stride는
  계속 1024이며 RoPE 위치도 초기화하지 않는다. 단계별 `graph_context`를 기록한다.
- **회전 위치 변경:** 행벡터 표기로 복원값이 `K = K_r R_K`, `V = V_r R_V`일 때
  `Q K^T = (Q R_K^T) K_r^T`, `A V = (A V_r) R_V`를 이용한다. 과거 KV 벡터의
  역회전을 제거하고 query와 누적 attention output만 변환한다. 현재 K/V도
  attention용으로 회전하며, 새 토큰의 cache encoder 회전은 별도로 남는다.
- 전역 mask/softmax와 past→current 순서는 유지한다. 기존 int8 grid는 회전된
  좌표에 유효하지 않으므로 새 QK/부분 AV는 **FP16**이다. 원래 좌표계의 score,
  softmax, 최종 attention output encoding을 유지하지만 실수 대수 동등성이
  기존 양자화 바이너리와의 bit-exact 동등성을 뜻하지 않는다.
- **scale 사전 계산:** `s = ||x|| / ||c[index]||`를 encoder에서 한 번 계산하고
  FP16으로 저장한다. norm correction이 꺼져 있으면 `s = ||x||`다. decoder는
  centroid 조회 후 s만 곱하며 norm의 제곱합·제곱근·나눗셈을 반복하지 않는다.
  모든 유효 토큰의 nibble unpack과 centroid 조회 자체는 여전히 필요하다.

format **2**는 packed byte 순서와 크기를 유지하지만 scalar 의미가 다르다.
I/O를 `tq_*_scale_{in,out}`으로 구분하고 config hash·cache snapshot version도
분리했다. format 1 norm을 format 2 scale로 읽으면 틀린 결과가 되므로 자동
혼용하지 않는다. 각 KV 벡터당 FP16 scalar 한 개여서 host KV 저장소 크기는 같다.
FP16에 들어가야 하는 값은 원래 norm이 아니라 **effective scale**이다. norm만
65504 이하라고 scale까지 표현 가능하다고 보장하지 않는다. host 저장은 범위를
검사하며, device encoder를 다른 모델에 적용할 때는 그 모델의 scale 범위도
검증해야 한다.

버킷 그래프는 part별 하나의 weight-shared context에 넣지만, graph metadata와
각 길이별 resident I/O buffer는 늘어난다. 저장소 절감과 프로세스 전체 메모리를
구분해서 보고한다. 이것도 그래프 수준 변환이며 native packed-attention fusion은
아니다. 성능 목표는 **int16 KV baseline**이고, 성능 측정은 구성당 1회다.

### 14.2 검증

- 빈/부분/전체 cache, GQA, causal mask, 마지막 짧은 tile, 한 tile짜리 작은
  버킷에서 float32 ONNX 결과를 기존 복원 방식 및 format-2 전체 복원과 비교한다.
  FP16/정수 실행의 품질을 CPU 동등성만으로 보증하지 않는다.
- cache format 혼용 거부, append/reset/snapshot, padding 길이 변경과 C++ runner의
  정확한 경계 선택(127→128, 255→256, 511→512)을 검사한다.
- 기존 경로의 회귀 검사를 포함한 단위 테스트 219개가 통과했다.
- `verify_rotated_attention.py`는 최종 DLC의 UINT8 packed / unencoded FP16 scale,
  16-bit KV 쓰기 경로, FP16 attention, decoder 역회전·norm reduction 제거,
  tile 크기를 검사한다. 빈/미완성 DLC 표는 실패로 처리한다.
- 새 scale codec의 실제 HTP 검증은 `p2_scaled_20260917/p2_report.json`에 있다.
  K/V × 1/128토큰 encode/decode 8개 그래프를 실제 layer 0/27 KV와 제한된
  범위 입력으로 검증했다. format-2 scale 범위 때문에 range 입력은 기존
  probe의 0.5배(원래 norm 최대 약 32500)이며 full-FP16-domain 검증은 아니다.
- 이 probe에서 **128토큰 K/V encoder의 scale 오차는 기존 norm-only 0.2%
  기준을 넘었다**(최대 약 0.2564%). 따라서 기존 기준의 전체 판정은 실패이며
  통과로 재표기하지 않았다. index 차이는 모두 허용한 인접 경계 안에 있고,
  decoder와 detailed HTP 실행은 통과했다. scale에 norm 보정이 합쳐지면서
  추가 FP16 반올림이 생긴다. 이 수치 한계를 보존하고 전체 모델 PPL을 별도로
  평가한다. 기존 norm/decoder tolerance는 완화하지 않는다.
- `htp_codec_validation.py build/all --profile k4_v4_scaled --rotation fwht --range-scale 0.5`로
  같은 종류의 bounded probe를 만들 수 있다. `compare`는 저장된 manifest의
  profile과 config hash를 사용하므로 format-2 결과를 format-1 oracle로 잘못
  해석하지 않는다. 기존 기기 출력을 재비교해도 위 실패가 그대로 재현됐다.

### 14.3 실기기 성능: 짧은 문맥은 개선, 전체 문맥 목표는 미달

S26/SM8850, 동일 가중치·runner·burst 설정에서 **구성·입력 조건당 1회** 측정했다.
아래 표는 35토큰 prompt와 128토큰 greedy 생성이다. profiling은 끄고 모델
로딩은 TTFT에서 제외했다. 별도 warmup 제외, 온도 통제, 실행 순서 교차는 없으며
단회 관찰값이다. 성능 측정을 재시도하거나 기존 결과와 평균하지 않았다.

| 지표 | int16 KV baseline | 기존 tiled-256 `k4_v4` | 새 구현, CL1024 고정 | 새 구현, 자동 버킷 |
|---|---:|---:|---:|---:|
| TTFT | 44.5ms | 118.3ms | 104.8ms | 62.2ms |
| prefill | 792.47 tok/s | 296.61 tok/s | 335.16 tok/s | 567.19 tok/s |
| decode | 32.239 tok/s | 10.507 tok/s | 14.650 tok/s | **44.600 tok/s** |
| host KV 저장소 | 112.0MiB | 28.875MiB | 28.875MiB | 28.875MiB |
| 프로세스 종료 VmRSS | 392.0MiB | 152.2MiB | 153.9MiB | 296.1MiB |
| PPL (4 window) | 20.108306* | 20.369400* | 20.388335 | 20.388335 |

`*` PPL만 변경하지 않은 동일 바이너리의 기존 측정값을 재사용했다. 새 구현의
두 구성은 각각 같은 WikiText 4 window, 총 4,092토큰으로 평가했다. window별
PPL은 두 구성 모두 12.743880 / 23.649163 / 21.229252 / 27.006912다.
int16 baseline 대비 +1.393%, 기존 tiled 대비 +0.093%다. 이 네 window에서
PPL/NLL이 같았다는 결과가 모든 입력이나 decode의 bit-exact 동등성을 보장하지는
않는다. §14.2의 standalone scale encoder 기준 실패도 그대로 남아 있다.

- CL1024 고정 결과는 기존 tiled 대비 **39.4%** 빨라졌다. 회전 이동과 scale
  사전 계산, 회전 좌표의 FP16 attention을 함께 적용한 결과이며 각 변경의
  독립적인 기여도를 분리한 실험은 아니다.
- 자동 버킷은 기존 tiled 대비 **4.24배**, 이번 CL1024 고정 int16 baseline 대비
  **38.3%** 높은 decode 처리량을 보였다. 127개 decode step 중 C128이 93개,
  C256이 34개다. 평균 past 계산 용량은 1023 → **161.27토큰**, 약 **84.24% 감소**다.
  유효 KV 자체를 버린 것이 아니라 큰 고정 그래프의 불필요한 padding을 줄였다.
- **int16 baseline에도 같은 버킷 최적화를 적용한 비교는 아니다.** 따라서
  이 결과는 현재 배포 구성끼리의 비교이지, 같은 attention 길이에서 4-bit 연산이
  int16보다 본질적으로 빠르다는 근거가 아니다. prefill/TTFT는 여전히 baseline이
  더 빠르다.
- 평균 decode step의 host prepare / QNN 호출 합계 / commit은 기존 tiled에서
  1.308 / 91.233 / 0.054ms, 새 고정 CL1024에서 0.964 / 65.400 / 0.032ms,
  자동 버킷에서 0.153 / 21.665 / 0.009ms다. 작은 버킷이 NPU 계산과 host I/O
  양쪽의 작업량을 줄였다. 이 합계에는 token 선택 등 모든 generation 비용이
  들어 있지는 않다.

별도 긴 입력 조건도 각각 **1회** 실행했다. 897토큰 prompt + 128토큰 생성으로
캐시가 정확히 1024에 도달하며, 모든 decode step이 C1024를 사용한다. 새 구현의
context 경계 검증 실행을 이 조건의 측정으로 함께 사용했고 다시 실행하지 않았다.

| 지표 | int16 KV baseline | 새 구현, 자동 버킷 |
|---|---:|---:|
| TTFT | 319.3ms | 722.1ms |
| prefill | 2811.58 tok/s | 1243.94 tok/s |
| decode | **34.912 tok/s** | **13.884 tok/s** |

**전체 문맥에서는 int16 baseline이 여전히 2.51배 빠르다.** 버킷으로 줄일 padding이
거의 없어지면 짧은 입력의 이점도 사라진다. 따라서 현재 구현은 짧은 문맥의
decode 목표를 달성했지만, 전체 1023토큰 처리 조건의 목표는 달성하지 못했다.

### 14.4 메모리·기능·남은 병목

- 최종 attention DLC 21개 모두 UINT8 packed / FP16 scale I/O, 16-bit KV 쓰기,
  FP16 회전 attention, 과거 KV 역회전·norm reduction 제거 검사를 통과했다.
  최대 단일 FP16 codec 중간 텐서는 여전히 **512KiB**다. 전체 원래 좌표의 FP16
  KV 텐서는 없지만 packed unpack과 centroid 복원은 각 선택 버킷의 past 용량만큼
  남는다. custom fused HTP kernel이 추가된 것은 아니다.
- host KV 저장소는 int16 대비 **74.22% 작다**. 반면 자동 버킷은 모든 길이의
  graph I/O를 resident로 유지하므로 I/O buffer가 96.93 → **222.20MiB**, 종료
  VmRSS가 153.9 → **296.1MiB**로 늘었다. KV 저장소 절감률을 프로세스 전체나
  peak HTP 메모리 절감률로 해석하면 안 된다.
- EOS는 11번째 생성 토큰에서 정지했다. 897토큰 prompt 경계, 두 세션 reset의
  동일 token ID, 600토큰 생성 중 128→256→512→1024 버킷 전환 및 매 step의
  최소 수용 길이 선택이 모두 통과했다. 이 기능 검증과 detailed profile의 시간
  수치는 위 짧은 조건 성능 표에 섞지 않았다.
- 별도 decode detailed profile에서 CL1024 고정 part 2/3/4의 accelerator execute
  시간은 19.879 / 19.748 / 20.810ms다. KV tile 복원 경로의 **op cycle 합계 비중**은
  84.5 / 85.4 / 80.2%다. decoder 역회전과 norm 보정은 제거됐지만 unpack,
  centroid 선택, scale 곱, layout 처리의 비용이 남는다. 이 비중은 벽시계 시간의
  비중이 아니며 이전 profile과의 절대 cycle 비교도 동일한 계측 조건 안에서
  제한적으로 해석해야 한다.
- C128 decode profile에서는 해당 비중이 56.6 / 56.7 / 41.9%다. 긴 문맥에서
  목표 속도에 더 접근하려면 남은 nibble unpack·centroid 선택을 packed KV를
  직접 소비하는 attention 연산과 통합하는 native HTP 최적화가 다음 후보다.
  이번 결과만으로 그 구현의 속도나 baseline 동등성을 보장하지 않는다.
- 단위 테스트 219개와 변경 파일 전체의 pre-commit 검사(mypy 포함)가 통과했다.
  다만 standalone scalar 오차 기준은 실패이므로 새 profile은 선택형으로 유지하고
  기존 기본 profile과 cache format은 바꾸지 않았다.

### 14.5 산출물

`~/.qaihm/tmp/turboquant/` 아래에 보존했다.

- 최종 번들 `qwen3_1_7b_rotated_scaled_buckets/`, device bundle
  `rotated_scaled_buckets`: part별 weight-shared context에 총 28개 그래프.
  part 1/3/4는 최종 번들에서 별도 변환 디렉터리를 symlink로 참조한다.
- `reports/comparison_rotated_scaled_buckets.json`: 비교 수치, 원본 경로,
  PPL 재사용 여부, profile·기능·수치 기준 결과를 모은 요약.
- `reports/perf_{baseline_int16_rotated_control,k4_v4_tiled_rotated_control,rotated_scaled_fixed,rotated_scaled_buckets}_once.json`:
  짧은 입력의 구성당 1회 성능 원본.
- `reports/perf_baseline_int16_rotated_long_once.json`,
  `reports/generation_boundary_rotated_scaled_buckets.json`: 긴 입력 각 1회.
- `reports/quality_rotated_scaled_{fixed,buckets}.json`,
  `reports/score_rotated_scaled_{fixed,buckets}_w{0,1,2,3}.json`: PPL 원본.
- `reports/boundary_rotated_scaled_buckets.json`,
  `reports/invariants_rotated_scaled_buckets.json`: 최종 DLC 및 기능 검사.
- `reports/profile_rotated_scaled_{fixed,buckets}.json`: 별도 detailed profile.
- `p2_scaled_20260917/p2_report.json`: 기존 tolerance를 그대로 적용한 standalone
  HTP codec 수치 결과. 전체 판정은 실패이며 성능 개선과 별개로 확인해야 한다.

## 15. Native unpack + LUT decoder

### 15.1 변경 범위와 재현 경로

`convert_parts.py --native-decoder-package PATH`로 선택하는 HTP V81 QHPI
패키지 `TurboQuantNative::Decode4`를 추가했다. 기존 graph decoder와 배포
바이너리는 보존하며 기본 실행 경로도 바꾸지 않는다. 지원 범위는 D128,
K4/V4, FP16 effective scale을 사용하는 format-2 rotated/tiled attention이다.

- UINT8 packed tile의 상위 nibble부터 인덱스를 추출하고, HVX `vlut16`으로
  16-entry FP16 centroid를 조회한 뒤 FP16 scale을 곱한다. 기존 그래프의
  unpack·centroid 선택·scale 곱을 하나의 native decoder 연산으로 대체한다.
- 두 행씩 벡터 처리하며 홀수 마지막 행은 유효한 64byte만 읽는다. 입력과
  출력의 정렬을 가정하지 않고, QHPI 실행 slice별로 겹치지 않는 행을 처리한다.
- encoder, packed cache ABI, query/output 회전, QK·global softmax·AV와
  calibration은 유지한다. **복원과 attention을 하나의 커널로 합친 것은 아니다.**
  FP16 tile 중간 결과는 여전히 존재하며 과거 KV를 처리하는 점근적 연산량도
  그대로다. 같은 작업을 훨씬 적은 벡터 명령과 graph op로 수행하는 변경이다.
- x86 library는 offline context 준비에, Hexagon library는 실제 HTP 실행에
  사용한다. runner는 context를 읽기 전에 package를 등록한다. 배포 manifest의
  DSP library SHA256을 push와 run 양쪽에서 검사한다.
- QAIRT 2.48, Hexagon SDK 6.6 및 V81 compiler가 필요하다. SDK 소스나
  라이브러리는 저장소에 포함하지 않았다. 빌드·기기 검증·변환·배포 명령은
  `scripts/llm/turboquant/README.md`의 Native 절에 있다.

### 15.2 수치 및 구조 검증

- standalone HTP decoder를 KV 길이 1/3/127/128/255/256/1023에서 실행했다.
  256종 packed byte, 난수, 0 및 작은/큰 FP16 scale을 포함한 실제 기기 결과는
  `FP16(FP16(centroid) * FP16(scale))` oracle과 **100% bit-exact**였다.
  이것은 테스트한 유한 입력의 결과이지 모든 실수 입력에 대한 오차 보장이 아니다.
- 동일 HVX 코드의 host libnative 검사는 정렬되지 않은 I/O, 홀수 tail 및
  canary 영역을 포함해 통과했다. ONNX 테스트에는 독립적인 정수 unpack/LUT
  reference function을 붙이지만, 배포 모델에는 붙이지 않아 converter가
  native op를 다시 일반 그래프로 펼치지 않도록 했다.
- attention DLC 21개가 raw UINT8 packed / FP16 scale I/O, native decoder의
  FP16 출력, 기존 decoder 노드 제거, FP16 회전 attention 및 16-bit KV 쓰기
  검사를 모두 통과했다. 최대 단일 FP16 복원 tile은 **512KiB**로 유지된다.
- format-2 encoder는 수정하지 않았다. §14.2의 **기존 encoder scale 기준
  실패(0.2564% > 0.2%)는 해결하지 않았으며**, native decoder 통과와 구분한다.
  centroid와 곱셈의 FP16 반올림을 포함하므로 end-to-end PPL도 별도 평가한다.

### 15.3 실기기 성능: 긴 문맥 decode 목표 도달

S26/SM8850, Qwen3-1.7B W4A16, CL1024, 같은 runner·가중치·burst 설정에서
구성 및 입력 조건당 **1회** 측정했다. baseline 두 바이너리의 기기 SHA256은
기존 배포 manifest와 일치한다. profiling은 끄고 모델 로딩은 TTFT에서 제외했다.
별도 warmup 제외, 온도 통제, 실행 순서 교차나 분산 추정은 하지 않았다.
성능 실행은 재시도하거나 과거 측정과 평균하지 않았다.

주요 조건은 **897토큰 prompt + 128토큰 생성**이다. 마지막 생성 토큰을
제외한 127개의 decode 입력까지 KV에 저장하여 캐시가 정확히 1024에 도달하며,
세 구성 모두 127개 decode step 전체가 C1024를 사용한다.

| 지표 | int16 KV baseline | 기존 TQ-Graph | TQ-Native LUT |
|---|---:|---:|---:|
| TTFT | 318.2ms | 718.4ms | 567.2ms |
| prefill | 2820.63 tok/s | 1249.20 tok/s | 1584.24 tok/s |
| decode | 34.756 tok/s | 13.893 tok/s | **38.150 tok/s** |
| decode 시간/token | 28.772ms | 71.979ms | **26.212ms** |
| host KV 저장소 | 112.000MiB | 28.875MiB | 28.875MiB |
| resident graph I/O buffer | 263.179MiB | 222.197MiB | 222.197MiB |
| 종료 VmRSS | 391.676MiB | 295.582MiB | 291.953MiB |
| 프로세스 VmHWM | 605.016MiB | 605.352MiB | 605.289MiB |
| PPL, 4 window / 4092토큰 | 20.108306* | 20.388335* | 20.498138 |

`*` PPL만 변경하지 않은 기존 바이너리의 측정값을 재사용했다. 성능은 세 구성
모두 이번에 새로 1회씩 측정했다. Native PPL은 4개 window에서 각각
12.654283 / 23.697976 / 21.626364 / 27.222334이며, 합산 NLL로 PPL을 계산했다.
기존 graph 대비 **+0.539%**, int16 대비 **+1.939%**다. standalone FP16 LUT
oracle과 bit-exact인 것은 기존 compiled graph 및 전체 모델과 bit-exact임을
뜻하지 않는다. 명시적인 FP16 LUT 연산 및 compiler 실행 계획 변경의 수치
영향을 분리한 ablation은 하지 않았다. 이 네 window로 일반적인 품질 동등성을
주장하지 않는다.

- 긴 문맥 decode는 기존 graph 대비 **2.746배**, int16 대비 **9.77%** 높은
  처리량이다. 이번 배포 구성의 단회 측정에서는 int16 목표에 도달했지만,
  반복 측정 없는 결과이므로 안정적인 우위나 다른 길이·기기의 우위를 보장하지 않는다.
- TTFT와 prefill은 기존 graph보다 좋아졌으나 **int16보다 여전히 느리다**.
- host KV는 int16 대비 74.22% 작고 기존 graph와 동일하다. 종료 VmRSS는
  줄었지만 **프로세스 peak VmHWM은 사실상 그대로**다. 이 수치들은 NPU 전체
  메모리나 peak HTP 메모리 절감률을 나타내지 않는다.
- 35토큰 prompt + 128토큰 생성의 별도 단회 결과는 아래와 같다. TQ 두 구성은
  C128 93step / C256 34step, int16은 C1024 고정이다. **버킷화한 int16과의
  비교가 아니므로**, 짧은 문맥의 차이를 decoder kernel만의 효과로 해석하지 않는다.

| 짧은 입력 지표 | int16 KV baseline | 기존 TQ-Graph | TQ-Native LUT |
|---|---:|---:|---:|
| TTFT | 42.0ms | 60.9ms | 58.9ms |
| prefill | 837.29 tok/s | 578.62 tok/s | 597.78 tok/s |
| decode | 35.049 tok/s | 44.327 tok/s | **54.427 tok/s** |

### 15.4 개선 원인과 남은 비용

긴 입력의 비계측 성능 실행에서 평균 decode 단계의 host prepare / QNN 호출
합계 / commit은 int16 **5.034 / 22.892 / 0.273ms**, 기존 graph
**2.901 / 66.410 / 0.055ms**, Native **1.136 / 24.475 / 0.014ms**다.
따라서 Native가 int16보다 빠른 **end-to-end** 결과에는 작은 packed cache의
host I/O 이점도 포함된다. QNN 호출 시간만 보면 아직 int16보다 약 1.58ms
느리며, native attention 계산 자체가 int16보다 빠르다고 주장하지 않는다.

별도 C1024 detailed profile에서는 다음을 확인했다. 이 진단 실행의 느려진
TTFT/decode 시간은 위 성능 수치에 포함하지 않았다.

| part | graph accelerator execute | Native accelerator execute | graph 복원 op cycle 비중 | Native 복원 op cycle 비중 |
|---|---:|---:|---:|---:|
| 2 | 19.851ms | 7.002ms | 84.5% | 46.5% |
| 3 | 20.553ms | 7.105ms | 84.4% | 46.6% |
| 4 | 20.475ms | 10.095ms | 80.5% | 39.1% |

Native decoder 실행 이벤트는 각각 80/80/64개로, 28개 레이어 × K/V × 4개
tile에 해당한다. 복원 경로의 op cycle 합계는 545,751,309 → 82,453,336으로
약 **84.9% 감소**했다. 단, op cycle의 합계·비중은 벽시계 시간의 비중이 아니다.
복원 경로에는 decoder 외에 tile slice/layout 등도 포함하며, Native Decode4
자체의 cycle 합계는 51,106,047이다. 복원과 attention 사이의 FP16 tile,
layout 전환, QK/AV 연산 및 host I/O는 여전히 남아 있다.

EOS 11번째 토큰 정지, 2회 세션 reset의 동일 token ID, 600토큰 생성 중 최소
버킷 선택과 128→256→512→1024 전환, 긴 성능 실행의 1024 캐시 경계가 모두
통과했다. 별도 기능 실행의 시간은 성능 표에서 제외했다. TurboQuant 단위
테스트 **238개** 및 변경 파일 pre-commit 검사(mypy 포함)도 통과했다.

### 15.5 산출물

`~/.qaihm/tmp/turboquant/` 아래에 원본 및 이번 결과를 함께 보존했다.

- `qwen3_1_7b_native_lut_buckets/`, device bundle `native_lut_buckets`:
  non-KV part 1만 기존 번들에서 재사용하고 attention part 2/3/4를 교체했다.
  각 part의 7개 그래프가 weight-shared context에 묶여 있다.
- `native_decoder_hvx_20260918/`: 실제 사용한 prepare/DSP package와 manifest.
  DSP library SHA256은
  `a7c004fa9f824b5646913c565d88a95c0f6440eb35ee6f0c92c575aa21b68b3c`다.
- `native_hvx_validation_20260918/correctness.json`: standalone 기기 수치 검증.
- `reports/comparison_native_lut_buckets.json`: 조건·성능·PPL·profile·기능 및
  수치 기준의 통합 요약. `summarize_native_results.py`로 원본에서 재생성한다.
- `reports/perf_{baseline_int16_native_control,graph_native_control,native_lut}_{short,long}_once.json`:
  총 6개 단회 성능 원본. `profile_*_long.json`과 분리했다.
- `reports/quality_native_lut_buckets.json`, `reports/score_native_lut_w{0,1,2,3}.json`:
  PPL 합산 및 window별 원본.
- `reports/boundary_native_lut_buckets.json`,
  `reports/generation_native_lut_{reset,eos,switches}.json`: 구조·기능 검사.
- 기존 graph와 int16 번들은 수정하지 않았으며, `--native-decoder-package`를
  지정하지 않으면 기존 graph decoder가 사용된다. §14.2의 encoder 수치 한계가
  남아 있으므로 새 경로는 opt-in으로 유지한다.

## 16. Dense QR 기본값과 Native 경로 비교

### 16.1 변경 범위

2026-09-18부터 별도 지정이 없으면 `TurboQuantConfig`, `get_profile(...)`,
`PolarQuantReference` 및 변환/host 평가 도구는 **dense QR**을 사용한다.
WHT는 `--rotation fwht` 또는 `get_profile(name, Rotation.FWHT)`로 명시적으로
선택한다. 기본 export에 TurboQuant를 강제로 활성화한 것은 아니며,
`baseline_int16_kv`와 `baseline_int8`의 동작·기존 config hash는 유지한다.

후속 기본값 설정: 최신 `k4_v4_scaled` 변환은 Native decoder, rotated
attention, tile 256도 옵션 생략 시 자동 선택한다. graph decoder는
`--no-native-decoder`로 명시적으로 선택한다. Native package가 없으면 오류로
중단하며 자동 빌드나 fallback은 하지 않는다. baseline·legacy raw-norm
프로필과 context bucket 기본값은 변경하지 않는다. 이 기본값 설정 후에는
추가 테스트·빌드·기기 실행·측정을 하지 않았으며, 아래 결과는 앞선 실측이다.

이번 비교의 알고리즘 변경은 회전 행렬뿐이다. K/V 각 4-bit, QJL off,
norm correction on, format-2 FP16 effective scale, Native unpack/LUT,
tile 256, rotated attention 및 C128/256/512/1024 버킷을 유지한다.
QJL을 추가하거나 K를 3-bit로 줄이지 않았다.

- K seed 42, V seed 542로 Gaussian 행렬을 생성하고 QR의 column sign과
  determinant를 보정한다. host에서 한 번 계산·캐시하고 상수로 export한다.
  NPU의 토큰 처리 경로에는 QR/RNG가 없다.
- 종전 WHT도 export 시 `D2 H D1`을 **dense MatMul**로 내렸으므로,
  이번 변경은 NPU butterfly 구현을 dense 연산으로 교체한 것이 아니다.
  Native decoder는 회전과 무관하게 index→centroid→scale만 수행하므로
  이전 DSP library를 변경 없이 재사용한다.
- LAPACK 차이를 추적하도록 실제 export float32 `R`의 SHA256을 config에
  기록한다. K는 `7e76eadbc9e5aec69be63829c8068ca5618b203ee09d3eaf4918ee6be9e0d016`,
  V는 `a89dbff3a12d727d4b9decbc82e1e81490165fde94a1efc12f649bf11b054309`이다.
- dense `k4_v4_scaled` config hash:
  `a1bd2907c7f1352c7e3472d7022dcb4c11cf6cf85b361f5316a0d155697c3c62`.
  이전 WHT config hash `94bf3075...`와 구분한다. packed ABI 크기는 같아도
  회전이 다른 cache state는 호환되지 않으며 host state 로드는 이를 거부한다.
- 변환 도구는 기존 출력 폴더의 rotation/config hash 또는 컴파일 설정이
  다르면 중단한다. 기존 바이너리가 새 Python 기본값으로 자동 변경되지는 않는다.

### 16.2 수치 검사

- 단위 테스트 **268개 통과**: dense 기본값, 기존 WHT golden 유지, 두 회전의
  Native attention oracle, cache 교차 로드 거부, rotation 상수 변조 검출,
  기존 번들 덮어쓰기 방지 및 분할 빌드 조립 검사를 포함한다.
- `verify_reference_dense_default.json`: 고정 참조 commit 대비 K/V dense QR
  행렬 최대 절대오차 0, 3/4-bit index 불일치 0, 복원 상대오차 0.
  이것은 float64 CPU 참조 대조이며 NPU bit-exact 주장과는 다르다.
- 28개 그래프/4개 context binary 빌드 완료. attention을 포함하는 **21개
  그래프 전부** rotation 상수·FP16 QK/AV·Native Decode4·scale I/O 검사를
  통과했다. Native decoder는 총 840개, 최대 FP16 KV tile은 524,288byte다.
  각 source graph는 회전 상수와 그 이름을 제외하면 기존 WHT+Native와
  직렬화 결과가 동일하다. KV가 없는 part1 binary도 이전과 SHA256이 같다.
  tile 크기는 tensor 단위 상한이며 전체 NPU peak memory 측정값은 아니다.
- `p2_dense_scaled_20260918/p2_report.json`: 실제 HTP에서 K/V encode의
  T1/T128, head-major, 실제 KV 두 layer와 합성 range 입력을 검사했다.
  4개 그래프 모두 accelerator 프로파일에서 누락 op이 없고 설명되지 않는
  index 불일치는 0이다. 하지만 **T128 effective-scale 허용오차 gate는 실패**다.
  최대 상대오차 0.275200%(실제 KV만 최대 0.247664%)로 고정 기준 0.2%를 넘는다.
  기존 WHT의 scale gate 실패(0.256412%, §14.2)와 마찬가지로 encoder의
  FP16 수치 한계가 남아 있다. 기준을 완화하거나 Native decoder 성공으로
  encoder 검사를 통과 처리하지 않았다. 합성 입력은 `--range-scale 0.5`의
  제한된 범위이며 전체 FP16 입력 범위 검증은 아니다.

### 16.3 3개 구성 실기기 비교 (각 조건 1회)

Qwen3-1.7B W4A16, S26/SM8850, capacity 1024, burst power,
profiling off, 생성 128개(순수 decode step 127개)로 측정했다.
각 구성마다 35-token 입력과 897-token 입력을 **각각 1회** 실행했다.
반복 평균이나 좋은 실행 선택은 없고, 이전 성능 값을 재사용하지 않았다.
TTFT는 모델 로딩을 제외한다. 성능 실행 순서는 각 입력 조건에서
int16 → WHT+Native → dense+Native이며, 온도 통제나 분산 추정은 없다.
PPL도 세 구성 모두 같은 WikiText 4×1024 window를 새로 채점했다(4,092 tokens).
리포트에 실제 선택한 입력 파일과 SHA256을 기록해 동일 입력 여부를 검증한다.

긴 입력(897 prompt + 128 generation): 모든 decode는 **C1024**를 사용했다.

| 지표 | int16 KV | 최신 WHT+Native | dense+Native (새 기본 회전) |
|---|---:|---:|---:|
| TTFT | 303.25 ms | 522.93 ms | 563.26 ms |
| prefill | 2,959.76 tok/s | 1,716.59 tok/s | 1,595.34 tok/s |
| decode | 35.6603 tok/s | 39.1088 tok/s | 38.0792 tok/s |
| decode/token | 28.0424 ms | 25.5697 ms | 26.2610 ms |
| host KV 저장소 | 112.000 MiB | 28.875 MiB | 28.875 MiB |
| resident graph I/O buffers | 263.179 MiB | 222.197 MiB | 222.197 MiB |
| 종료 VmRSS | 391.461 MiB | 292.020 MiB | 292.352 MiB |
| 프로세스 VmHWM | 605.105 MiB | 605.563 MiB | 605.492 MiB |
| PPL (4 window, 낮을수록 좋음) | 20.108306 | 20.498138 | 20.607753 |

짧은 입력(35 prompt + 128 generation):

| 지표 | int16 KV | WHT+Native | dense+Native |
|---|---:|---:|---:|
| TTFT | 43.686 ms | 59.208 ms | 59.080 ms |
| prefill | 805.42 tok/s | 594.97 tok/s | 596.20 tok/s |
| decode | 34.9665 tok/s | 54.4383 tok/s | 54.3845 tok/s |
| 종료 VmRSS | 388.934 MiB | 292.219 MiB | 292.359 MiB |

짧은 입력의 int16은 고정 C1024, 두 TurboQuant는 context bucket을 사용한다.
따라서 짧은 입력에서 int16 대비 55.5% 빠르다는 결과를 회전 알고리즘만의
속도 차이로 해석하면 안 된다. 긴 입력에서도 prefill의 버킷 사용은 다르다.

해석:

- Dense의 긴 문맥 end-to-end decode는 int16 대비 **6.78% 빠르고**,
  기존 WHT+Native 대비 **2.63% 느린** 관측값이다. 짧은 문맥의 두 Native
  경로는 54.44/54.38 tok/s로 비슷하다. 1회 측정으로 유의한 차이나
  dense 회전 자체의 실행 비용을 확정하지 않는다. 두 회전 모두 MatMul 경로다.
- 긴 문맥 평균 decode QNN 호출은 int16/WHT/dense 각각
  **22.710 / 24.259 / 24.508 ms**, host prepare는
  **4.739 / 0.919 / 1.146 ms**다. Dense의 end-to-end 이득에도
  압축된 host KV I/O가 기여하며, NPU 호출 자체가 int16보다 빠르다는 뜻은 아니다.
- Dense는 host KV를 int16 대비 **74.22% 줄이지만**, TTFT/prefill은
  여전히 int16보다 느리다. 종료 RSS는 낮아도 프로세스 최고 RSS(VmHWM)는
  약 605 MiB로 비슷하다. 이 값들은 NPU peak memory 계측이 아니다.
- Dense PPL은 WHT보다 **0.53%**, int16보다 **2.48%** 높다.
  이번 회전 변경이 품질을 개선했다고 주장하지 않는다. QJL은 적용하지 않았다.
- Reset(2회, 8-token 출력 동일), EOS(11-token 종료), 600-token 생성 중
  C128→256→512→1024 전환, 긴 성능 실행의 cache length 1024 경계 검사를
  통과했다. 이 진단 실행들의 timing은 위 성능 수치에 포함하지 않았다.

### 16.4 산출물과 재현

기준 디렉터리: `~/.qaihm/tmp/turboquant/`.

- 최종 번들: `qwen3_1_7b_dense_native_buckets_final/`, 기기 이름 `dense_native_buckets`.
  `qwen3_1_7b_dense_native_buckets/`는 part1+2 빌드 디렉터리이므로 최종 번들이 아니다.
- 기존 비교군은 `qwen3_1_7b_baseline_int16_kv_cl1024/`와
  `qwen3_1_7b_native_lut_buckets/`의 변경 없는 context binaries다.
- 종합 결과: `reports/comparison_dense_native_buckets.json`.
- 성능 원본: `reports/perf_{baseline_int16_dense_control,fwht_native_dense_control,dense_native}_{short,long}_once.json`.
- 품질 원본: `reports/score_{baseline_int16_dense_control,fwht_native_dense_control,dense_native}_w{0,1,2,3}.json`.
- 구조/기능: `reports/boundary_dense_native.json`,
  `reports/generation_dense_native_{reset,eos,switches}.json`.
- 변경 없는 DSP library:
  `native_decoder_hvx_20260918/hexagon-v81/libTurboQuantNative.so`, SHA256
  `a7c004fa9f824b5646913c565d88a95c0f6440eb35ee6f0c92c575aa21b68b3c`.

새 모델 변환 명령은 tools README의 Dense 절을 사용한다(회전 옵션 생략).
성능 실행을 다시 하지 않고 기존 원본을 집계하려면:

```bash
PYTHONPATH=src python scripts/llm/turboquant/summarize_rotation_results.py \
    --reports ~/.qaihm/tmp/turboquant/reports \
    --out ~/.qaihm/tmp/turboquant/reports/comparison_dense_native_buckets.json
```

## 17. Orthogonal K-only QJL: K3+1 / V4

### 17.1 알고리즘과 저장 형식

`k3qjl_v4_scaled`는 별도의 **format 3** 실험 profile이다. 기존
`k4_v4_scaled`의 Dense+Native 기본값과 설정 해시는 바꾸지 않는다.
사용자가 선택한 총 비트 예산은 K 3-bit MSE + QJL 1-bit, V 4-bit MSE다.
norm/scale 메타데이터는 이 4-bit 예산에 포함하지 않는다.
따라서 K4 MSE와의 비교는 동일 payload 예산의 알고리즘 비교이며,
같은 K4 MSE에 QJL만 더한 ablation은 아니다. QJL 추가와 MSE 비트 축소의
영향이 함께 포함되므로 성능/품질 변화를 QJL 한 가지 효과로 단정하지 않는다.

참조는 `turboquant_plus/turboquant/qjl.py`, checkout
`7f601a135735842a7f12b6bf861561154c410ff4`이며 파일 SHA256은
`2a9fcda9de3de4c2edcf7272369c513904d15336d8bd337a1ae90de0b730ee2c`다.
Gaussian 행렬의 QR 분해 후 R 대각 부호로 Q의 열 부호를 보정한다.
PolarQuant 회전과 달리 **determinant +1 보정은 하지 않는다**.
QJL seed는 K seed + 1000 = 1042다.

새 K를 저장할 때:

1. Dense MSE 회전 R로 3-bit 인덱스와 FP16 effective scale을 계산한다.
2. Native Decode4로 실제 저장값을 복원하여 `r = K - K_mse`를 계산한다.
3. 직교행렬 S로 `sign(S r)`를 구한다. 정확히 0이면 +1로 처리한다.
4. `a = sqrt(pi/2) / sqrt(d) * ||r||`를 FP16으로 저장한다.

복원은 `K_hat = K_mse + a * S.T @ signs`다. 요청한 고전적 계수를
그대로 사용하며 `2/pi` shrinkage는 적용하지 않는다. 기준 코드와의
알고리즘 일치를 목표로 하되, 실제 기기에서는 저장 scale, centroid,
곱셈, 회전의 FP16 반올림이 추가된다. 논문의 모든 이론 조건/커널을
그대로 재현했다거나 이 유한 차원 구현이 정확히 불편향이라고 주장하지 않는다.

K의 각 nibble은 `mse_index | (positive_sign << 3)`로 저장한다.
기존 K packed payload 크기를 유지하고 `tq_key_L_qjlscale_{in,out}`만
추가한다. Qwen3-1.7B, 28 layers, 8 KV heads, D128, C1024 기준:

| 구성 | K/V payload | vector별 메타데이터 | host KV |
|---|---|---|---:|
| int16 KV | K16 / V16 | 별도 codec scale 없음 | 112 MiB |
| Dense+Native | K4 / V4 | K FP16 1개, V FP16 1개 | 28.875 MiB |
| Dense+Native+QJL | K3+1 / V4 | K FP16 2개, V FP16 1개 | 29.3125 MiB |

QNN runner는 추가 stream을 기존 KV와 동일하게 append/reset/복사하며,
bucket 전환 시에도 동일한 token axis를 사용한다. 일반 Python
`TurboQuantKVCache`는 format 3을 아직 지원하지 않으며 오류를 내도록 했다.
QJL을 누락하고 실행하는 fallback은 없다. CPU oracle은 `QJLKeyReference`다.

### 17.2 attention 계산과 검증

기존 rotated-domain MSE score에
`(q @ S.T) @ (signs * a).T`를 **타일마다** 더한 다음 기존 mask/global
softmax를 적용한다. 현재 graph의 새 K는 압축하지 않은 경로를 그대로 쓰므로
QJL correction을 0으로 채운다. V는 기존 4-bit MSE/AV/output inverse rotation
경로를 유지한다.

과거 전체 K나 QJL residual을 원래 좌표계로 복원하지 않는다. K 역회전은
새 토큰 encoder에서 잔차를 계산할 때 한 번만 수행한다. Native Decode4
라이브러리 자체는 변경하지 않고 LUT를 바꿔 사용한다:

- MSE K: 8-entry 3-bit centroid를 두 번 반복한 16-entry LUT.
- QJL K: `[-1]*8 + [+1]*8` LUT.
- MSE V: 기존 16-entry 4-bit LUT.

따라서 이것은 Native decoder를 사용하는 tiled graph이며, unpack과 QK/AV
전체를 단일 커널로 합친 fused attention은 아니다. 추가 query projection,
QJL 타일 decode, score MatMul/Add 비용이 있다.

참조 QJL 행렬·부호는 CPU에서 정확히 일치했고 복원 오차 최대값은
`8.881784197001252e-16`이다. 두 새 Native LUT는 HTP에서 길이
1/3/127/128/255/256/1023 및 모든 byte pattern을 검사해 모두 FP16 oracle과
bit-exact였다. 관련 원본은 `~/.qaihm/tmp/turboquant/` 아래의
`qjl_{sign,mse3}_decoder_20260919/correctness.json`이다.

단독 encoder 검증(`qjl_encoder_20260919/correctness.json`)은 실제 KV
layer0/layer27와 0.5배 synthetic range, T1/T128을 사용했다. QJL 단계는
설명되지 않는 부호 오류 0개, 잔차 scale 최대 상대 오차 0.19731%로 기존
0.2% 기준을 통과했다. 그러나 MSE effective scale은 두 T128 실제 KV
case에서 0.21314% / 0.20707%로 기준을 넘었다. **전체 encoder gate는 실패**이며
허용오차를 완화하지 않았다. 기존 V4 encoder의 알려진 제한도 해결했다고
주장하지 않는다. 이 수치 gate와 LLM 성능/PPL 결과는 구분해야 한다.

### 17.3 실기기 비교 결과 (2026-09-19)

S26 Ultra / SM8850 HTP V81, Qwen3-1.7B W4A16, 최대 context 1024에서
int16 KV / 기존 Dense+Native K4/V4 / 새 Dense+Native K3+QJL/V4를 비교했다.
int16은 모델 전체가 FP16이라는 뜻이 아니라 **KV가 uFxp_16인 baseline**이다.
같은 runner와 burst 설정을 사용했으며, 두 TurboQuant의 Native DSP library도
동일하다. 기존 int16/Dense context 바이너리를 수정하지 않고 새 이름으로
배포했다. 입력 SHA256과 config hash도 원본 리포트에서 확인했다.

성능은 구성 및 입력 조건당 **1회**, 총 6회다. profiling은 끄고 로딩은
TTFT에서 제외했다. warmup 제외, 온도 통제, 실행 순서 교차, 분산 추정은
하지 않았다. 기능 진단과 PPL 실행 시간은 성능 표에 포함하지 않는다.
PPL도 세 구성 모두 이번에 같은 1024-token WikiText window 4개를 새로
측정했으며, window별 평균 PPL이 아니라 합산 NLL / 4092로 계산했다.

주요 조건은 **897-token prompt + 128-token 생성**이다. 마지막 생성 토큰은
다시 입력하지 않으므로 캐시는 정확히 1024에 도달하고, 127개 decode step은
모두 C1024를 사용한다.

| 지표 | int16 KV baseline | 기존 Dense+Native K4/V4 | Dense+Native K3+QJL/V4 |
|---|---:|---:|---:|
| TTFT | 304.176 ms | 526.543 ms | 574.604 ms |
| prefill | 2950.761 tok/s | 1704.871 tok/s | 1562.162 tok/s |
| decode | 35.8169 tok/s | 38.1123 tok/s | 33.1602 tok/s |
| decode/token | 27.920 ms | 26.238 ms | 30.157 ms |
| host KV 저장소 | 112.000 MiB | 28.875 MiB | 29.3125 MiB |
| resident graph I/O buffer | 263.179 MiB | 222.197 MiB | 223.782 MiB |
| 종료 VmRSS | 391.262 MiB | 280.035 MiB | 300.508 MiB |
| 프로세스 VmHWM | 605.055 MiB | 605.246 MiB | 604.980 MiB |
| PPL, 4 window / 4092토큰 | 20.108306 | 20.607753 | 34.721674 |

짧은 조건은 **35-token prompt + 128-token 생성**이다.

| 지표 | int16 KV baseline | 기존 Dense+Native K4/V4 | Dense+Native K3+QJL/V4 |
|---|---:|---:|---:|
| TTFT | 40.483 ms | 62.317 ms | 59.540 ms |
| prefill | 868.493 tok/s | 565.178 tok/s | 591.528 tok/s |
| decode | 34.9560 tok/s | 54.5074 tok/s | 51.6818 tok/s |

짧은 조건에서 int16은 고정 C1024이고 두 TurboQuant는 C128 93step /
C256 34step이다. 버킷화한 int16과의 비교가 아니므로 짧은 문맥 차이를
Native/QJL 커널만의 효과로 해석하지 않는다. 긴 조건의 prefill도 버킷
사용이 다르다.

해석과 제한:

- QJL 구성의 긴 문맥 decode는 기존 TurboQuant 대비 **12.99%**,
  int16 대비 **7.42%** 낮다. 기존 TurboQuant는 int16 대비 6.41% 높다.
  모두 단회 관측값이며 통계적으로 안정적인 차이라고 주장하지 않는다.
- 긴 문맥 평균 host prepare / QNN 호출 / commit은 int16
  **4.691 / 22.641 / 0.223 ms**, 기존 TurboQuant
  **1.126 / 24.511 / 0.015 ms**, QJL
  **1.043 / 28.631 / 0.015 ms**다. 관측된 지연 증가는 주로 QNN 호출에
  있다. QJL에는 새 K 잔차 encoder, query projection, 타일 sign decode와
  score 보정이 추가되지만, 별도 op profiling 없이 각 연산의 기여도를
  확정하지 않는다.
- QJL host KV는 int16 대비 **73.83%** 작다. 기존 TurboQuant보다
  0.4375 MiB 늘어난 것은 추가 FP16 K 잔차 scale 때문이다.
  종료 RSS는 줄어도 프로세스 VmHWM은 세 구성 모두 약 605 MiB다.
  host KV / RSS / VmHWM을 NPU 전체 또는 peak HTP 메모리로 해석하지 않는다.
- **QJL PPL은 34.721674로 기존 TurboQuant보다 크게 나빠졌다.**
  이번 결과는 속도나 품질 개선을 보여주지 않는다. K4→K3 비트 축소,
  고전적 QJL 보정 및 배포 FP16 연산의 영향을 분리한 ablation은 하지
  않았으므로 원인을 한 항목으로 단정하지 않는다. 특히 §17.2의 작은
  scale 오차만으로 이 PPL 증가를 설명했다고 주장하지 않는다.
  기본 profile은 기존 Dense+Native K4/V4로 유지하고 QJL은 실험형으로 둔다.
- Reset 2회/8-token 출력 동일, EOS 10번째 토큰 종료, 600-token 생성 중
  C128→256→512→1024 전환 및 긴 성능 실행의 1024 캐시 경계 검사를 통과했다.
  최종 attention DLC 21개 구조 감사도 통과했다. Native op는 총 1456개이며
  encoder MSE 복원과 타일 QJL sign decode도 포함한다. 최대 단일 FP16 KV
  타일은 512 KiB다. 이는 전체 동시 할당량이나 peak memory가 아니다.
- TurboQuant 단위 테스트 **285개**와 변경 파일 pre-commit(mypy 포함)이
  통과했다. 이 결과는 **전체 HTP encoder 수치 gate 실패**를 대체하지 않는다.

### 17.4 산출물과 재현

기준 디렉터리: `~/.qaihm/tmp/turboquant/`.

- 최종 QJL 번들: `qwen3_1_7b_qjl_native_20260919_final/`.
  `*_v3_part2/`는 part1+2, `*_v3_part3/`, `*_v3_part4/`는 각 part의
  빌드 산출물이다. 최종 번들만 배포/평가에 사용했다.
- 비교군: `qwen3_1_7b_baseline_int16_kv_cl1024/`,
  `qwen3_1_7b_dense_native_buckets_final/`.
- 기기 번들 이름: `qjl_compare_20260919_{baseline_int16,dense_native,qjl_native}`.
  공통 runner: `qnn_runner/qjl-20260919/qnn-llm-runner`.
- 종합 결과: `reports/qjl_20260919_v3/comparison_qjl.json`.
  같은 디렉터리의 `experiment.json`에 runner SHA256과 구성 해시를 기록했다.
- 성능 원본: `perf_{baseline_int16,dense_native,qjl_native}_{short,long}_once.json`.
  품질 원본: `score_{baseline_int16,dense_native,qjl_native}_w{0,1,2,3}.json`.
- 구조/기능: `boundary_qjl_native.json`,
  `generation_qjl_native_{reset,eos,switches}.json`.
- 수치 진단: `qjl_encoder_20260919/correctness.json`,
  `qjl_{sign,mse3}_decoder_20260919/correctness.json`.

새 성능 실험은 `benchmark_qjl_once.py`에 위의 세 번들, runner, assets,
새 `--reports` 및 고유 `--device-prefix`를 지정해 `push` → `functional`
순서로 실행한다. 기능 결과를 확인한 뒤 `performance` → `quality`를 실행한다.
스크립트는 기존 리포트나 시도 로그를 덮어쓰지 않는다. 원본 리포트만
다시 집계하려면 (기존 요약을 보존하도록 새 출력 이름 사용):

```bash
PYTHONPATH=src python scripts/llm/turboquant/summarize_qjl_results.py \
    --reports ~/.qaihm/tmp/turboquant/reports/qjl_20260919_v3 \
    --encoder-report ~/.qaihm/tmp/turboquant/qjl_encoder_20260919/correctness.json \
    --out ~/.qaihm/tmp/turboquant/reports/qjl_20260919_v3/comparison_recheck.json
```

## 18. 출처

turboquant_plus(Copyright 2026 Tom Turney, Apache-2.0, https://github.com/TheTom/turboquant_plus, commit `ba52ad1`). 이 구현은 참조 코드를 복사하지 않고 알고리즘을 재구현했다. 참조를 실행해 얻은 codebook·sign 상수와 golden fixture에는 출처와 commit을 기록했다.
