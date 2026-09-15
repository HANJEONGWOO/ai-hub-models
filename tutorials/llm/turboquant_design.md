# Qwen3 TurboQuant KV-cache — 설계(ABI·수치 계약)와 P0–P3 결과

작성일: 2026-09-15 · 작업 브랜치: `turboquant-kv-cache` (기준 commit `2a895603e`)

작업 명세: [turboquant_npu_implementation_spec.md](turboquant_npu_implementation_spec.md). 이 문서는 명세 P0의 `design.md` 산출물이며, P1(참조 구현), P2(최소 HTP 실행 검증), P3(Qwen3-1.7B 실기기 통합, context 1024)의 실제 결과를 포함한다. P4의 체계적 벤치마크(여러 context, 응답 품질 평가, HTP 측 메모리 계측)와 P5(다른 모델 크기)는 수행하지 않았다.

## 1. 완료 상태 (명세 10.2 기준)

| 상태 | 판정 | 근거 |
|---|---|---|
| 참조 구현 완료 | **완료** | 고정 commit 원본 대비 index 불일치 0, FWHT 복원 오차 0.0, packed byte 동일 (§4.4). 단위 테스트 138개 통과 |
| 최소 codec HTP 실행 (P2) | **완료** | S26(SM8850) HTP에서 encode/decode 8개 그래프가 fp16 허용오차로 oracle과 일치, 모든 op이 accelerator 프로파일에 기록됨 (§6) |
| NPU 기능 검증 완료 (1.7B 전체 생성 루프) | **완료, 단 메모리 경로 조건 미충족** | `k8_v4`·`k4_v4` 모두 S26에서 prefill + 128토큰 decode, EOS 처리, 세션 reset, context 경계(1024) 통과. codec encode/decode op이 28개 layer 전부 HTP detailed profile에 기록됨 (§8). KV는 호출 사이에 host에서 packed로 보관되지만, 그래프 안에서 과거 KV 전체를 매 스텝 복원하므로 명세 5.3의 "최종 메모리 경로"는 아님 |
| 압축 효과 입증 | 부분 | host KV 저장소(56.0→28.9 MiB)와 프로세스 RSS(224.7→144.2 MiB) 감소를 측정. HTP 측 intermediate·scratch·shared memory는 측정하지 않음 |
| 품질 평가 완료 | 부분 | 실기기 WikiText teacher-forced PPL(4 window) 측정. 응답 품질 평가(Grace 등)와 retrieval 시험은 수행하지 않음 |
| 성능 개선 입증 | **미달** | 동일 runner 반복 측정에서 decode가 baseline 대비 19–27배 느림 (§8.4) |
| 타 모델 검증 완료 | 미착수 | 0.6B/4B/8B는 config·shape 테스트만 통과 |

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
| `src/qai_hub_models/models/templates/llm/turboquant/config.py` | 프로파일(`baseline_int8`, `k8_v4`, `k4_v4`, `k8_v3`, `k4_v3`), format version, `config_hash()`, 모델 shape 검사 |
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
| `.../turboquant/graph_surgery.py` | P3: split part ONNX의 `past_*_in`/`past_*_out`에 codec 서브그래프 삽입, encodings 재사용 |
| `src/qai_hub_models/test/test_models/test_turboquant_graph_surgery.py` | 합성 delta-KV part로 surgery 배선을 onnxruntime에서 검증 |
| `scripts/llm/turboquant/split_checkpoint.py` | 배포 AIMET checkpoint를 repo split 코드로 part별 번들로 로컬 분할 |
| `scripts/llm/turboquant/convert_parts.py` | part별 surgery(프로파일) → `qairt-converter --quantization_overrides` → `qairt-quantizer --enable_float_fallback` → prompt/token 그래프 weight-shared context binary |
| `scripts/llm/turboquant/qnn_runner/` | 공개 QNN C API로 작성한 Android runner(`qnn-llm-runner`): 이름 기반 I/O 역할, pyref KV 배치, generate/score, 세션 반복, detailed profile, JSON 리포트 |
| `scripts/llm/turboquant/run_device_llm.py` | RoPE 표·토큰 asset 생성, sha256 기반 push, 장치 실행과 리포트 수집 |

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
| `k8_v4` | 56 MiB (V만) | 1.75 MiB | 448 MiB(host float K. 배포 int8 K는 112 MiB) |

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
  - UINT_8 Concat/Gather/Transpose와 fp16→uint8 Cast는 prepare에서 거부된다 → unpack은 `Cast(UINT8→INT32)` + `Gather([256,2] fp16 LUT)`
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
| `k8_v4` (K float) | 18.535 | −0.11% | 0.00096 | 99.0% |
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
- 명세의 순서(`k8_v4` 먼저)대로 진행하는 것이 타당하다. `k4_v4` 채택 여부는 K outlier 처리나 K bit 상향 같은 추가 실험과 합의가 필요하다. 합격 임계값은 사용자가 정하지 않았으므로 판정하지 않는다.

## 8. P3: Qwen3-1.7B 실기기 통합 (context 1024)

### 8.1 선택한 경로

- **재양자화 없음.** 배포된 w4a16 checkpoint(`qwen3_1_7b` asset v2, SpinQuant R1+R3 → AdaScale → Calibration)를 repo split 코드로 4개 part로 나눈다. 기존 weight와 activation encodings를 그대로 쓰고 KV 저장 방식만 바꿔 baseline과 비교한다.
- **ONNX surgery**(`graph_surgery.py`):
  - `past_{kind}_{L}_in`을 `tq_{kind}_{L}_{packed,norm}_in`으로 교체하고, 복원 서브그래프가 기존 Slice→Concat 소비자로 들어간다. K는 복원 뒤 token 축을 hub layout으로 되돌린다.
  - `past_{kind}_{L}_out`은 내부 텐서로 남기고, encode 서브그래프가 `tq_*_out`을 만든다.
  - 제거한 `past_*_in` encodings만 삭제한다.
- **로컬 변환**(`convert_parts.py`): `qairt-converter --quantization_overrides` 뒤 `qairt-quantizer --enable_float_fallback --float_bitwidth 16 --act_bitwidth 16`.
  - encodings가 없는 codec 서브그래프는 fp16으로 남는다. QAIRT가 int8↔fp16 경계에 `Convert` op을 자동으로 넣고, codec 상수(회전·경계·LUT)는 fp16으로 유지됨을 dlc-info로 확인했다.
  - 둘째 part부터 prompt(AR=128)와 token(AR=1) 그래프를 weight-sharing context binary 하나로 묶었다. 그래프 변환 1개당 약 2분이 걸렸다.
- **전용 runner**(`qnn_runner/`):
  - SampleApp·Genie 소스는 QAIRT 라이선스상 소스 재배포가 허용되지 않아, 헤더의 공개 API만으로 새로 작성했다.
  - tensor 이름으로 역할을 정하므로 codec 스트림도 int8 KV와 같은 경로로 처리한다.
  - KV 배치는 ai-hub-models `HubCompatibleGenerator`와 같은 pyref 방식(오른쪽 정렬 past, 앞쪽 pad)이다.
  - RoPE는 Python에서 만든 float32 표를 쓰고, 양자화 규칙은 Python과 같다(round-half-even).
  - host는 KV를 packed 상태로 보관하고, 매 스텝 그래프 입력 버퍼로 복사한다. RAW client buffer를 쓴다.
- **이중 양자화 명시.** 배포 w4a16 그래프의 새 토큰 K/V는 이미 int8 activation이다. K는 R3 Hadamard 뒤 텐서다. 따라서 이번 codec은 **int8 activation을 입력으로 4-bit 압축**한다. PC 참조 평가(§7)는 float K/V를 압축했으므로 경로가 다르다.

### 8.2 baseline 재현 확인

| 번들 | 생성 결과(35토큰 프롬프트) | WikiText window 0 PPL |
|---|---|---:|
| AI Hub 빌드 번들(QAIRT 2.45, 10 graph/part) | "Gravity is the force that pulls objects toward Earth." | 11.586 |
| 로컬 변환 baseline(QAIRT 2.48, 2 graph/part) | 같은 첫 문장, EOS 뒤 continuation은 다름 | 11.651 (+0.55%) |

같은 runner로 AI Hub 번들과 로컬 번들이 모두 동작했고, 로컬 변환이 baseline 품질을 재현했다. EOS 뒤 차이는 QAIRT 버전 간 수치 차이로 보이며, 원인은 검증하지 않았다.

### 8.3 기능 검증

모든 항목을 세 프로파일(`baseline_int8`, `k8_v4`, `k4_v4`)에서 같은 runner로 실행했다.

| 검증 | 결과 |
|---|---|
| prefill + 128토큰 greedy decode | 세 프로파일 모두 첫 답변 동일, 128토큰 생성 완료 |
| EOS 처리 | `--stop-on-eos`에서 EOS(`<\|im_end\|>`, 11번째 토큰)로 중단하고 리포트에 `stop_reason=eos` 기록 |
| 세션 reset | 한 프로세스에서 reset 후 재실행한 세션의 생성 토큰이 첫 세션과 전부 동일(2–4세션) |
| context 경계 | 897토큰 프롬프트 + 128토큰 생성으로 KV 1024/1024 채움. 2세션 동일, 생성 문장이 WikiText 문맥을 이어감 |
| 초과 거부 | `P + N − 1 > C`이면 runner가 시작 전에 오류(host 검사) |
| NPU 실행 증거 | QNN detailed profile에서 decode 1스텝 기준 `k8_v4`는 part당 V codec op 약 410개, `k4_v4`는 K+V op 약 820개가 모두 accelerator cycle로 기록됨. `k8_v4`는 prefill 청크 profile에서도 28개 layer의 encode·decode op이 기록됨(`k4_v4` prefill profile은 수집하지 않음). HTP 백엔드에는 CPU 분할 옵션이 없고 context는 `dspArch 81 / socModel 87` |

### 8.4 동일 조건 측정 (S26, CL=1024, 35토큰 프롬프트, 128토큰 생성, warmup 1회 뒤 3회 median)

| 지표 | `baseline_int8` | `k8_v4` | `k4_v4` |
|---|---:|---:|---:|
| TTFT | 48 ms | 596 ms | 740 ms |
| prefill (35토큰 청크) | 730 tok/s | 58.8 tok/s | 47.4 tok/s |
| decode | 38.97 tok/s (25.7 ms/tok) | 2.05 tok/s (486.7 ms/tok) | 1.42 tok/s (703.0 ms/tok) |
| 반복 간 decode 편차 (min–max) | 38.95–39.00 | 2.053–2.061 | 1.421–1.425 |
| host KV 저장소 (1024토큰 할당) | 56.0 MiB | 42.4 MiB | 28.9 MiB |
| runner I/O 버퍼 | 151.2 MiB | 124.1 MiB | 96.9 MiB |
| 프로세스 VmRSS (종료 시) | 224.7 MiB | 184.5 MiB | 144.2 MiB |
| WikiText PPL (4 window, 4,092토큰 통합) | 19.903 | 19.933 (+0.15%) | 20.889 (+4.95%) |

- 측정 범위:
  - TTFT는 prefill 시작부터 첫 argmax까지이며 모델 로딩은 제외한다(로딩 0.5–1.1초는 리포트에 따로 기록).
  - RSS는 host 프로세스 기준이며, HTP·DMA-BUF 측 메모리는 포함하지 않는다.
  - runner는 RAW client buffer와 pyref 전체 복사를 쓰므로 baseline 수치도 Genie 수준의 최적 경로가 아니다.
- decode가 느린 원인(detailed profile):
  - 매 스텝 과거 1023토큰 전체를 복원하는 decode 서브그래프가 accelerator cycle의 약 81%(`k8_v4`), 약 85%(`k4_v4`)를 차지한다.
  - 그중 LUT `Gather`(`dec_centroid_pairs`)가 codec cycle의 약 77%다.
  - 새 토큰 encode는 layer당 약 40만 cycle로 작다.
- 품질:
  - `k8_v4`는 이 측정 범위에서 저하가 드러나지 않았다.
  - `k4_v4`는 +4.95%로, PC 참조 평가(float K/V 압축, +30%)보다 훨씬 작다. 원인 후보는 두 가지이며 검증하지 않았다: 장치 경로의 K가 R3 Hadamard 회전 뒤 int8 activation이라는 점, 그리고 비교 baseline 자체가 int8 KV라는 점이다.
  - PPL 개선이나 품질 동등을 주장하지 않는다(window 4개, 단일 실행). 응답 품질 평가는 수행하지 않았다.

### 8.5 명세 대비 남은 조건과 다음 단계

1. **메모리 경로(명세 5.3):** 복원이 그래프 안에서 전체 past에 대해 일어난다. 복원 결과(fp16)와 int8 변환 텐서가 HTP intermediate로 잡히므로, packed 저장만으로는 실제 peak 메모리 이득을 주장할 수 없다. tile 단위 복원이나 packed 소비형 attention 융합이 필요하다.
2. **속도:** decode 경로의 LUT Gather가 병목이다. 후보는 세 가지이며 모두 실험 전이다.
   - 복원을 과거 토큰 증분만 처리하도록 바꾸는 구조(예: 이전 스텝의 int8 past를 그래프 출력으로 유지)
   - custom HVX op(`TQDecodeTile`)
   - attention과 융합
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

# 프로파일별 변환 (baseline_int8 | k8_v4 | k4_v4); part 한 그래프당 약 2분
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

이번 측정의 baseline과 `k8_v4` 번들은 그래프 이름 수정 전에 변환되어 `--graph-suffix _float`로 실행했다. 이후 변환부터는 접미사가 붙지 않는다.

장치 파일은 `/data/local/tmp/qaihm_turboquant/` 아래에만 쓴다. 클라우드 작업은 제출하지 않는다.

## 10. 미지원 항목과 잔여 리스크

- 미지원: QJL, 3-bit NPU 그래프(3-bit는 host oracle만), batch>1, head_dim≠128, 128보다 긴 prefill chunk 그래프, native KV, 0.7B(현재 repo에 해당 checkpoint가 없음. 0.6B는 shape 테스트만 수행).
- HTP fp16 index는 경계 근처에서 oracle과 다를 수 있다(실제 KV에서 측정 ≤0.19%). byte 단위 재현이 필요한 용도에는 fp32 host 경로만 bit-exact를 보장한다.
- host cache(`cache.py`)는 `new == 128`일 때 shape로 K layout 실수를 잡지 못한다. 장치 runner는 I/O 이름으로 역할을 정하므로 이 문제가 없다.
- Hexagon SDK 6.6.0.0(tools 19.0.07)은 QAIRT 문서가 V81용으로 명시한 6.4.0/19.0.04와 다르다. custom op이 필요해지면 ABI 호환부터 확인해야 한다.
- P3 codec은 int8 activation을 입력으로 압축한다(§8.1). float 입력 경로는 재양자화나 encodings 변경이 필요하며 수행하지 않았다.
- 성능은 baseline보다 크게 느리고, HTP 측 peak 메모리는 측정하지 않았다(§8.4, §8.5).
- 변환한 context는 1024 하나다. 다른 context 길이, 3-bit 프로파일, 128 초과 prefill 청크는 실행하지 않았다.
- QAIRT 2.45(AI Hub)와 2.48(로컬) 사이 수치 차이로 EOS 뒤 생성이 달라진다. 비교는 반드시 같은 변환 경로의 번들끼리 해야 한다.

## 11. 출처

turboquant_plus(Copyright 2026 Tom Turney, Apache-2.0, https://github.com/TheTom/turboquant_plus, commit `ba52ad1`). 이 구현은 참조 코드를 복사하지 않고 알고리즘을 재구현했다. 참조를 실행해 얻은 codebook·sign 상수와 golden fixture에는 출처와 commit을 기록했다.
