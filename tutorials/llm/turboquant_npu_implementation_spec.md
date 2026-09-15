# Qwen3 TurboQuant KV-cache — 실기기 NPU 구현 작업지시서

작성일: 2026-09-15

문서 상태: 구현을 위한 작업 명세. 이 문서 작성 시점에는 TurboQuant 코드 변경, 모델 재변환, 장치 실행을 수행하지 않았다.

## 1. 목표와 범위

`ai-hub-models`를 최대한 재사용하여 Qwen3-1.7B에 TurboQuant 계열 KV-cache 압축을 적용하고, Qualcomm 실기기 NPU/HTP에서 실행·검증한다. 이후 다른 Qwen3 크기에도 같은 구현을 설정으로 적용할 수 있게 한다.

필수 목표:

- Qwen3-1.7B의 prefill 및 autoregressive decode에 압축 KV-cache를 연결한다.
- KV 압축/복원 또는 이를 포함하는 융합 연산이 실기기 HTP에서 실행되어야 한다. CPU/GPU codec 실행을 NPU 구현 완료로 간주하지 않는다.
- 모델 호출 사이에 KV-cache를 실제 packed 상태로 보관한다. 저비트 값을 float/int8 원소 하나씩에 담는 fake quantization만으로 끝내지 않는다.
- 기존 모델 분리, ONNX 생성, 양자화, QNN 변환, 실행·벤치마크 자동화를 가능한 범위에서 재사용한다.
- 정확도 변화, KV 메모리, 전체 peak memory, TTFT, prefill/decode 속도를 동일 조건의 baseline과 비교한다.
- 기본 export 및 다른 LLM의 동작은 유지하고, TurboQuant는 명시적인 opt-in 기능으로 추가한다.

PyTorch/NumPy 실험은 참조값 생성과 디버깅을 위한 중간 단계다. 최종 산출물은 PC 정확도 실험이 아니라 실기기 NPU 실행 및 측정 결과다. 속도 향상은 사전에 보장하지 않는다.

1차 범위는 batch=1, text-only, 일반적인 prefill→decode, 단일 세션이다. Beam search, speculative decoding, 다중 사용자 동시 실행, 4K 초과 context 확장은 후속 작업으로 분리한다. 지원하지 않는 조합은 명확한 오류를 내야 한다.

## 2. 기준 버전과 대상 모델

### 2.1 버전 고정

- 조사한 `ai-hub-models` commit: `2a895603ec90151e92d49ab86db2ed8c352d6f03`.
- 알고리즘 기준: [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus/tree/ba52ad107d1fdd02bc9be8fd85308226b75c905b).
- 참고 commit: `ba52ad107d1fdd02bc9be8fd85308226b75c905b`.
- 구현을 시작할 때 실제 작업 브랜치와 차이를 확인한다. 원격 `main`의 변경을 자동 추종하지 않는다.
- 모델 checkpoint/revision, tokenizer, quantization recipe, QAIRT, Hexagon toolchain, 실행 엔진, 장치 OS/SoC/HTP 버전을 실험 manifest에 기록한다.

### 2.2 모델별 적용

아래는 현재 로컬 모델 코드의 값이다. 구현 시 checkpoint config와 대조하며, codec 내부에 이 값을 하드코딩하지 않는다.

| 우선순위 | 모델 ID | 레이어 | Q heads | KV heads | head_dim | 현재 split 수 |
|---|---|---:|---:|---:|---:|---:|
| 1차 실기기 검증 | `qwen3_1_7b` | 28 | 16 | 8 | 128 | 4 |
| 소형 모델 후보 | `qwen3_0_6b` | 28 | 16 | 8 | 128 | 2 |
| 확장 | `qwen3_4b` | 36 | 32 | 8 | 128 | 4 |
| 확장 | `qwen3_8b` | 36 | 32 | 8 | 128 | 5 |

사용자가 언급한 **0.7B에 대응하는 모델은 현재 repo에서 확인되지 않았다.** 현재 확인된 소형 Qwen3는 0.6B다. 0.6B를 지원 후보로 설계하되, 0.7B 요청을 충족했다고 표시하지 않는다. 별도 0.7B checkpoint를 의미한다면 정확한 HF ID/config를 확인한 뒤 어댑터를 추가한다. Qwen3.5 등 다른 아키텍처로 임의 대체하지 않는다.

모델별 근거: [1.7B](../../src/qai_hub_models/models/qwen3_1_7b/model.py), [0.6B](../../src/qai_hub_models/models/qwen3_0_6b/model.py), [4B](../../src/qai_hub_models/models/qwen3_4b/model.py), [8B](../../src/qai_hub_models/models/qwen3_8b/model.py).

## 3. 알고리즘 기준 — QJL을 잘못 가져오지 말 것

### 3.1 실사용 경로와 연구용 경로 구분

참고 repo의 [README](https://github.com/TheTom/turboquant_plus/blob/ba52ad107d1fdd02bc9be8fd85308226b75c905b/README.md)는 실사용 경로에서 K와 V 모두 QJL을 제외한다고 명시한다. 반면 Python [`TurboQuant`](https://github.com/TheTom/turboquant_plus/blob/ba52ad107d1fdd02bc9be8fd85308226b75c905b/turboquant/turboquant.py)는 여전히 `(b-1)-bit PolarQuant + 1-bit QJL`이며, [`KVCacheCompressor`](https://github.com/TheTom/turboquant_plus/blob/ba52ad107d1fdd02bc9be8fd85308226b75c905b/turboquant/kv_cache.py)는 이 클래스를 K에 사용한다.

따라서 클래스를 이름만 보고 그대로 연결하지 않는다. 이번 기본 구현은 참고 repo의 실사용 정책을 따른다.

| 프로파일 | K | V | QJL | 용도 |
|---|---|---|---|---|
| `baseline_int8` | 기존 repo의 int8 KV | 기존 repo의 int8 KV | 없음 | 주 비교 기준 |
| `k8_v4` | 기존 int8 K | 4-bit PolarQuant | 꺼짐 | V-only 단계의 연결·정확도 확인 |
| `k4_v4` | 4-bit PolarQuant | 4-bit PolarQuant | 꺼짐 | 1차 전체 압축 구현 |
| `k8_v3`, `k4_v3` | 프로파일별 지정 | 3-bit PolarQuant | 꺼짐 | 후속 압축률 실험 |
| `qjl_reference` | 별도 명세 | 별도 명세 | 명시적 연구 옵션 | 추가 요청 시 비교 실험 |

여기서 K8은 llama.cpp의 `q8_0`가 아니라 이 repo baseline의 affine int8 KV 경로다. 기본 4-bit는 16개 centroid의 PolarQuant이며, 3-bit centroid와 QJL 1-bit를 합친 형식이 아니다.

QJL 제외 이유는 참고 프로젝트의 실험적 선택으로 기록한다. 해당 결과가 모든 모델·NPU에서 보편적으로 성립한다거나, 같은 성능 개선이 재현된다고 단정하지 않는다. 근거 문서: [turbo4-resurrection](https://github.com/TheTom/turboquant_plus/blob/ba52ad107d1fdd02bc9be8fd85308226b75c905b/docs/papers/turbo4-resurrection.md).

### 3.2 PolarQuant 구현 계약

다음 계산을 독립된 CPU 참조 구현과 NPU 구현에서 공유한다.

1. 각 KV head의 토큰 벡터를 `head_dim` 방향으로 블록화한다. 1차 블록 크기는 128이다.
2. 블록의 L2 norm을 추출하고 unit-norm으로 정규화한다. 영벡터는 유한값으로 처리하며 복원 결과도 영벡터여야 한다.
3. 고정 random sign과 정규화된 Walsh–Hadamard transform으로 회전한다.
4. 고정 Lloyd–Max 계열 codebook으로 nearest-centroid index를 계산한다.
5. index를 실제 bit packing하고, 블록 norm과 함께 저장한다.
6. 복원 시 centroid lookup, 선택된 norm correction, 역회전, 원래 norm 복원을 수행한다.

참고 Python [`PolarQuant`](https://github.com/TheTom/turboquant_plus/blob/ba52ad107d1fdd02bc9be8fd85308226b75c905b/turboquant/polar_quant.py)는 dense QR rotation을 사용한다. FWHT 함수가 존재한다는 이유로 Python 클래스도 FWHT를 쓴다고 가정하면 안 된다.

- `dense_reference`와 `fwht_reference`를 구분한다. NPU 결과는 동일한 FWHT·상수·packing을 쓰는 참조값과 비교한다.
- FWHT의 sign 적용 순서, `1/sqrt(block_size)` 정규화, 역변환 순서를 명세한다. 참조 [`rotation.py`](https://github.com/TheTom/turboquant_plus/blob/ba52ad107d1fdd02bc9be8fd85308226b75c905b/turboquant/rotation.py)의 fast 경로는 순방향 `D2 H D1`, 역방향 `D1 H D2`다.
- RNG를 장치에서 다시 실행하는 대신, 생성한 sign/codebook 상수를 export하고 해시를 기록한다. K/V의 seed 정책도 고정한다.
- codebook은 [`codebook.py`](https://github.com/TheTom/turboquant_plus/blob/ba52ad107d1fdd02bc9be8fd85308226b75c905b/turboquant/codebook.py)를 기준으로 생성하고, 문서에 반올림된 centroid 숫자를 손으로 복사하지 않는다. 경계값의 동률 처리도 고정한다.
- `norm_correction=true`를 초기 정책으로 사용한다. 참조 구현처럼 centroid 복원 벡터를 재정규화하는 동작이며, 생략하거나 norm에 미리 접어 넣을 때에는 동등성 테스트가 필요하다.
- norm 저장은 FP16을 첫 후보로 두되 overflow/underflow 및 정확도 검증을 통과해야 한다. FP32로 바꾸면 format/config/메모리 결과에 명시한다.
- non-power-of-two 또는 block에 맞지 않는 차원을 조용히 truncate하지 않는다. 1차 미지원이면 오류를 내고, 추후 padding을 지원할 때는 padded 좌표와 역변환을 포함하여 검증한다.
- 이 명세는 알고리즘·수치 정책의 이식이다. llama.cpp의 기존 binary cache format과 byte-compatible하다고 주장하지 않는다.

### 3.3 연구용 QJL을 추가하는 경우의 계약

기본 NPU 경로에는 QJL과 관련 버퍼·연산을 넣지 않는다. 연구용 QJL 추가는 1차 필수 범위가 아니다. 추가한다면 논문을 기억에 의존해 재구현하지 말고, 고정 commit의 [`qjl.py`](https://github.com/TheTom/turboquant_plus/blob/ba52ad107d1fdd02bc9be8fd85308226b75c905b/turboquant/qjl.py) 및 `TurboQuant`를 수치 기준으로 삼는다.

- QR로 생성한 `d×d` 직교 행렬 `S`를 사용하며 PolarQuant와 독립된 seed를 사용한다. 해당 클래스의 QJL seed는 `seed + 1000`이다.
- residual은 동일 PolarQuant 복원값을 사용한 `r = x - x_mse`로 계산한다.
- `z = sign(Sr)`이고, 0의 sign은 `+1`로 처리한다.
- 복원식은 `r_hat = alpha * sqrt(pi/2) / sqrt(d) * ||r|| * S^T z`이다.
- 소스 기본 `alpha=1.0`과 선택적 shrinkage `alpha=2/pi`를 구분한다. shrinkage를 기본 동작인 것처럼 바꾸지 않는다.
- QJL을 WHT로 대체하거나 MSE 회전과 같은 행렬을 재사용하면 별도 변형으로 표시한다.
- signs는 실제 1-bit packing한다. 원본 norm과 residual norm을 모두 저장·계산하며, QJL을 끄면 둘째 norm/sign 버퍼도 제거한다.
- 소스 주석의 unbiasedness 등 이론적 표현을 목표 차원에서의 검증 없이 추가 보장으로 내세우지 않는다. 구현 일치성과 실제 attention 오차를 각각 측정한다.

## 4. 기존 repo에서 수정·재사용할 위치

| 기존 위치 | 작업 내용 |
|---|---|
| [templates/qwen3/model_adaptations.py](../../src/qai_hub_models/models/templates/qwen3/model_adaptations.py) | Q/K normalization, RoPE, cache update, QK/AV 연결 지점에 opt-in codec 적용 |
| [templates/qwen3/model.py](../../src/qai_hub_models/models/templates/qwen3/model.py) | Qwen3 공통 구성, PreSplit/Part/Collection에 codec 설정 전달 |
| [templates/llm/model.py](../../src/qai_hub_models/models/templates/llm/model.py) | packed I/O spec, split 경계, ONNX export, 양자화 설정, 모델 캐시 키 |
| [templates/llm/common.py](../../src/qai_hub_models/models/templates/llm/common.py) | 기존 `LLMIOType`과 새 cache ABI의 관계 정의 |
| [sha_dynamic_kvcache.py](../../src/qai_hub_models/models/templates/llm/sha_dynamic_kvcache.py), [native_kv_cache.py](../../src/qai_hub_models/models/templates/llm/native_kv_cache.py) | 기존 cache 표현 분석, packed cache adapter 구현 |
| [lm_driver/generator.py](../../src/qai_hub_models/models/templates/lm_driver/generator.py), [native_kv_generator.py](../../src/qai_hub_models/models/templates/llm/native_kv_generator.py) | 참조 생성기의 append/reset/prefill→decode 상태 관리 |
| [templates/llm/_utils.py](../../src/qai_hub_models/models/templates/llm/_utils.py), [quantize.py](../../src/qai_hub_models/models/templates/llm/quantize.py) | int8 KV tying 및 precision 검사와 codec I/O를 분리 |
| [utils/export/multi_graph_collection_pipeline.py](../../src/qai_hub_models/utils/export/multi_graph_collection_pipeline.py) | 현재 Collection export 경로와 수정 모델 번들 연결 |
| [run_geniex_bench_benchmarks.py](../../src/qai_hub_models/scripts/run_geniex_bench_benchmarks.py), [utils/llm/geniex/jobs.py](../../src/qai_hub_models/utils/llm/geniex/jobs.py) | 수정 artifact 선택, 실행·수집, 실험 메트릭 확장 |

위 파일을 모두 무조건 수정하라는 뜻은 아니다. 호출 경로를 확인하여 최소한의 확장점을 선택한다. 모델별 파일에 codec 알고리즘을 복제하지 않는다.

권장 신규 모듈 경계는 아래와 같다. 경로와 클래스명은 제안이며 현재 존재하는 API가 아니다.

```text
src/qai_hub_models/models/templates/llm/turboquant/
  config.py        # opt-in 설정, versioned format, config hash
  reference.py     # CPU 수치 oracle: dense / FWHT 구분
  packing.py       # byte 단위 pack/unpack 및 metadata
  cache.py         # packed cache 상태·append·reset·I/O adapter
  export.py        # ONNX lowering, 상수와 encoding 연결
scripts/llm/turboquant/
  ...              # fixture 생성, 로컬 변환, 실기기 실행·비교 도구
```

HTP C/C++ 구현, converter 설정, runner adapter, 테스트 파일의 최종 위치는 초기 설계에서 결정한다. Qualcomm SDK 전체 또는 배포 권한이 없는 바이너리를 repo에 넣지 않는다.

## 5. Cache ABI와 attention 연결

### 5.1 필수 ABI 명세

다음 항목을 Python → ONNX → QNN → runner 전체에서 일치시킨다.

- format version, K/V별 codec/bit width, block size, codebook/sign hash, norm dtype.
- layer/head/token/block/packed-byte 축 순서, stride, padding, alignment, endianness, bit 순서.
- logical length와 allocated capacity, cache position, 유효 토큰 mask, split별 global layer ID.
- packed data와 norms의 입력·출력 이름, raw integer tensor의 의미, ownership, aliasing 가능 여부.
- decode 한 단계의 새 토큰 쓰기 위치와 prefill chunk 처리 규칙.
- reset, context limit, 세션 분리 및 cache serialization의 버전 불일치 처리.

현재 native cache의 K는 `(kv_heads, batch, head_dim, context)`, V는 `(kv_heads, batch, context, head_dim)`이다. K/V의 시퀀스 축이 다르므로 bit packing 축을 기존 메모리의 마지막 축으로 무조건 잡으면 안 된다. 근거: [NativeKVLayer](../../src/qai_hub_models/models/templates/llm/native_kv_cache.py).

1차에는 기존 delta-cache 또는 native-cache 중 검증 가능한 하나를 선택해 명시적으로 지원한다. 다른 I/O 조합으로 조용히 잘못 실행되지 않게 검사한다. 기존 `genie_input_ids` 등의 이름을 유지한 채 tensor 의미만 바꾸지 않는다.

### 5.2 Attention 연결 규칙

- K는 기존 Qwen3의 `k_norm`과 RoPE 적용 이후 압축한다. 기존 Q/K normalization 및 attention scale 순서를 보존한다.
- V는 원래 V projection 결과에 적용하며 K처럼 RoPE를 추가하지 않는다.
- GQA의 KV head 단위로 저장한다. Q head 수만큼 KV-cache를 복제해서 메모리 이득을 잃지 않는다.
- 최초 구현은 동일 기준 좌표계로 복원한 K/V를 attention에 공급하는 방식으로 검증할 수 있다.
- 이후 회전 좌표계에서 직접 QK/AV를 계산하려면 Q의 대응 회전, K/V별 회전 차이, norm correction, AV 결과의 역회전 및 O projection 앞의 좌표계를 수학적으로 검증한다.
- token decode 단계마다 과거 KV 전체를 재양자화하지 않는다. 새 토큰/chunk만 encode한다.
- 이미 packed된 과거 token을 재압축하거나, 기존 int8 KV를 무조건 한번 거쳐 다시 저비트화하는 이중 양자화를 기본 구현으로 숨기지 않는다.

### 5.3 메모리 유지 조건

전체 KV를 복원하는 구현은 수치 디버깅용으로만 허용한다. 최종 메모리 검증 경로는 packed 상태를 유지하고, 필요 시 제한된 tile/block 단위로 복원하거나 attention과 융합한다.

- 원본 KV의 전체 fp16/fp32 복사본을 세션 내내 유지하지 않는다.
- native-cache 전체를 매 토큰 입출력 복사하는 비용도 계측한다. in-place/aliasing은 QNN이 실제 지원하고 검증한 경우에만 사용한다.
- QNN intermediate, HTP scratch, shared buffer, host mirror 및 double buffer를 peak memory에서 누락하지 않는다.
- cache 배열의 논리적 크기와 실제 allocator 할당량을 구분한다. packed 값을 담았더라도 allocation이 줄지 않았다면 실제 메모리 절감으로 보고하지 않는다.

## 6. 양자화 및 모델 artifact 관리

현재 `w4a16` 경로는 `_apply_int8_kv_cache_tying_and_lm_head`에서 KV I/O를 8-bit symmetric으로 묶고, `quantize.py`도 해당 precision 계약을 검사한다. `--precision w4a16`만 바꾸거나 int4 KV로 치환하는 것으로 TurboQuant가 구현되지 않는다.

필수 작업:

- 모델 weight/activation quantization과 KV codec 설정을 분리한다. 기존 `w4a16`의 weight 예외, lm_head 및 SpinQuant/AdaScale 관련 처리는 보존한다.
- raw packed bytes, centroid index, sign bit를 AIMET/QNN의 일반 affine activation으로 재양자화하지 않는다. backend dtype/encoding 처리의 byte 보존을 테스트한다.
- codec 입력·복원 출력의 dtype과 matmul encoding은 별도 설계한다. 기존 int8 KV tying을 전역 해제해서 다른 모델을 바꾸지 않는다.
- codec 설정을 모델 인스턴스 캐시, ONNX/context binary 캐시, compile job 식별자, metadata 및 artifact hash에 포함한다.
- 변경된 graph에 새 ONNX/encodings/context binary를 생성한다. 기존 encoding 재사용 가능 부분과 재보정 필요 부분을 검증하고 기록한다.
- 기존 다운로드 checkpoint가 사전 생성 ONNX를 사용하면 PyTorch 수정이 반영되지 않을 수 있다. 실제 변환 입력과 최종 graph에서 codec node/상수/packed I/O를 확인한다.
- baseline과 TurboQuant artifact를 별도 경로에 저장하며 원본 `DEFAULT_W4A16` 또는 배포용 release asset을 덮어쓰지 않는다.

## 7. 실기기 NPU 실행 경로 결정

### 7.1 먼저 작은 그래프로 실행 가능성을 검증

full-model 작업에 앞서 `encode → packed bytes/norms → decode` 최소 그래프를 만든다. norm, FWHT, codebook search/lookup, packing, 필요한 update 연산별 지원 dtype/shape를 기록한다.

1. 해당 QAIRT와 목표 HTP에서 표준 ONNX/QNN 연산으로 구현 가능한지 확인한다.
2. 변환 성공만으로 끝내지 않고 실기기에서 실행·정확성·배치를 검증한다.
3. 미지원 또는 비효율적인 부분에 한해 HTP custom op를 선택한다.

custom op가 없어도 목표를 충족한다면 억지로 만들지 않는다. 반대로 Python custom function이나 ONNX custom node 선언만으로 HTP kernel이 생긴다고 가정하지 않는다.

### 7.2 Custom op가 필요한 경우

아래를 하나의 실행 가능한 묶음으로 제공한다.

- ONNX custom domain/op schema와 shape/dtype/attribute 계약.
- 해당 SDK 버전에 맞는 converter mapping/config 및 Op Package 등록 절차.
- host-side 검증·shape inference/등록 부분과 HTP-side 실제 kernel 구현.
- 필요한 타깃별 라이브러리 빌드, 배포 위치, runtime package 등록·로드 절차.
- 정렬, tail 처리, scratch 크기, 오류 코드, 버전 호환성 테스트.

가능한 연산 경계는 `TQEncode`, `TQDecodeTile`, packed-cache 소비형 attention이다. 이는 제안명이며 기존 SDK 제공 op가 아니다. 일반 matmul/softmax 등은 재사용하되, 메모리·지연 측정 결과에 따라 융합 범위를 조정한다.

### 7.3 변환·runner·클라우드 선택

- 기본 Hub/GenieX 경로에서 수정 graph, Op Package, cache ABI를 지원하는지 먼저 조사한다.
- `ai-hub-models export` 또는 public Workbench job에 사용자 Op Package를 올릴 수 있다고 가정하지 않는다. 확인한 지원 범위와 실제 테스트 결과를 남긴다.
- 표준 경로로 처리하지 못하면 로컬 QAIRT/QNN 변환 + package 등록 + 장치 runner 확장 경로로 진행한다. 로컬 변환에는 별도 명령/스크립트와 환경 구성이 필요하다.
- 기존 GenieX 실행 엔진이 package 로딩과 packed KV lifecycle을 지원하면 재사용한다. 지원하지 않으면 엔진/adapter를 확장하거나 SDK 예제 기반의 최소 C++ runner를 작성한다.
- `qnn-net-run` 등은 단일 op/block 검증 후보이며, 그것만으로 LLM 생성 루프가 구현되었다고 간주하지 않는다.
- Android runner는 장치에서 실행하는 소프트웨어다. Android UI 앱은 필수가 아니다. 접근 가능한 물리적 장치와 ADB 또는 대응 device-farm 실행 환경은 필요하다.
- CPU에서 tokenizer, sampling, graph scheduling을 수행하는 것은 허용하되, KV encode/decode/attention의 실행 위치를 구분하여 기록한다.

NPU 실행 증거에는 목표 backend, package load 성공, graph/op 실행 프로파일과 fallback 여부를 포함한다. `desired_compute_unit: npu` 설정이나 장치 이름만으로 NPU 실행을 증명하지 않는다. 필요한 증거를 수집할 수 없으면 미검증으로 보고한다.

## 8. 구현 순서와 단계별 산출물

### P0. 환경·baseline·계약 고정

- 실제 목표 장치/SoC/HTP 및 사용 가능한 QAIRT·Hexagon SDK를 확인한다. Galaxy S26 이름만으로 칩셋 변형을 가정하지 않는다.
- 원본 Qwen3-1.7B의 동일 backend 실기기 실행을 확보한다. baseline 실패와 TurboQuant 실패를 구분한다.
- 최종 측정은 동일 runner/backend 비교를 원칙으로 한다. runner 변경이 필요하면 원본 cache 경로도 같은 runner에서 실행한다.
- baseline checkpoint/encoding, prompt token IDs, context, 생성 길이, seed, chat template 및 thinking mode를 고정한다.
- `design.md`에 cache ABI, 모듈 경계, 수치 허용오차와 선택한 실행 경로를 작성한다.

완료 조건: baseline 결과, 환경 manifest, ABI 초안 및 다음 단계에 필요한 접근 권한/도구 목록이 존재한다. 장치나 SDK가 없으면 완료 표시하지 않고, 가능한 오프라인 작업과 외부 차단 항목을 구분한다.

### P1. Codec·packing 참조 구현

- 기본 QJL-off 4-bit FWHT codec과 dense 비교 oracle을 구현한다.
- byte 단위 pack/unpack, norm correction, 고정 상수, format version을 구현한다.
- 랜덤 입력과 실제 Qwen3 KV snapshot으로 golden fixtures를 생성한다.
- 동일 상수·연산 순서 기준으로 수치 허용오차를 정한다. index 경계 근처의 FP 반올림 차이와 kernel 오류를 구분한다.

완료 조건: byte packing은 bit-exact, 수치 경로는 사전 명세한 오차 내에서 통과한다. 해당 결과를 실기기 완료로 보고하지 않는다.

### P2. 최소 HTP 실행 검증

- 128차원 블록과 실제 prefill/decode shape에서 codec 그래프를 실행한다.
- 필요 시 HTP Op Package를 구현하고 실제 package 로딩부터 검증한다.
- host oracle와 장치의 packed bytes/복원 결과를 비교한다. float 연산 경계에서 허용한 index 차이도 복원 오차로 검증한다.
- 새 토큰 encode와 tile decode의 시간, 복사량, scratch를 측정한다.

완료 조건: 최소 codec의 실기기 HTP 실행 증거와 수치 비교 로그가 있다. CPU fallback 상태로 P3 전체 통합을 완료 처리하지 않는다.

### P3. Qwen3-1.7B 통합

- `k8_v4`로 V-only 경로를 먼저 확인한 뒤 `k4_v4`를 연결한다.
- 공통 Qwen3 attention, split, packed I/O, encodings, runner 상태 관리에 연결한다.
- 짧은 context부터 prefill chunk 경계와 multi-token decode를 검증한다.
- 원본 KV mirror를 제거하고 packed allocation 및 scratch 사용량을 확인한다.

완료 조건: 전체 1.7B 모델이 실기기에서 prefill 및 연속 128-token decode를 수행하고, early EOS를 포함한 처리 규칙이 로그에 남는다. 성능 시험은 생성 길이를 통제하고, 정확도 시험은 EOS를 정상 처리한다. 신규 세션 reset 및 context 경계에서도 cache가 오염되지 않는다.

### P4. 벤치마크·품질 평가

- 아래 9절의 baseline 대비 비교표와 원시 로그를 생성한다.
- 속도 저하, 측정 불가 메모리, 실패한 context도 결과에 포함한다.
- 성능용 비계측 실행과 분석용 profiling 실행을 분리한다.

완료 조건: 기능 실행, 압축 효과, 품질 변화, 속도 변화를 각각 판단할 수 있는 재현 가능한 결과가 있다.

### P5. 다중 모델 확장

- 0.6B 후보/4B/8B의 config·shape·GQA·split 조합을 parameterized test로 검증한다.
- codec 복제 없이 동일 구현과 모델별 설정만으로 연결한다.
- 각 모델의 변환과 실기기 검증은 별도 상태로 기록한다. 1.7B 성공으로 다른 모델의 NPU 실행까지 검증했다고 표시하지 않는다.
- 장치 메모리 부족 시 이를 용량 제한으로 보고하며 레이어·context·backend를 몰래 바꾸지 않는다.

1차 인수 대상은 1.7B의 NPU 실증과 크기 확장 가능한 구조다. 다른 크기의 전체 실기기 인수는 모델별 후속 마일스톤으로 관리한다.

## 9. 벤치마크 요구사항

### 9.1 기존 도구 재사용과 artifact 선택

기존 [`run_geniex_bench_benchmarks.py`](../../src/qai_hub_models/scripts/run_geniex_bench_benchmarks.py)는 모델/장치/plugin/precision을 선택하고 submit/collect를 수행한다. 기본 QAIRT 경로는 `release-assets.yaml`에 등록된 번들을 가져오므로, 소스 코드만 수정하고 실행하면 원래 모델을 측정할 수 있다.

- 실험용 번들 경로 또는 override manifest를 명시적으로 받도록 연결한다. 기존 옵션으로 부족하면 새 옵션을 추가하고 문서화한다.
- 실행 직전에 모델·codec config·context binary·Op Package·runner의 해시를 출력/저장한다.
- baseline과 각 codec 프로파일의 결과 디렉터리를 분리한다.
- 기존 `perf.yaml`/`numerics.yaml` 결과를 덮어쓰지 않는다. 기존 수집기의 `--skip-perf-update` 또는 동등한 격리 경로를 사용한다.
- 기존 Android 배포·실행 코드는 [`test_geniex_bench_android.py`](../../src/qai_hub_models/utils/llm/geniex/device_scripts/geniex_pytest/test_geniex_bench_android.py)를 재사용/확장할 수 있다. 여기서 받는 `geniex-bench` 바이너리 내부도 수정 모델을 지원해야 한다.
- QDC/AWS 인증, 사용 가능 장치, 프로젝트 설정, 사용자 바이너리 실행 허용 여부는 별도 확인한다. Workbench 인증만으로 이 벤치마크 경로도 사용 가능하다고 가정하지 않는다.
- 클라우드 작업 제출·외부 업로드는 명시적으로 선택된 환경과 권한 범위에서만 수행한다. 이 문서는 작업을 실제 제출하는 명령이 아니다.

### 9.2 비교 조건

- 필수 모델: `qwen3_1_7b`.
- 필수 프로파일: `baseline_int8`, `k8_v4`, `k4_v4`.
- context capacity 후보: 512, 1024, 2048, 4096. 실제 지원 graph 조합을 확인한다.
- 각 capacity에서 입력 길이 `P`, 생성 길이 `N`, prefill chunk 크기를 별도로 명시하고 `P + N <= capacity`를 검증한다.
- 1차 성능 시험의 생성 길이는 128 tokens로 고정한다. capacity 값과 실제 사용 길이를 혼동하지 않는다.
- 같은 checkpoint/weight quantization, tokenizer/prompt IDs, thinking mode, sampling 설정, backend/runner, 전력 모드를 유지한다.
- warmup 이후 최소 3회 측정하고 median과 변동 폭을 기록한다. 가능하면 baseline/실험 순서를 교차하고 온도·throttling 및 cold/warm 상태를 기록한다.
- 빌드, 다운로드, 모델 로딩, prefill, decode 시간을 분리한다. TTFT에 포함된 범위를 명시한다.

### 9.3 필수 측정값

| 구분 | 측정값 |
|---|---|
| 속도 | 직접 측정 TTFT, prefill tokens/s, decode tokens/s, ms/token |
| Codec | 새 KV encode 시간, tile decode/융합 op 시간, 호출 수, host↔device 복사량 |
| 메모리 | packed payload, norms/metadata, padding, 실제 KV allocation, scratch/intermediate, host mirror, 전체 peak |
| 수치 | KV 복원 오차, attention/logit 오차, teacher-forced logit KL 또는 동등한 지표 |
| 품질 | 동일 corpus의 perplexity, 고정 프롬프트 응답 평가, context 내 retrieval 시험 |
| 실행 증거 | backend/HTP 정보, package load, graph/op 프로파일, fallback 검사 |

현재 [`GenieXBenchMetrics`](../../src/qai_hub_models/utils/llm/geniex/jobs.py)는 주로 TTFT/prefill/decode 및 token 수를 수집한다. KV allocation이나 codec별 시간은 별도 계측·수집 항목을 추가해야 한다. 측정 API가 제공하지 않는 값은 추정값 또는 미측정으로 표시한다. Android RSS/PSS만으로 HTP/shared memory 전체를 측정했다고 주장하지 않는다.

`perf.yaml`의 TTFT min/max는 수집 코드에서 prompt 길이에 따라 환산될 수 있다. TurboQuant 비교의 주 지표는 동일 길이 입력의 직접 측정값으로 하고, 환산값과 섞지 않는다.

`--run-eval`의 기존 100-prompt 응답 수집 및 후속 Grace2 채점을 재사용할 수 있다. [`numerics.yaml`](../../src/qai_hub_models/models/qwen3_1_7b/numerics.yaml)에 저장된 과거 점수는 새 실험 baseline을 대체하지 않는다. Grader 버전·prompt와 데이터셋을 고정하고, calibration 데이터와 평가 데이터를 분리한다.

실기기 품질 평가가 필요한데 runner가 logits를 제공하지 않으면 teacher-forced logits 수집 기능 또는 별도 평가 runner를 추가한다. PC에서 계산한 perplexity를 실기기 측정값으로 표시하지 않는다.

### 9.4 메모리 산정 검산

메타데이터를 제외한 KV payload의 기본식은 다음과 같다.

```text
payload_bytes = batch * layers * kv_heads * tokens * head_dim * (k_bits + v_bits) / 8
```

블록별 ceil, byte padding, norm, 정렬 및 allocation overhead를 별도로 더한다. 참고 repo의 `compressed_size_bits()` 또는 `memory_stats()`를 실측값처럼 사용하지 않는다. 예를 들어 `CompressedVector`는 원본 norm과 residual norm을 모두 갖고, V에도 norm이 존재하므로 저장 필드 전체를 직접 검산해야 한다.

1.7B, batch=1, 4096 tokens, 양쪽 cache가 int8일 때 payload는 224 MiB다. 양쪽 4-bit는 112 MiB이며, 128차원 블록마다 FP16 norm 하나씩을 K/V에 저장하면 norm이 3.5 MiB 추가된다. 따라서 정렬·scratch 등을 제외한 예시 합계는 115.5 MiB다. 이는 산술 검산값이지 예상 실측값이나 성능 보장이 아니다.

주 비교 기준은 기존 repo의 int8 KV다. fp16 대비 압축률만 제시하여 실제 baseline 대비 이득을 과장하지 않는다.

## 10. 테스트 및 인수 기준

### 10.1 필수 테스트

- rotation/역회전, norm 보존 및 영벡터 처리.
- 모든 bit pattern, byte 경계, tail, alignment의 pack→unpack bit-exact 테스트.
- centroid 경계값, 작은/큰 norm, NaN/Inf 입력 정책, norm dtype 변환.
- GQA mapping, K/V 축 차이, split global layer 번호, codec config hash.
- prefill chunk 처리와 토큰별 append 결과의 일치성. 연산 순서 차이는 명세한 오차로 처리한다.
- cache reset, 세션 간 격리, 최대 context, overflow 거부, 미지원 I/O 거부.
- CPU oracle ↔ ONNX/reference execution ↔ HTP 결과 비교. ORT에 custom op가 없으면 검증용 구현을 제공하거나 해당 단계의 대체 비교 방법을 명시한다.
- TurboQuant 비활성 상태의 기존 모델/export 동작 회귀 테스트.
- Qwen3 모델 크기별 config/shape 검증. `head_dim = hidden_size // num_attention_heads`를 항상 사용하지 않는다.
- 최종 번들에 수정 graph가 실제 포함되었는지 확인하는 artifact smoke test.

### 10.2 완료 상태를 구분하여 보고

| 상태 | 판정 조건 |
|---|---|
| 참조 구현 완료 | codec/packing 수치 테스트 통과. 실기기 완료가 아님 |
| NPU 기능 검증 완료 | 1.7B 전체 생성 루프, codec HTP 실행 증거, packed cache 유지, 회귀 테스트 통과 |
| 압축 효과 입증 | 실제 KV allocation 감소 확인, metadata/scratch/mirror와 전체 peak 변화까지 보고 |
| 품질 평가 완료 | baseline 대비 PPL/응답 평가/오차 결과와 데이터·설정 공개 |
| 성능 개선 입증 | 동일 조건 반복 실측에서 개선 확인. 단순 실행 성공이나 이론 압축률로 대체 불가 |
| 타 모델 검증 완료 | 해당 모델의 변환·실기기 로그가 각각 존재 |

사용자가 품질/속도 허용 임계값을 지정하지 않았으므로, 임의의 수치를 사용자 요구처럼 확정하지 않는다. 수치 구현 일치성 tolerance는 P1에서 명세·고정한다. 제품 채택을 위한 PPL/Grace 하락 및 속도 기준은 별도로 제안하고 합의한다. 기존 scorecard의 허용 하락 폭을 TurboQuant 합격 기준으로 자동 복사하지 않는다.

속도가 느리거나 전체 peak가 증가하더라도 실패 결과를 숨기지 않는다. 기능 검증 완료와 최적화 성공을 분리하여 보고한다.

## 11. 최종 산출물

- 공통 codec/config/cache/ONNX 확장 코드와 모델 크기별 설정.
- ABI 및 dense/FWHT/QJL 정책 차이를 설명한 설계 문서.
- 필요 시 HTP Op Package 소스·converter 설정·빌드 스크립트·runner adapter.
- 원본 baseline과 수정 모델을 각각 재생성하는 명령, 환경 버전 및 artifact manifest.
- CPU 및 실기기 테스트, golden fixtures, 원시 benchmark 로그와 비교표.
- 로컬 Android 실행 절차와, 사용 가능할 경우 별도 device-farm 실행 절차.
- 라이선스/출처 고지. 참고 코드의 재사용·수정 시 원 저작권과 [LICENSE](https://github.com/TheTom/turboquant_plus/blob/ba52ad107d1fdd02bc9be8fd85308226b75c905b/LICENSE), [NOTICE](https://github.com/TheTom/turboquant_plus/blob/ba52ad107d1fdd02bc9be8fd85308226b75c905b/NOTICE)를 보존한다.
- 미지원 모델·context·backend, 측정 불가 항목, 잔여 리스크 목록.

완료 보고에는 변경 파일, 실제 실행한 명령, 통과/실패 테스트, 장치 실행 증거, baseline 대비 결과 및 다음 작업을 포함한다. 미실행 항목을 통과했다고 표시하거나, 사용자의 기존 변경을 덮어쓰거나, 접근 권한 없는 외부 서비스를 사용하지 않는다.

## 12. 구현 착수 시 사용할 요약 지시

> 이 명세에 따라 Qwen3-1.7B부터 TurboQuant KV-cache 압축을 구현한다. `turboquant_plus`의 고정 commit을 참고하되 기본 경로는 QJL-off PolarQuant/FWHT로 하고, 연구용 Python `TurboQuant`/`KVCacheCompressor`를 실사용 기본값으로 오인하지 않는다. 기존 ai-hub-models의 공통 템플릿과 변환·벤치마크 인프라를 재사용한다. 먼저 baseline과 최소 codec의 실기기 HTP 실행 가능성을 검증한 뒤 전체 모델에 통합한다. packed cache ABI, quantization 경계, 실제 allocation과 scratch를 관리하고 CPU/GPU fallback을 숨기지 않는다. 완료는 PC 실험이 아니라 1.7B 실기기 NPU 실행 및 재현 가능한 성능·품질·메모리 보고로 판단한다. 0.6B 후보/4B/8B는 공통 설정으로 확장하며, 사용자가 언급한 0.7B의 정확한 모델 ID는 별도 확인한다.
