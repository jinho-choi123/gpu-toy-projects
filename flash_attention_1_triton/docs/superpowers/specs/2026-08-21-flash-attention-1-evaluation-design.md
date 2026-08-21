# FlashAttention-1 평가 시스템 설계

## 상태

이 설계는 2026-08-21 설계 검토에서 승인되었다. 공개 연산자 interface와 정확성 검증, benchmark, profile, artifact 동작을 정의한다. FlashAttention-1 Triton 알고리즘 자체의 구현이나 세부 설계는 포함하지 않는다.

## 목표

직접 작성한 FlashAttention-1 연산자를 재현 가능하게 평가하는 H100 중심 프로젝트를 만든다. Forward와 backward를 명시적인 same-dtype PyTorch reference와 비교하고 steady-state latency, memory footprint, effective throughput, hardware utilization을 보고한다.

## 목표가 아닌 것

- `@triton.jit` forward/backward kernel 구현 또는 tuning
- Dropout, attention bias, padding mask, ragged input, cross-attention, MQA, GQA
- 32, 64, 128 이외의 head dimension
- FP32 입력, higher-order gradient, forward-mode automatic differentiation
- PyTorch SDPA, `torch.compile`, 다른 최적화 attention provider
- Pytest에서 성능 수치를 합격 조건으로 강제하는 것
- GPU CI 구성
- NVIDIA profiling tool이나 counter 권한을 자동으로 설치·변경하는 것

## 용어와 의사결정 기록

프로젝트 용어는 [`CONTEXT.md`](../../../CONTEXT.md)에 정의한다. 이 설계를 뒷받침하는 주요 의사결정은 [`docs/adr`](../../adr/)에 기록한다.

성능과 관련된 두 개념을 구분한다.

- **Hardware utilization**은 Nsight Compute hardware counter에서 얻는다.
- **Effective throughput**은 attention 문제의 알고리즘상 연산량을 steady-state latency로 나눈 값이다. 실제 hardware FLOP rate로 표현하지 않는다.

## 지원하는 연산자 계약

Package는 다음 callable 하나만 공개한다.

```python
def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
) -> torch.Tensor: ...
```

Interface의 불변 조건은 다음과 같다.

- Q, K, V의 shape은 `[batch, heads, sequence, head_dim]`이다.
- 세 tensor의 shape, dtype, CUDA device가 같다.
- 입력은 contiguous FP16 또는 BF16 tensor다.
- `batch`, `heads`, `sequence`는 양수다.
- `head_dim`은 32, 64, 128 중 하나다.
- Score에는 항상 `1 / sqrt(head_dim)`을 곱하며 caller가 scale을 변경할 수 없다.
- Causal attention에서 query 위치 `i`가 볼 수 있는 key 위치는 정확히 `j <= i`다.
- 반환 tensor의 shape, dtype, device는 입력과 같다.
- 공개 callable은 attention output만 반환한다.
- `backward()`와 `torch.autograd.grad()`를 통한 Q, K, V의 1차 gradient를 지원한다.
- Q, K, V 중 일부만 gradient를 요구하는 경우도 지원한다.

Interface는 device 이동, dtype 변환, 입력 contiguous 변환을 암묵적으로 수행하지 않는다. Non-contiguous `grad_output`은 private backward implementation 안에서 contiguous로 만들 수 있다. Benchmark는 미리 contiguous인 `grad_output`을 사용한다.

잘못된 rank, shape, 크기, head dimension에는 `ValueError`, 잘못된 dtype에는 `TypeError`, CPU 또는 서로 다른 device에는 `RuntimeError`를 발생시킨다. Higher-order derivative는 지원 계약 밖이다.

## 아키텍처

```text
src/flash_attention_1_triton/
├── __init__.py
├── _attention.py
├── _kernels.py
├── _reference.py
├── _evaluation.py
├── _artifacts.py
└── _nsight.py

bench/
└── benchmark.py

profile/
├── profile.py
└── summarize.py
```

의존성은 script에서 평가 module 방향으로, 공개 연산자에서 kernel 방향으로 흐른다. 연산자 module은 benchmark 또는 profile module을 import하지 않는다.

### `_attention.py`

깊은 연산자 module이다. 입력 검증, autograd function, tensor 할당, 고정 scale 계산, Python launcher를 소유한다. 내부 launcher seam은 다음과 같다.

```python
def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]: ...  # output, FP32 logsumexp

def _backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
```

Forward launcher는 `[B, H, N]` shape의 FP32 log-sum-exp state를 생성한다. Autograd는 Q, K, V, output, log-sum-exp, causal flag를 저장한다. 일부 입력의 gradient만 필요해도 backward launcher는 세 gradient를 모두 계산할 수 있으며, autograd가 불필요한 입력에 `None`을 반환한다.

### `_kernels.py`

사용자가 구현하는 영역이다. `_attention.py`가 호출하는 private `@triton.jit` kernel body와 tuning constant만 둔다. 초기 stub은 `NotImplementedError`를 발생시켜 kernel이 없는 프로젝트가 수치적으로 완성된 것처럼 보이지 않게 한다.

### `_reference.py`

Private reference는 다음 same-dtype PyTorch 연산을 명시적으로 사용한다.

```text
scores = Q @ Kᵀ
scores = scores / sqrt(D)
causal이면 scores[j > i] = -inf
probabilities = softmax(scores)
output = probabilities @ V
```

Reference gradient는 PyTorch autograd가 계산한다. SDPA, `torch.compile`, custom kernel을 호출하지 않는다.

### `_evaluation.py`

Case 정의, 결정적 입력 생성, provider 선택, PyTorch CUDA Graph capture, PyTorch CUDA event timing, PyTorch 연산을 이용한 L2 flush, memory 측정을 소유한다. 얇은 benchmark/profile script가 호출하는 작은 내부 interface를 제공한다. Reference와 FlashAttention은 이 seam의 두 private provider adapter이며 공개 provider framework는 만들지 않는다.

### `_artifacts.py`

Schema version이 있는 manifest와 CSV serialization, 결정적 row 순서, 원자적 manifest 갱신, terminal formatting을 소유한다.

### `_nsight.py`

Tool 및 counter 접근 사전 검사, nsys/ncu 실행, native report와 log 보존, tool CSV export, profile 결과 정규화를 담당한다. Tool을 설치하거나 시스템 권한을 변경하지 않는다.

## 정확성 설계

### 결정적 입력

기본 seed는 0이며 CLI에서 변경할 수 있다. 각 case는 고정 integer index를 가지며 case seed는 `base_seed + case_index`다. 따라서 suite 순서나 filter가 달라져도 같은 case의 tensor는 변하지 않는다. Q, K, V, `grad_output`은 case dtype의 표준정규분포에서 생성한다.

### Pairwise case 표

Correctness suite는 Cartesian product 대신 다음 고정 case를 사용한다.

| Case | B | H | N | D | Dtype | Causal |
|---|---:|---:|---:|---:|---|---|
| `boundary-001` | 1 | 1 | 1 | 32 | FP16 | 아니요 |
| `boundary-017` | 1 | 3 | 17 | 64 | BF16 | 예 |
| `boundary-063` | 2 | 8 | 63 | 128 | FP16 | 예 |
| `boundary-064` | 1 | 16 | 64 | 32 | BF16 | 아니요 |
| `boundary-065` | 2 | 3 | 65 | 64 | FP16 | 아니요 |
| `boundary-127` | 1 | 8 | 127 | 128 | BF16 | 예 |
| `boundary-128` | 2 | 16 | 128 | 32 | FP16 | 예 |
| `boundary-129` | 1 | 3 | 129 | 64 | BF16 | 아니요 |
| `boundary-257` | 2 | 8 | 257 | 128 | FP16 | 아니요 |
| `boundary-1024` | 1 | 16 | 1024 | 64 | BF16 | 예 |

작은 지원 shape에서 다음 scenario 세 개를 추가 실행한다.

- Q와 K가 모두 0인 경우
- Q와 K의 모든 행이 동일한 non-zero 값이라 score tie가 생기는 경우
- 표준정규분포 Q와 K에 8을 곱해 softmax가 포화되는 경우

Causal property test는 미래 위치의 K와 V를 변경하고 그보다 앞선 output 위치가 tolerance 안에서 변하지 않는지 검사한다.

### 수치 일치

`torch.testing.assert_close`로 output, dQ, dK, dV를 각각 비교한다.

| 결과 | FP16 atol / rtol | BF16 atol / rtol |
|---|---|---|
| Output | `1e-2 / 1e-2` | `2e-2 / 2e-2` |
| dQ, dK, dV | `2e-2 / 2e-2` | `5e-2 / 5e-2` |

NaN과 infinity는 허용하지 않는다. 실패 시 최대 absolute error와 relative error를 출력한다. 모든 입력이 gradient를 요구하는 경우와 일부만 요구하는 경우를 `backward()` 및 `torch.autograd.grad()` 양쪽에서 검사한다.

## 테스트 구성

```text
tests/
├── conftest.py
├── test_validation.py
├── test_reference.py
├── test_correctness.py
├── test_autograd.py
├── test_cuda_graph.py
├── test_cases.py
├── test_artifacts.py
├── test_cli.py
└── test_nsight_summary.py
```

- Pure test는 case 생성, seed 안정성, validation, schema, row 순서, CLI parsing, exit code, tool-export CSV parsing, 실패 결과 serialization을 검사한다.
- CUDA가 없으면 CUDA correctness와 CUDA Graph test를 skip한다.
- CUDA가 있는 환경에서는 kernel 미구현을 xfail하지 않고 실패로 처리한다.
- CUDA Graph integration test는 forward replay, pure-backward replay, numerical agreement, static output buffer 갱신을 검사한다.
- Export된 nsys/ncu CSV fixture로 GPU 없이 정규화 로직을 검사한다.
- 실제 profiler smoke test는 `profile` pytest marker를 사용하며 기본 pytest 실행에서는 제외한다.
- Pytest는 latency, speedup, memory reduction, utilization threshold를 강제하지 않는다.

이번 범위에서 CI workflow는 만들지 않는다.

## Benchmark 설계

### CLI

```bash
uv run python bench/benchmark.py
uv run python bench/benchmark.py --suite full --metric all
```

기본 suite는 `smoke`, 기본 metric 선택은 `all`이다. Filter는 `--device`, `--case-id`, `--dtype`, `--causal`, `--provider`, `--phase`이며 기본 device index는 0이다. Smoke configuration은 `B=1, H=8, N=128, D=64, BF16, non-causal`이며 두 provider와 두 phase를 모두 측정한다.

### Full matrix

Full matrix는 다음 세 sweep의 중복을 제거한 합집합이다.

1. Sequence sweep: `B=1, H=16, D=64`, `N ∈ {128,256,512,1024,2048,4096,8192}`
2. Head-dimension sweep: `B=1, H=16, N=2048`, `D ∈ {32,64,128}`
3. Batch/head sweep: `N=1024, D=64`, `(B,H) ∈ {(1,1),(1,8),(1,16),(4,16),(8,32)}`

각 unique shape는 두 dtype과 두 causal mode로 실행되어 총 52개 configuration이 된다. CUDA를 초기화하지 않은 coordinator가 configuration마다 새 worker process를 spawn한다. Worker는 종료하기 전에 두 provider와 두 phase를 모두 측정한다.

### 수치 사전 검사

각 worker는 FlashAttention provider를 측정하기 전에 output과 gradient를 reference tolerance와 비교한다. 잘못된 결과는 유효한 고속 결과로 기록하지 않는다.

### CUDA Graph latency

Graph orchestration에는 `torch.cuda.CUDAGraph`와 `torch.cuda.graph`만 사용한다. Triton benchmark helper는 사용하지 않는다.

각 provider와 phase에 대해 worker는 다음을 수행한다.

1. 결정적인 static Q, K, V, `grad_output`을 초기화한다.
2. Capture stream에서 compilation을 포함한 warm-up call을 10회 실행한다.
3. 연산자 호출 한 번을 graph로 capture한다.
4. 각 replay 전에 256 MiB PyTorch-managed CUDA buffer 연산으로 L2를 flush한다.
5. Flush 다음에 start CUDA event를 기록하고 graph를 한 번 replay한 뒤 end event를 기록한다.
6. Event pair를 100회 반복하고 synchronize한 뒤 sample을 읽는다.

Flush는 동일 stream에서 start event보다 먼저 실행되므로 elapsed time에서 제외된다. p20, median, p80 GPU duration을 기록하며 CPU graph-launch 시간은 포함하지 않는다.

Pure-backward graph는 미리 계산한 forward output, saved state, contiguous `grad_output`을 사용해 `torch.autograd.grad(..., retain_graph=True)`를 capture한다. Gradient 누적과 초기화는 제외한다. 파생 `forward_backward` row는 두 phase median의 합을 사용하고 `derivation=sum_of_phase_medians`로 표시한다.

### Effective throughput

Shape `(B,H,N,D)`에 대한 관례적 알고리즘 연산량은 다음과 같다.

```text
forward_flops = 4 * B * H * N^2 * D / (causal이면 2, 아니면 1)
backward_flops = 2.5 * forward_flops
forward_backward_flops = 3.5 * forward_flops
effective_tflops = phase_flops / median_seconds / 1e12
```

이 값은 `effective_tflops`라는 이름의 파생 benchmark metric이며 Nsight hardware counter가 아니다.

### Memory

입력은 baseline 기록 전에 할당한다. 다음 네 항목에 대해 absolute 및 baseline-relative allocated/reserved byte를 보고한다.

1. 일반 forward peak
2. 일반 forward+backward peak
3. forward CUDA Graph private-pool footprint
4. forward+backward CUDA Graph private-pool footprint

일반 peak는 PyTorch peak-memory 통계와 synchronize를 이용한다. Graph pool footprint는 capture 전후 resident-memory 차이다. 다음 독립 측정 전에 graph를 파괴하고 측정 구간 밖에서 PyTorch cache를 비운다.

### Worker 격리와 실패

OOM과 runtime error는 측정 단위로 잡는다. 실패 row는 `status/error`를 기록하고 수치 cell을 비운다. Worker가 계속 사용할 수 있으면 다른 case 측정을 계속한다. Worker가 비정상 종료하면 해당 configuration의 남은 결과를 실패로 기록한다. 실패 case가 하나라도 있으면 artifact를 모두 기록한 후 coordinator는 non-zero로 종료한다.

## Profiling 설계

### CLI와 case

```bash
uv run python profile/profile.py --suite representative --tool all
uv run python profile/summarize.py results/<run-id>
```

Profile CLI는 `--device`, `--case-id`, `--dtype`, `--causal`, `--provider`, `--phase`를 지원한다. 기본 suite는 smoke profile이며 `--suite representative`는 다음 case를 선택한다.

| Case | B | H | N | D | Dtype | Causal |
|---|---:|---:|---:|---:|---|---|
| `profile-small` | 1 | 16 | 512 | 64 | BF16 | 아니요 |
| `profile-long` | 1 | 16 | 4096 | 64 | BF16 | 아니요 |
| `profile-causal-wide` | 1 | 16 | 4096 | 128 | BF16 | 예 |

### Capture range

NVTX domain은 `flash_attention_1_triton`이며 range 이름은 다음과 같다.

```text
reference/forward/<case-id>
reference/backward/<case-id>
flash/forward/<case-id>
flash/backward/<case-id>
```

Compilation, warm-up, L2 flush는 range 밖에서 실행한다. Range 안에는 PyTorch CUDA Graph replay 한 번만 포함한다.

### Nsight Systems

Profile case마다 `.nsys-rep` 하나를 만들며 네 provider/phase range를 모두 포함한다. Summary 추출에는 NVTX GPU projection, CUDA kernel, CUDA API report를 사용한다.

### Nsight Compute

Case/provider/phase마다 `.ncu-rep` 하나를 만들어 representative suite에서 총 12개 report를 생성한다. Graph profiling은 node mode를 사용하며 다음 section을 수집한다.

- `LaunchStats`
- `Occupancy`
- `SpeedOfLight`
- `ComputeWorkloadAnalysis`
- `MemoryWorkloadAnalysis`

Summary는 raw kernel metric을 유지하면서 다음 항목을 강조한다.

- `dram__bytes.sum.per_second`
- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`
- `lts__t_sector_hit_rate.pct`
- occupancy
- `gpu__time_duration.sum`

절대 actual Tensor TFLOP/s와 비용이 큰 hierarchical Tensor Roofline 수집은 version 1에서 제외한다.

### 집계

Kernel-scope row는 실측값을 그대로 보존한다. 여러 kernel로 구성된 provider에는 다음 operator-scope 파생 row를 추가한다.

- DRAM GB/s는 전체 실측 DRAM byte 합을 전체 kernel duration 합으로 나눈다.
- Peak 대비 throughput과 utilization은 duration-weighted 값으로 계산한다.
- 의미 있게 집계할 수 없는 값은 kernel scope로만 남긴다.
- 가중치 없는 percentage 평균은 만들지 않는다.

Native `.nsys-rep`와 `.ncu-rep`가 source of truth다. 기존 report가 있으면 GPU 없이 summary를 다시 만들 수 있다.

## Artifact 설계

```text
results/<UTC-timestamp>-<git-sha>/
├── manifest.json
├── benchmark.csv
├── profile_summary.csv
├── profiles/
│   ├── nsys/
│   └── ncu/
└── logs/
```

`results/`는 gitignore 대상이다. 기존 run을 덮어쓰지 않으며 `--output-dir`로 root를 바꿀 수 있다. 작업 시작 전에 `status=running` manifest를 원자적으로 기록하고 종료할 때 `completed` 또는 `failed`로 원자적으로 교체한다.

Benchmark run은 `benchmark.csv`만 만들고 profile run은 `profile_summary.csv`와 `profiles/`만 만든다. 위 tree는 두 실행 종류에서 생길 수 있는 artifact의 합집합이며, 요청하지 않은 종류의 빈 결과 파일은 만들지 않는다.

### Manifest schema

Schema version 1은 다음 정보를 기록한다.

- Run ID, status, UTC timestamp, CLI argument, seed
- Git commit SHA와 dirty boolean. Diff 자체는 저장하지 않는다.
- Python, PyTorch, Triton, CUDA, driver, nsys, ncu version
- 선택 GPU index, 이름, UUID, compute capability
- CUDA Graph, L2 flush, tolerance, metric 설정

전체 환경이나 임의의 환경변수는 기록하지 않는다.

### Benchmark CSV

한 row는 provider, phase, configuration 하나를 나타낸다. Column은 다음과 같다.

```text
schema_version,run_id,case_id,provider,phase,
batch,heads,sequence,head_dim,dtype,causal,
latency_p20_ms,latency_median_ms,latency_p80_ms,
effective_tflops,
peak_allocated_bytes,peak_reserved_bytes,
peak_allocated_delta_bytes,peak_reserved_delta_bytes,
graph_pool_allocated_bytes,graph_pool_reserved_bytes,
speedup,memory_reduction_ratio,derivation,status,error
```

Reference speedup은 1.0이다. FlashAttention speedup은 reference median을 FlashAttention median으로 나눈 값이다. 비교 가능한 reference와 FlashAttention row가 모두 성공한 경우에만 memory reduction을 기록한다.

`phase=forward` row에는 일반 forward peak와 forward graph-pool footprint를 기록한다. `phase=forward_backward` row에는 일반 training peak와 forward+backward graph-pool footprint를 기록하며 `latency_median_ms`는 두 phase median의 합이다. 이 파생 row의 p20/p80은 비워 두고 `derivation=sum_of_phase_medians`를 기록한다. Pure `phase=backward` row의 memory column은 비워 둔다.

### Profile summary CSV

Long-form column은 다음과 같다.

```text
schema_version,run_id,case_id,tool,source_report,
provider,phase,batch,heads,sequence,head_dim,dtype,causal,
scope,kernel_name,kernel_index,
metric_name,metric_unit,metric_value,derivation,status,error
```

파일에는 base unit을 저장하고 terminal 출력만 읽기 좋은 단위로 변환할 수 있다. Row는 case/provider/phase/metric 순으로 결정적인 순서를 갖는다.

## CLI와 실패 동작

모든 script는 `argparse`와 표준 라이브러리를 사용한다. Exit code는 다음과 같다.

- 0: 요청한 작업이 모두 성공
- 1: 하나 이상의 측정 또는 profile 작업 실패
- 2: 잘못된 CLI 입력 또는 충족하지 못한 prerequisite

GPU가 없는 것은 pytest CUDA test에서만 skip 조건이다. Benchmark와 profile은 CUDA를 요구하며 오해를 낳는 결과를 만들기 전에 실패한다. Tool 미설치, counter 권한 부족, 호환되지 않는 실행은 사전 검사에서 실패한다. 실행 명령, stdout, stderr, partial native report는 보존하며 tool을 설치하거나 권한을 자동으로 변경하지 않는다.

현재 환경에서는 `ERR_NVGPUCTRPERM`이 발생하므로 profiling 완료 조건을 검증하려면 관리자가 performance-counter 접근을 활성화해야 한다.

H100이 아닌 GPU에서도 GPU 모델 경고는 출력하지 않는다. Manifest에는 device 정보를 기록하며 kernel이 지원하지 않는 architecture는 일반 runtime error로 실패한다.

## Dependency

Package는 `torch==2.11.0`, `triton==3.6.0`을 선언한다. Workspace가 개발용 pytest를 제공한다. CLI, JSON, CSV, process 관리, report 정규화에는 Python 표준 라이브러리를 사용한다. NumPy, pandas, 별도 benchmark package는 추가하지 않는다. nsys와 ncu는 외부 executable prerequisite다.

## 완료 조건

사용자가 Triton kernel을 구현한 후 다음 조건을 모두 만족해야 한다.

1. CUDA 환경에서 `profile` marker를 제외한 기본 pytest가 통과한다.
2. CPU-only 환경에서는 pure test가 실행되고 CUDA test가 skip된다.
3. 고정된 모든 output 및 gradient case가 dtype별 tolerance를 만족한다.
4. Forward와 pure-backward PyTorch CUDA Graph capture/replay test가 통과한다.
5. Smoke/full benchmark가 최종 manifest와 schema-valid CSV를 만든다.
6. Benchmark 실패가 artifact에 남고 최종 non-zero exit로 보고된다.
7. nsys가 예상된 모든 NVTX range를 포함한 native report를 만든다.
8. 권한이 활성화된 후 ncu가 native report와 필요한 utilization metric을 만든다.
9. GPU 없이 native report에서 profile summary를 다시 만들 수 있다.
10. README에 setup, 명령, metric 의미, 지원 case, 제약을 설명한다.

`_kernels.py`가 여전히 `NotImplementedError`를 발생시키는 상태에서 주변 scaffolding test만 통과하는 것은 완료로 인정하지 않는다.
