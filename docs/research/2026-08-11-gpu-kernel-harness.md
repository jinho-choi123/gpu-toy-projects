# GPU kernel repository harness 조사

- 조사일: 2026-08-11
- 대상: Linux의 CPython 3.12–3.13, PyTorch, Triton, NVIDIA CuTe DSL 기반 GPU kernel 모노레포
- 결정 반영: 공통 CUDA 12.9.1, PyTorch cu129, CuTe DSL 4.6.1
- 방법: 레포의 현재 파일을 먼저 확인한 뒤, 프로젝트 소유자가 제공하는 공식 문서·공식 소스·명세만 사용했다.
- 범위: 이 문서는 후보와 적용 순서를 제안할 뿐, 설정 파일이나 구현을 변경하지 않는다.

## 결론

1. **P0 CuTe DSL 호환성 결정은 공통 CUDA 12.9.1 environment로 확정됐다.** 공식 quick start는 CUDA Toolkit 12.9 또는 13.1을 명시하고 CUDA 12.9에는 driver 575.51.03 이상을 요구한다. 현재 H100 host driver 580.178.04는 이 조건을 충족한다. [CuTe DSL Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
2. 사용자는 Linux의 CPython 3.12–3.13에서 `nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04`, `torch==2.11.0+cu129`, `nvidia-cutlass-dsl==4.6.1`의 공통 environment를 채택했다. Triton과 CuTe는 같은 toolkit을 사용하되 compile/runtime smoke와 test selection은 독립적으로 유지한다. Windows와 macOS는 이 contract에서 지원하지 않는다. CuTe DSL은 공개 beta이므로 버전 고정이 특히 중요하다. [CuTe DSL overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
3. commit 때는 수 초 안에 끝나는 file hygiene와 Ruff만 실행하고, type check·CPU test는 pre-push/CI, GPU correctness·sanitizer·benchmark는 GPU CI로 분리한다. `pre-commit`은 `pre-commit`, `pre-push`, `commit-msg`, `manual` stage를 구분할 수 있다. [pre-commit stages](https://pre-commit.com/#confining-hooks-to-run-at-certain-stages)
4. 테스트는 `CPU/reference → Triton interpreter → GPU compile smoke → GPU runtime smoke → numerical correctness → sanitizer → benchmark`의 층으로 나눈다. Triton interpreter는 CPU에서 실행되지만 실제 GPU compilation을 우회하고 기능 제한도 있으므로 GPU compile/runtime test를 대신할 수 없다. [Triton debugging/interpreter](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
5. 성능 gate는 바로 켜지 않는다. 먼저 고정된 GPU/driver/toolchain에서 warm-up, 반복 측정, quantile, 환경 metadata를 수집해 자연 변동폭을 정한다. CUDA kernel launch는 비동기이므로 일반 CPU timer만 쓰면 잘못 측정하기 쉽다. Triton의 GPU-aware benchmark API 또는 CUDA event를 사용해야 한다. [Triton `do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html), [CUDA timing guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#timing)

## 조사 시점의 레포 상태

| 항목 | 확인된 상태 | 의미 |
|---|---|---|
| Packaging | root `pyproject.toml`이 `uv` workspace이고 `flash_attention_triton`이 member이며 `uv.lock`이 있음 | workspace 전체가 한 lockfile을 공유하는 방향은 적절함 |
| Python/GPU stack | Linux의 CPython 3.12–3.13, Torch 2.11.0+cu129, Triton 3.6.0 lock | 공통 CUDA 12.9 기준 조합으로 기록 |
| CuTe DSL | `nvidia-cutlass-dsl==4.6.1` dependency/lock과 `scripts/verify_gpu_stack.py` 추가 | 공통 CUDA 12.9 environment에서 명시적 CuTe kernel compile/launch 검증 |
| Dev tools | `pre-commit`, `pytest`, `ruff`, `ty`가 dev dependency에 있음 | 도구 설치는 시작됐지만 정책과 실행 진입점이 필요함 |
| Container | `nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04` tag로 변경 | devcontainer rebuild 뒤 toolkit 12.9.1이 적용됨; digest는 아직 고정되지 않음 |
| Tests/CI | pytest test와 GitHub Actions workflow는 없고 수동 H100 smoke script가 있음 | 수동 GPU preflight와 향후 CPU/GPU CI 비용 경계를 구분함 |
| Documentation | root와 member README가 비어 있고 `docs/` 관례가 없음 | 이 문서는 기본 경로 `docs/research/`에 저장함 |

`uv` workspace는 member별 `pyproject.toml`을 두면서 단일 lockfile을 공유한다. `uv.lock`은 exact resolved version을 담으므로 version control에 넣는 것이 공식 권장사항이다. [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/), [uv project layout/lockfile](https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile)

## P0: CuTe DSL 환경 결정

| 선택지 | 장점 | 위험/비용 | 판정 |
|---|---|---|---|
| CUDA 12.8.1 컨테이너에 최신 CuTe DSL 추가 | 이미지 하나 | 공식 지원 조합이 아님 | **채택하지 않음** |
| Triton CUDA 12.8 lane + CuTe CUDA 12.9 lane 분리 | 회귀 격리 | 이미지와 GPU CI job이 하나 늘어남 | 검토했으나 채택하지 않음 |
| 공통 CUDA 12.9.1로 승격 | Triton과 CuTe가 같은 toolkit contract 사용 | Torch cu129/Triton/NVSHMEM 전체 재검증 필요 | **채택** |
| CuTe CUDA 13.1 lane | 최신 toolkit 기능 | 별도 driver/toolchain compatibility와 비용 | Blackwell/13.x 기능이 필요할 때 추후 |

CuTe wheel은 kernel 생성에 필요한 요소를 포함하지만, 공식 quick start가 요구하는 OS, Python, toolkit/driver 조합은 여전히 지켜야 한다. 현재 문서는 Linux x86_64/aarch64, Python 3.10–3.14, CUDA Toolkit 12.9 또는 13.1을 명시한다. [CuTe DSL 4.4 Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)

## Harness 후보: packaging, 정적 검사, 로컬 hook

우선순위는 다음 뜻이다.

- **필수**: 이 harness branch에서 기본 경로와 CI contract를 만드는 것이 좋음
- **권장**: 기본 harness가 자리 잡은 직후 추가
- **추후**: 데이터나 운영 필요가 생긴 뒤 gate

| 도구/장치 | 왜 필요한가 | 로컬 또는 CI | 우선순위 | 주의점과 공식 근거 |
|---|---|---|---|---|
| `uv` workspace + committed `uv.lock` | Python, Torch, Triton, CUDA Python wheel을 같은 해석 결과로 재현 | 로컬 + 모든 CI | **필수** | `uv sync`는 기본적으로 exact sync이고 workspace는 lock 하나를 공유한다. [lock/sync](https://docs.astral.sh/uv/concepts/projects/sync/), [workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) |
| `uv lock --check`, `uv sync --locked` | PR이 metadata와 lockfile을 어긋나게 만들지 못하게 함 | CI, 로컬 진단 | **필수** | `--locked`는 lock이 낡았으면 변경하지 않고 실패한다. `--frozen`은 최신성 확인도 생략하므로 최종 CI 검증에는 `--locked`가 맞다. [uv lock checking](https://docs.astral.sh/uv/concepts/projects/sync/#checking-the-lockfile) |
| dependency groups (`dev`, 향후 `test`, `bench`) | CPU CI가 benchmark 전용 도구를 불필요하게 설치하지 않게 함 | 로컬 + CI | **권장** | `uv`는 PEP 735 dependency group을 지원하며 group별 sync가 가능하다. CuTe DSL은 레포의 주 runtime이므로 공통 CUDA 12.9 environment의 직접 dependency로 고정한다. [uv development dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/#development-dependencies) |
| `pre-commit` | 모든 개발자와 CI가 같은 빠른 검사를 같은 revision으로 실행 | commit + CI `--all-files` | **필수** | hook `rev`를 tag/SHA로 고정하고 CI에서도 전 파일을 재실행한다. GPU test나 network audit은 commit stage에 두지 않는다. [pre-commit](https://pre-commit.com/), [supported stages](https://pre-commit.com/#confining-hooks-to-run-at-certain-stages) |
| `pre-commit-hooks` file hygiene | merge marker, 잘못된 YAML/TOML, trailing whitespace, 큰 binary, private key를 일찍 차단 | commit + CI | **필수** | 최소 후보: `check-merge-conflict`, `check-yaml`, `check-toml`, `end-of-file-fixer`, `trailing-whitespace`, `check-added-large-files`, `detect-private-key`. GPU dump인 PTX/CUBIN과 benchmark data를 실수로 commit하는 것도 큰 파일 hook으로 보조한다. [official hooks](https://github.com/pre-commit/pre-commit-hooks) |
| Ruff linter | import, 오류 패턴, pyupgrade 계열을 빠르게 통합 검사 | commit + CI | **필수** | `ruff check --fix`는 로컬 hook, CI는 `ruff check`처럼 수정 없이 실패시키는 구성이 명확하다. [Ruff linter](https://docs.astral.sh/ruff/linter/) |
| Ruff formatter | Python formatting을 단일 도구로 고정 | commit + CI `--check` | **필수** | formatter는 import를 정렬하지 않으므로 lint의 `I` 규칙을 먼저 적용한다. formatter와 충돌하는 lint 규칙은 공식 목록을 따라 피한다. [Ruff formatter](https://docs.astral.sh/ruff/formatter/) |
| `ty check` | host Python, launcher, reference implementation의 type 오류 탐지 | pre-push + CPU CI | **권장** | repo에 이미 pin되어 있다. DSL decorator/constexpr 코드는 일반 Python과 의미가 달라 false positive가 날 수 있으므로 rule-specific suppression과 directory override를 사용하고 blanket exclude는 피한다. [ty check](https://docs.astral.sh/ty/type-checking/), [ty suppression](https://docs.astral.sh/ty/suppression/) |
| CuTe용 `mypy==1.19.1` compatibility probe | NVIDIA가 현재 CuTe 개발 dependency로 이 version을 명시 | 공통 CUDA 12.9 GPU CI | **권장** | `ty`와 mypy를 전 repo에 중복 gate하지 않는다. 작은 CuTe package에서 NVIDIA 권장 mypy와 현재 ty 중 어느 쪽이 실제 annotation을 더 잘 처리하는지 먼저 비교한다. [CuTe recommended dependencies](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html#recommended-dependencies) |
| ShellCheck | `.devcontainer/*.sh`와 이후 CI helper의 quoting/error handling 검사 | commit 또는 CPU CI | **권장** | shell formatter가 아니라 정적 분석기다. false positive는 code별, 근거 있는 suppression만 둔다. [ShellCheck official source](https://github.com/koalaman/shellcheck) |
| Hadolint | CUDA devcontainer의 Dockerfile syntax와 common build 실수 검사 | CPU CI, 선택적으로 commit | **권장** | 현 Dockerfile은 installer와 apt/toolchain 특수성이 많아 무조건 모든 rule을 켜기보다 failure threshold와 좁은 예외를 문서화한다. Hadolint는 Dockerfile AST와 RUN의 ShellCheck를 결합한다. [Hadolint](https://github.com/hadolint/hadolint) |
| `actionlint` | GitHub Actions YAML expression, job/step 구조 오류를 실행 전에 발견 | workflow 변경 시 commit + CPU CI | **권장** | workflow를 추가할 때 활성화한다. [actionlint official source](https://github.com/rhysd/actionlint) |

### 권장 stage 경계

| Stage | 포함 | 제외 |
|---|---|---|
| `pre-commit` | file hygiene, Ruff lint fix, Ruff format | type check, test, package audit, GPU 접근 |
| `pre-push` | `ty check`, 빠른 CPU unit test(선택) | GPU correctness, sanitizer, benchmark |
| CPU CI | lock check, pre-commit all-files, type check, CPU/reference test, package build smoke | CUDA runtime 필요 test |
| GPU PR CI | compile/runtime smoke, 작은 correctness matrix | full sanitizer, 긴 benchmark sweep |
| nightly/manual GPU CI | full correctness, sanitizer, benchmark, profiling artifact | commit latency에 영향을 주는 작업 없음 |

## Harness 후보: pytest와 test taxonomy

| 도구/장치 | 왜 필요한가 | 로컬 또는 CI | 우선순위 | 주의점과 공식 근거 |
|---|---|---|---|---|
| `pytest` + 외부 `tests/` layout | kernel launcher/reference/test data를 package와 분리하고 installed package를 검사 | 로컬 + CI | **필수** | pytest는 `testpaths`와 표준 discovery를 지원하며 새 project에는 `importlib` import mode를 권장한다. [pytest good practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html) |
| 등록 marker + `strict_markers` | CPU/GPU/비용/목적을 명시하고 typo로 test가 누락되지 않게 함 | 로컬 + CI | **필수** | 제안 marker: `cpu`, `gpu`, `triton_interpreter`, `compile`, `smoke`, `correctness`, `slow`, `sanitizer`, `benchmark`, 필요 시 `sm80`/`sm90`/`sm100`. pytest는 `-m` selection과 unknown marker error를 지원한다. [pytest custom markers](https://docs.pytest.org/en/stable/example/markers.html) |
| `pytest-timeout` | deadlock/잘못된 synchronization으로 CI가 무기한 점유되는 것을 방지 | CPU + GPU CI | **필수** | per-test timeout과 CI job-level timeout을 함께 둔다. `thread` 방식은 가장 확실하지만 process를 종료해 fixture teardown/JUnit XML이 끝나지 않을 수 있고, `signal`은 code의 `SIGALRM`과 충돌할 수 있다. [pytest-timeout project docs](https://pypi.org/project/pytest-timeout/) |
| deterministic seed fixture | 실패 input을 재현하고 randomized shape test를 안정화 | 로컬 + correctness CI | **필수** | Python/NumPy/Torch seed를 기록하되 release/platform/CPU-GPU 간 완전 재현은 보장되지 않는다. deterministic algorithm은 느릴 수 있으므로 correctness와 benchmark mode를 분리한다. [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness) |
| `pytest.parametrize` shape/dtype/layout matrix | GPU kernel의 boundary condition을 표로 보존 | 로컬 GPU + GPU CI | **필수** | 최소 shape는 1, 작은 수, tile 경계, power-of-two ±1, 큰 수를 포함하고 dtype, contiguous/non-contiguous, stride, causal/bias 등 kernel contract를 축으로 둔다. pytest는 parameterized case를 개별 node로 수집한다. [pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html) |
| Hypothesis property tests | 수작업 matrix가 놓치는 shape/stride 조합을 축소 가능한 failing example로 탐색 | CPU reference 우선, 제한된 GPU nightly | **추후** | GPU example 수를 작게 제한하고 deadline/seed/profile을 고정한다. 먼저 명시적 edge matrix를 만든 뒤 추가한다. [Hypothesis documentation](https://hypothesis.readthedocs.io/en/latest/) |

### 제안 marker contract

```text
cpu                 GPU 없이 항상 실행 가능
triton_interpreter  TRITON_INTERPRET=1에서 가능한 제한된 kernel semantic test
gpu                 실제 CUDA device/driver가 필요
compile             대표 specialization의 compile 성공 여부
smoke               최소 shape 1~2개, PR마다 빠르게
correctness         reference와 dtype별 tolerance 비교
slow                넓은 shape/dtype matrix
sanitizer           Compute Sanitizer 아래에서 실행할 작은 deterministic subset
benchmark           고정 runner에서만 실행, correctness suite와 별도
sm80/sm90/sm100     특정 architecture가 실제 contract일 때만 사용
```

예상 진입점은 다음처럼 서로 겹치지 않아야 한다.

```bash
# 표준 CPU CI
uv run pytest -m "not gpu and not sanitizer and not benchmark"

# Triton interpreter 전용; 실제 compile test가 아님
TRITON_INTERPRET=1 uv run pytest -m triton_interpreter

# 빠른 GPU PR lane
uv run pytest -m "gpu and (compile or smoke or correctness) and not slow"

# nightly correctness
uv run pytest -m "gpu and correctness"
```

GPU가 없을 때 `gpu` test가 조용히 pass하는 것보다 명시적으로 deselect되거나, GPU job에서는 CUDA unavailable을 즉시 실패시키는 편이 좋다. 일반 CPU job은 marker expression으로 GPU test를 수집 대상에서 빼고, GPU job 시작에는 device name, capability, driver를 확인하는 preflight를 둔다. Pytest는 `skipif`와 marker selection을 공식 지원한다. [pytest skip/xfail](https://docs.pytest.org/en/stable/how-to/skipping.html), [pytest invocation](https://docs.pytest.org/en/stable/how-to/usage.html#specifying-which-tests-to-run)

## Triton/CuTe compile 및 runtime smoke

추적된 수동 진입점은 `uv run --locked python scripts/verify_gpu_stack.py`이다. 이 command는 CUDA와 H100 capability `(9, 0)`를 먼저 확인하고 version metadata를 출력한 뒤, Triton add-one kernel의 결과를 assert하고 명시적 CuTe `@cute.kernel`을 `@cute.jit` launcher로 실행한다.

| 대상 | test 내용 | 위치 | 우선순위 | 주의점과 공식 근거 |
|---|---|---|---|---|
| Triton import/package smoke | package import, kernel symbol import, reference function CPU 실행 | CPU CI | **필수** | Python packaging 파손을 GPU queue 전에 잡는다. 이것은 kernel compile을 증명하지 않는다. |
| Triton interpreter smoke | 작은 supported dtype/operation으로 load/compute/store semantic 확인 | CPU CI | **권장** | `TRITON_INTERPRET=1`은 compilation을 우회해 NumPy equivalent로 순차 simulation한다. BF16과 indirect memory access 등 공식 제한이 있으므로 지원되는 test만 별도 marker로 둔다. [Triton interpreter](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html#using-the-interpreter) |
| Triton compile smoke | 대표 dtype/shape/constexpr specialization마다 JIT warmup/compile 성공 확인 | GPU PR CI | **필수** | `@triton.jit` 함수는 호출 때 compile되고 GPU에서 실행된다. compile-only 경로를 쓸 때도 current target/driver가 필요하다고 가정하고 GPU lane에 둔다. cache hit가 test를 가리지 않도록 nightly cold compile에는 `TRITON_ALWAYS_COMPILE=1`을 사용한다. [triton.jit](https://triton-lang.org/main/python-api/generated/triton.jit.html), [Triton official source/debug knobs](https://github.com/triton-lang/triton#tips-for-building) |
| Triton runtime smoke | 최소 input을 실제 launch하고 synchronize 후 output/basic invariant 확인 | GPU PR CI | **필수** | compile 성공만으로 launch/resource/synchronization 오류는 잡히지 않는다. compile과 runtime 결과를 구별해 report한다. |
| CuTe `cute.compile` smoke | 공통 CUDA 12.9 environment에서 대표 JIT function을 명시적으로 compile해 executor 생성 | GPU PR CI | **필수** | `cute.compile`은 cache를 우회해 매번 compile하고 JIT executor를 반환하며 module/function pointer까지 load한다. cold compile smoke에 적합하지만 비싸므로 작은 specialization만 둔다. [CuTe JIT caching/`cute.compile`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html) |
| CuTe runtime smoke | compile된 executor로 최소 kernel launch, synchronize, invariant 확인 | CuTe GPU PR CI | **필수(호환성 결정 후)** | compile과 argument conversion/launch를 모두 검사한다. `CUTE_DSL_DEBUG=1`은 diagnostics를 늘리지만 generated code와 cache key가 달라지므로 normal smoke와 benchmark를 대체하지 않는다. [CuTe debugging](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html) |
| Architecture smoke matrix | 실제 지원 대상 SM별로 최소 compile/runtime case 실행 | nightly 또는 release CI | **권장** | H100 `sm90` 하나만으로 `sm80`/`sm100` 지원을 주장하지 않는다. 특정 architecture instruction을 쓰는 kernel은 marker와 runner label을 함께 요구한다. |

## Numerical correctness/reference comparison

NVIDIA의 CUDA best-practices guide는 known-good reference output 비교와 unit test를 correctness의 핵심으로 권장하고, floating-point parallelization은 연산 순서와 FMA 때문에 bitwise equality가 성립하지 않을 수 있다고 설명한다. [CUDA reference comparison and unit testing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#verification), [CUDA numerical accuracy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#numerical-accuracy-and-precision)

| 장치 | 왜 필요한가 | 로컬 또는 CI | 우선순위 | 주의점과 공식 근거 |
|---|---|---|---|---|
| 명확한 PyTorch reference | 최적화 kernel과 독립적인 readable oracle 제공 | GPU correctness CI | **필수** | 가능하면 단순 composition으로 작성하고 kernel과 index logic을 그대로 복사하지 않는다. 입력 생성 seed와 reference dtype을 기록한다. |
| `torch.testing.assert_close` | abs/relative tolerance, dtype/device/layout 검사를 일관되게 수행 | GPU correctness CI | **필수** | 판정식은 `abs(actual-expected) <= atol + rtol*abs(expected)`이다. dtype·reduction length·fast-math contract별 `rtol/atol`을 명시하고 default 하나를 모든 kernel에 쓰지 않는다. [PyTorch testing API](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close) |
| error metrics artifact | max abs/rel error와 mismatch 위치를 CI artifact에 남김 | GPU CI failure | **권장** | NaN/Inf, all-zero reference에서 relative error만 보는 함정을 따로 처리한다. `equal_nan`은 API contract가 NaN 보존을 요구할 때만 켠다. [PyTorch `assert_close`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close) |
| forward/backward 분리 | FlashAttention 같은 autograd kernel에서 forward pass만 맞는 오류 방지 | GPU nightly, 안정화 후 PR | **권장** | backward reference는 input별 gradient를 각각 비교하고 accumulation/zeroing을 통제한다. large matrix보다 작은 edge matrix부터 시작한다. |
| invariant/metamorphic checks | reference 비용이 큰 shape에서도 보존 법칙, mask, monotonic property 확인 | GPU nightly | **추후** | reference comparison의 대체가 아니라 보완이다. property가 수학적으로 항상 참인 경우만 gate한다. |

Correctness mode와 benchmark mode는 분리한다. deterministic mode나 debug line info는 correctness 진단에 도움을 줄 수 있지만 성능을 바꾸며, PyTorch도 deterministic operation이 느릴 수 있음을 명시한다. [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness)

## Benchmark와 performance regression

| 도구/장치 | 왜 필요한가 | 로컬 또는 CI | 우선순위 | 주의점과 공식 근거 |
|---|---|---|---|---|
| `triton.testing.do_bench` | GPU warm-up과 반복 timing을 일관되게 수행 | 고정 GPU runner | **권장** | `warmup`, `rep`, quantile/median을 명시한다. 단일 측정이나 wall-clock `time.perf_counter`로 비동기 launch를 재지 않는다. [Triton `do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html), [CUDA timing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#timing) |
| `triton.testing.perf_report` | shape/provider sweep와 plot/report 생성 | nightly/manual | **권장** | report 생성용이며 merge gate threshold 저장소는 별도 JSON으로 둔다. [Triton `perf_report`](https://triton-lang.org/main/python-api/generated/triton.testing.perf_report.html), [Benchmark API](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) |
| cold compile benchmark | compiler/JIT 변화로 개발·startup latency가 악화되는지 확인 | nightly/manual | **추후** | runtime benchmark와 분리한다. Triton은 `TRITON_ALWAYS_COMPILE=1`, CuTe는 cache를 우회하는 `cute.compile`을 활용하되 disk/memory cache 조건을 artifact에 기록한다. [Triton debug knobs](https://github.com/triton-lang/triton#tips-for-building), [CuTe JIT caching](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html) |
| baseline JSON + comparator | main 기준 대비 latency/throughput regression을 기계적으로 검출 | 고정 GPU CI | **추후 gate** | key에 kernel, shape, dtype, provider, GPU model/SM, driver, CUDA, Torch, Triton/CuTe version을 포함한다. 먼저 여러 주간 advisory 결과로 variance를 측정한 뒤 threshold를 정한다. |
| profile recipe (Nsight Systems/Compute) | gate가 느려진 이유를 timeline/metric으로 분석 | 수동/실패 후 | **추후** | profiler 자체를 PR gate로 만들지 말고 재현 command와 artifact naming만 표준화한다. [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [Nsight Compute CLI guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) |

성능 비교는 같은 GPU class와 통제된 runner에서만 한다. 최소 metadata는 GPU name/UUID, SM capability, driver, CUDA runtime/toolkit, clock/power/temperature 상태, Python/Torch/Triton/CuTe version, git SHA, input shape/dtype, warm-up/repetition이다. `nvidia-smi`는 selective query와 `-q`를 통해 driver, architecture, clock, power 등 GPU 상태를 기록할 수 있다. [NVIDIA SMI documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

초기에는 결과를 artifact/comment로만 남기고 merge를 막지 않는다. runner noise가 파악된 후, 예를 들어 “median 또는 선택한 quantile이 같은 baseline 대비 관찰된 정상 variance를 유의미하게 넘었을 때” gate하도록 정한다. 임의의 5%/10% 수치를 지금 고정하면 false positive 또는 실제 regression 누락 가능성이 크다.

## NVIDIA Compute Sanitizer

| 도구/장치 | 왜 필요한가 | 로컬 또는 CI | 우선순위 | 주의점과 공식 근거 |
|---|---|---|---|---|
| `memcheck` | out-of-bounds, misaligned global/local/shared access와 hardware exception 검출 | 로컬 진단 + nightly GPU | **필수 subset** | 가장 먼저 실행한다. 작은 deterministic `sanitizer` marker subset에 `--error-exitcode 1`을 주어 CI failure로 변환한다. [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#memcheck-tool) |
| `racecheck` | shared-memory hazard 및 async copy synchronization 문제 검출 | nightly/manual | **권장** | memcheck를 먼저 통과해야 한다. racecheck 자체는 memory access error check를 하지 않는다. Ampere+ async-copy hazard도 지원한다. [Racecheck](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#racecheck-tool) |
| `initcheck` | 초기화되지 않은 device global/shared memory read 검출 | nightly/manual | **권장** | memory error check를 포함하지 않으므로 memcheck 뒤에 실행한다. 기본 address space는 global이며 shared는 옵션으로 별도 확인한다. [Initcheck](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#initcheck-tool) |
| `synccheck` | divergent barrier, 잘못된 synchronization primitive 사용 검출 | nightly/manual | **권장** | memory check가 아니며 architecture/instruction 지원 범위를 확인한다. [Synccheck](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#synccheck-tool) |
| source line info | sanitizer report를 Python/DSL source와 연결 | sanitizer job | **권장** | CuTe는 `CUTE_DSL_LINEINFO=1` 또는 debug option을 제공한다. debug 설정은 IR/PTX와 cache key를 바꾸므로 그 결과로 benchmark하지 않는다. [CuTe debugging](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html#source-code-correlation) |

대표 command contract는 다음과 같다.

```bash
compute-sanitizer \
  --tool memcheck \
  --error-exitcode 1 \
  --target-processes all \
  uv run pytest -m sanitizer
```

`--error-exitcode`는 application 자체가 성공해도 sanitizer가 error를 찾으면 자동화 suite를 실패시키기 위한 공식 옵션이다. `--target-processes all`은 test runner가 만든 child process도 추적한다. [Compute Sanitizer CLI options](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#command-line-options)

전체 correctness matrix를 sanitizer 아래서 돌리면 매우 느리다. 각 memory pattern, mask path, tile boundary를 대표하는 작은 subset을 만들고 memcheck는 정기 gate, race/init/sync는 nightly 또는 kernel 변경 label에서 실행하는 구성이 현실적이다. Triton 공식 문서도 NVIDIA GPU debugging에 `compute-sanitizer`를 권장한다. [Triton debugging tools](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html#using-third-party-tools)

## GPU CI runner 전략

| 도구/장치 | 왜 필요한가 | 로컬 또는 CI | 우선순위 | 주의점과 공식 근거 |
|---|---|---|---|---|
| GitHub standard hosted CPU runner | lock/lint/type/CPU test를 GPU queue 없이 병렬 처리 | 모든 PR | **필수** | standard runner에는 여기서 필요한 NVIDIA GPU를 가정하지 않는다. CPU lane을 merge-required로 먼저 둔다. [GitHub-hosted runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) |
| GitHub larger GPU runner | 자체 GPU host 운영 없이 managed GPU lane 사용 | 조직/enterprise GPU CI | **권장 후보** | larger runner는 Team/Enterprise 조직용이며 GPU-powered option을 제공한다. 실제 SKU, 지역, 비용, 이용 가능 architecture가 지원 matrix와 맞는지 먼저 확인한다. [GitHub larger runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#larger-runners) |
| ephemeral self-hosted GPU runner | H100 등 원하는 GPU/driver/toolkit을 정확히 고정 | private/trusted GPU CI | **권장 후보** | GitHub는 autoscaling에 persistent보다 ephemeral self-hosted runner를 권장하며 runner당 job 하나만 할당한다. job 뒤 disk/cache/credential을 폐기하고 runner log는 외부 저장한다. [self-hosted runners reference](https://docs.github.com/en/actions/reference/runners/self-hosted-runners#ephemeral-runners-for-autoscaling) |
| runner group + labels | `gpu`, `h100`, `sm90`, `cuda-12.9`를 정확히 routing | GPU CI | **필수(자체 runner 사용 시)** | GitHub custom label은 GPU hardware routing 용례를 공식적으로 지원하고 group으로 repo/workflow 접근을 제한할 수 있다. [labels](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow#using-custom-labels-to-route-jobs), [runner groups](https://docs.github.com/en/actions/concepts/runners/runner-groups) |
| concurrency 1 + clean benchmark runner | 다른 job의 GPU utilization/clock interference 방지 | benchmark CI | **필수(benchmark gate 시)** | correctness runner와 benchmark runner를 같은 시간에 공유하지 않는다. runner metadata와 preflight를 baseline key에 남긴다. |
| trusted trigger policy | fork PR code가 privileged GPU host/credential을 공격하지 못하게 함 | self-hosted GPU CI | **필수** | GitHub는 public repo fork PR이 self-hosted machine에서 위험한 code를 실행할 수 있어 self-hosted runner를 private repo에만 쓰기를 권장한다. public repo라면 untrusted fork에는 CPU hosted CI만 주고, GPU 실행은 검토된 SHA와 제한된 runner group에서 수행한다. [runner access warning](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/manage-access) |

권장 workflow 분해:

1. `ci-cpu.yml`: PR마다 lock, pre-commit all-files, Ruff, type, CPU/reference test, build smoke.
2. `ci-gpu-smoke.yml`: trusted PR에서 공통 CUDA 12.9 환경으로 작은 Triton/CuTe compile/runtime/correctness를 각각 실행.
3. `ci-gpu-nightly.yml`: 넓은 correctness matrix와 architecture matrix.
4. `ci-sanitizer.yml`: schedule/manual, sanitizer marker subset.
5. `ci-benchmark.yml`: schedule/manual, 고정 runner, baseline artifact; 충분한 data 전까지 non-blocking.

## Reproducibility와 environment capture

| 도구/장치 | 왜 필요한가 | 로컬 또는 CI | 우선순위 | 주의점과 공식 근거 |
|---|---|---|---|---|
| `uv.lock` + exact sync | Python dependency graph 재현 | 로컬 + CI | **필수** | lockfile을 수동 편집하지 않고 CI에서 `--locked`로 검증한다. [uv lockfile](https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile) |
| container base digest pin | 같은 CUDA userspace/toolchain image 재사용 | devcontainer + CI image | **권장** | tag는 이동할 수 있다. digest는 immutable identifier지만 security update도 자동으로 받지 않으므로 Dependabot/정기 갱신 절차가 필요하다. [Docker pull by digest](https://docs.docker.com/reference/cli/docker/image/pull/#pull-an-image-by-digest-immutable-identifier) |
| uv/tool installer version pin | rebuild 시 uv, Node, Codex installer 변화 방지 | container build | **권장** | uv 공식 Docker guide도 reproducible build에는 SHA256 pin을 best practice로 설명한다. 현재 Dockerfile의 unversioned remote installer는 별도 갱신 정책과 hash 검증 후보다. [uv in Docker](https://docs.astral.sh/uv/guides/integration/docker/#installing-uv) |
| `python -m torch.utils.collect_env` | PyTorch build CUDA, runtime, OS, compiler, GPU/driver 정보를 failure artifact로 남김 | 모든 GPU job | **필수** | PyTorch가 공식 제공하는 environment collector다. [torch.utils.collect_env](https://docs.pytorch.org/docs/stable/utils.html#torch-utils-collect-env) |
| `nvidia-smi -q` 및 selective CSV query | GPU model/architecture/driver/clock/power/temperature 기록 | 모든 GPU job; 상세치는 benchmark | **필수** | `nvidia-smi`가 제공하는 query field와 의미를 그대로 사용한다. serial 같은 불필요한 민감/식별 정보는 artifact에서 제외한다. [NVIDIA SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html) |
| test manifest JSON | git SHA, seed, marker, shape/dtype, package versions, runner label을 결과와 묶음 | GPU/sanitizer/benchmark CI | **권장** | log text만 파싱하지 말고 machine-readable artifact를 한 schema로 둔다. secret/environment 전체 dump는 금지한다. |
| compiler artifact on failure | TTIR/LLVM/PTX/CUBIN을 compile failure 분석에 사용 | 실패 시 제한 업로드 | **추후** | CuTe는 `CUTE_DSL_KEEP`/dump dir와 compiled object의 PTX/CUBIN/MLIR access를 제공한다. artifact에 proprietary kernel이나 민감 경로가 포함될 수 있으므로 보존 기간과 공개 범위를 정한다. [CuTe debugging artifacts](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html#save-generated-artifacts-to-files) |

PyTorch는 같은 seed라도 release, platform, CPU/GPU 사이 완전한 재현을 보장하지 않는다고 명시한다. 따라서 “seed가 같다”는 정보만으로 성능/correctness 결과를 다른 hardware baseline과 직접 비교하지 않는다. [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness)

## Documentation, commit, security checks

| 도구/장치 | 왜 필요한가 | 로컬 또는 CI | 우선순위 | 주의점과 공식 근거 |
|---|---|---|---|---|
| README + CONTRIBUTING test matrix | GPU가 필요한 command, supported SM/CUDA, marker, tolerance/benchmark policy를 사람이 찾을 수 있게 함 | version control + review | **필수** | root README에는 빠른 CPU path와 GPU path를 분리하고 각 kernel member README에는 contract, reference, supported dtype/shape/SM을 기록한다. |
| `markdownlint-cli2` | Markdown/CommonMark의 구조 오류와 일관성 검사 | docs 변경 commit + CPU CI | **권장** | Node runtime이 필요하지만 pre-commit hook 또는 container 사용을 공식 지원한다. kernel 수식/긴 URL에 필요한 rule exception만 좁게 둔다. [markdownlint-cli2](https://github.com/DavidAnson/markdownlint-cli2#pre-commit) |
| Conventional Commits + Commitizen | release/changelog automation이 필요할 때 machine-readable history 생성 | `commit-msg` + CI | **추후** | 초기 연구 repo에는 commit 형식 강제가 가치보다 마찰이 클 수 있다. 자동 release를 도입할 때 `commit-msg` hook을 설치한다. [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/), [Commitizen pre-commit integration](https://commitizen-tools.github.io/commitizen/) |
| `detect-private-key` + GitHub push protection | API token, SSH/private key 유출을 commit/push 양쪽에서 차단 | local hook + GitHub | **필수/권장** | local pattern check는 방어층 하나일 뿐이다. GitHub push protection은 supported secret을 push 시점에 차단한다. [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks#detect-private-key), [GitHub push protection](https://docs.github.com/en/code-security/concepts/secret-security/push-protection) |
| `pip-audit` | installed Python dependency의 알려진 vulnerability 탐지 | weekly CI + dependency PR | **권장** | `uv sync --locked`로 만든 실제 venv를 audit하거나 supported locked export를 사용한다. `pip-audit`은 malicious package/static code analyzer가 아니고 wheel 내부 native shared library 취약점을 모두 찾는다고 가정하면 안 된다. [pip-audit official source/security model](https://github.com/pypa/pip-audit) |
| Dependabot `uv` + `pre-commit` ecosystems | lock dependency와 hook revision의 정기 update PR 생성 | GitHub scheduled | **권장** | GitHub는 `uv`와 `pre-commit` ecosystem을 공식 지원한다. GPU compatibility를 보존하기 위해 update PR에서도 CPU + GPU smoke를 통과시킨다. [Dependabot supported ecosystems](https://docs.github.com/en/code-security/reference/supply-chain-security/supported-ecosystems-and-repositories) |
| Dependency Review Action | PR이 새로 추가하는 취약 dependency를 merge 전 검토 | GitHub PR CI | **추후/권장** | GitHub dependency graph/feature availability를 확인한 뒤 활성화한다. [dependency review action](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/manage-your-dependency-security/configure-dependency-review-action) |
| CycloneDX SBOM export | release/experiment 환경의 dependency 목록 보관 | release/manual | **추후** | `uv export --format cyclonedx1.5`를 공식 지원한다. SBOM은 vulnerability scan 자체가 아니라 입력 artifact다. [uv export](https://docs.astral.sh/uv/concepts/projects/export/) |

## 권장 도입 순서

### P0 — 이 branch에서 contract를 먼저 확정

1. 공통 CUDA 12.9.1 environment에서 Torch cu129/Triton과 CuTe DSL 4.6.1을 함께 사용하고, `scripts/verify_gpu_stack.py`로 두 DSL의 실제 H100 compile/runtime smoke를 실행한다.
2. `uv lock --check`/`uv sync --locked`, pre-commit file hygiene, Ruff lint/format을 단일 command contract로 만든다.
3. pytest marker와 strict marker, timeout, CPU/GPU selection contract를 만든다.
4. 각 kernel package에 최소 CPU reference test, GPU compile smoke, runtime smoke, numerical correctness test의 자리와 naming을 정한다.
5. root README/CONTRIBUTING에 setup, CPU-only path, GPU path, supported environment를 기록한다.

### P1 — 첫 kernel과 함께 required CI

1. CPU GitHub-hosted CI를 required check로 둔다.
2. trusted GPU runner에서 Triton compile/runtime/correctness smoke를 둔다.
3. CuTe 공통 CUDA 12.9 environment에서는 `cute.compile`/runtime smoke를 별도 job으로 둔다.
4. `torch.utils.collect_env`, `nvidia-smi`, test manifest를 artifact로 남긴다.
5. memcheck의 작은 sanitizer subset과 weekly dependency audit를 추가한다.

### P2 — data가 쌓인 뒤

1. 고정 runner에서 benchmark를 비차단 mode로 수집한다.
2. 정상 variance와 hardware/toolchain baseline key가 검증된 뒤 regression threshold를 merge gate로 승격한다.
3. racecheck/initcheck/synccheck, architecture matrix, Hypothesis, Nsight profile recipe를 확대한다.
4. release automation이 생길 때 commit message gate와 SBOM을 도입한다.

## 추천하는 최소 완료 기준

이 레포의 첫 harness는 다음 조건을 만족하면 충분히 작으면서도 실용적이다.

- fresh clone에서 pinned `uv` workflow로 `uv sync --locked`가 성공한다.
- H100에서 `uv run --locked python scripts/verify_gpu_stack.py`가 device/version을 보고하고 Triton correctness와 CuTe compile/launch를 완료한다.
- `pre-commit run --all-files`, Ruff lint/format, type check가 문서화된 한 command로 재현된다.
- CPU CI는 GPU가 없어도 reference/unit test를 실행하고 GPU test를 명시적으로 제외한다.
- GPU CI는 device preflight 뒤 Triton compile/runtime smoke와 reference correctness를 실행한다.
- CuTe DSL 4.6.1은 공통 CUDA 12.9.1 environment의 required dependency이며, Triton과 각각 독립 compile/runtime smoke를 둔다.
- timeout과 marker typo가 CI를 무한 대기시키거나 test를 조용히 누락시키지 않는다.
- benchmark는 GPU-aware timer를 사용하고 correctness/debug mode와 분리한다.
- sanitizer와 performance 결과에는 재현 가능한 environment metadata가 붙는다.
- dependency/secret/docs 검사는 GPU 자원을 소비하지 않는 CPU lane에서 수행한다.

이 기준 이후의 도구는 kernel 수와 운영 비용이 늘어날 때 점진적으로 추가하는 편이 낫다. 특히 모든 검사와 GPU test를 `pre-commit` 한 곳에 몰아넣거나, 다른 GPU/driver에서 얻은 benchmark 숫자를 직접 비교하거나, interpreter test를 실제 GPU compile test로 간주하는 것은 피해야 한다.
