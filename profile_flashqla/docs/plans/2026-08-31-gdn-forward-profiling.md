# GDN Forward Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two SM90-only entry points that compare measured CUDA Graph replays of the FlashQLA and FLA Triton GDN forward implementations.

**Architecture:** Two thin scripts select the backend through `FLA_FLASH_QLA` and delegate to `gdn_profile.py`. The runner owns deterministic input creation, fixed-device validation, eager warmup, graph capture/replay, NVTX, and CUDA Profiler API lifecycle; `utils/gdn.py` owns the shared input contract and FLA forward call used by production and tests.

**Tech Stack:** Python 3.13, PyTorch 2.11, flash-linear-attention 0.5.2, flash-qla 0.1.2, Loguru 0.7.3, pytest, Nsight Systems 2026.2, Nsight Compute 2025.2.

**Spec:** [docs/specs/gdn-forward-profiling.md](../specs/gdn-forward-profiling.md)

## Global Constraints

- Work from the Git root; paths in this plan are Git-root relative.
- Support only the fixed `cuda:0` device on SM90 and fail before allocation otherwise.
- Profile forward-only CUDA Graph replays under `torch.inference_mode()`; never call backward.
- Fix Q/K/V to BF16 and both key/value head dimensions to 128.
- Use `output_final_state=True` and `use_qk_l2norm_in_kernel=True`.
- Allocate deterministic inputs once and reuse them for eager warmup, graph capture, graph replay warmup, and measurement.
- Run 10 eager warmups on a side stream and 10 graph replay warmups; neither count is configurable.
- Keep graph capture and both warmup phases outside the CUDA Profiler API measurement range.
- Use 10 measured graph replays in documented Nsight Systems commands and one in Nsight Compute commands.
- Keep Loguru's default stderr handler and do not create metadata sidecars.
- Do not perform numerical validation in profiling entry points.
- Do not commit generated `.nsys-rep` or `.ncu-rep` files.

## File Map

- Modify `pyproject.toml`: include `profile_flashqla` in the root `ty` source set.
- Modify `profile_flashqla/tests/test_flashqla.py`: share the GDN contract/forward helper and retain small plus long-context comparisons.
- Create `profile_flashqla/src/profile_flashqla/utils/gdn.py`: shared `GdnInput` and `run_forward`.
- Create `profile_flashqla/src/profile_flashqla/utils/__init__.py`: export the shared contract.
- Create `profile_flashqla/src/profile_flashqla/gdn_profile.py`: deterministic CUDA Graph replay runner.
- Create `profile_flashqla/scripts/profile_gdn_flash_qla.py`: FlashQLA selector.
- Create `profile_flashqla/scripts/profile_gdn_fla_triton.py`: FLA Triton selector.
- Modify `profile_flashqla/README.md`: direct, Nsight Systems, and Nsight Compute commands.
- Modify `profile_flashqla/.gitignore`: ignore `profiles/` relative to the package directory.

---

### Task 1: Keep correctness coverage for both backends

**Files:**

- Modify: `profile_flashqla/tests/test_flashqla.py`

**Interfaces:**

- Produces: `make_gdn_input(*, seq_len: int, num_heads: int) -> GdnInput`.
- Produces: small and canonical long-context fixtures.
- Verifies: backend dispatch plus output/final-state comparison at ratio `0.008`.

- [ ] **Step 1: Keep deterministic small and long-context fixtures**

Use batch size 1, head dimension 128, seed 42, BF16 Q/K/V, and FP32 gate/beta/initial state. Instantiate fixtures with these exact calls:

```python
@pytest.fixture
def gdn_input_fixture() -> GdnInput:
    return make_gdn_input(seq_len=1024, num_heads=4)


@pytest.fixture
def gdn_long_context_input_fixture() -> GdnInput:
    return make_gdn_input(seq_len=16_384, num_heads=16)
```

- [ ] **Step 2: Compare both implementations through the shared helper**

Both comparison tests select the backend through the environment and call the same helper:

```python
monkeypatch.setenv("FLA_FLASH_QLA", "0")
golden_output, golden_final_state = run_forward(gdn_input_fixture)

monkeypatch.setenv("FLA_FLASH_QLA", "1")
flash_qla_output, flash_qla_final_state = run_forward(gdn_input_fixture)
```

Apply the existing `assert_close` ratio to both output and final state. Repeat the same flow for `gdn_long_context_input_fixture`.

- [ ] **Step 3: Run focused correctness tests**

```bash
uv run pytest \
  profile_flashqla/tests/test_flashqla.py::test_flash_qla_matches_fla_triton \
  profile_flashqla/tests/test_flashqla.py::test_flash_qla_matches_fla_triton_long_context \
  -v
```

Expected: both tests pass on SM90.

---

### Task 2: Share the GDN workload contract and enforce type coverage

**Files:**

- Create: `profile_flashqla/src/profile_flashqla/utils/gdn.py`
- Create: `profile_flashqla/src/profile_flashqla/utils/__init__.py`
- Modify: `profile_flashqla/src/profile_flashqla/gdn_profile.py`
- Modify: `profile_flashqla/tests/test_flashqla.py`
- Modify: `pyproject.toml`

**Interfaces:**

- Produces: `GdnInput` and `run_forward(GdnInput) -> tuple[torch.Tensor, torch.Tensor]`.
- Consumes: FLA's public `chunk_gated_delta_rule` API.

- [ ] **Step 1: Define the shared input and forward contract**

Create `utils/gdn.py` with the single FLA call site:

```python
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GdnInput:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    initial_state: torch.Tensor
    scale: float


def run_forward(gdn_input: GdnInput) -> tuple[torch.Tensor, torch.Tensor]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    output, final_state = chunk_gated_delta_rule(
        q=gdn_input.q,
        k=gdn_input.k,
        v=gdn_input.v,
        g=gdn_input.g,
        beta=gdn_input.beta,
        scale=gdn_input.scale,
        initial_state=gdn_input.initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    if final_state is None:
        raise RuntimeError("GDN forward did not return the requested final state.")
    return output, final_state
```

Keep the existing keyword-scoped type-checker ignores on the FLA call because the dependency's published typing does not model this call correctly.

- [ ] **Step 2: Export and consume the shared contract**

`utils/__init__.py` exports only these names:

```python
from .gdn import GdnInput, run_forward

__all__ = ["GdnInput", "run_forward"]
```

Import them from `.utils` in `gdn_profile.py` and from `profile_flashqla.utils` in the correctness tests. Remove the duplicated dataclass and duplicated FLA keyword call from both consumers.

- [ ] **Step 3: Add the package to root type checking**

Set the root configuration to:

```toml
[tool.ty.src]
include = ["flash_attention_1_triton", "profile_flashqla", "scripts"]
exclude = ["docs/"]
```

- [ ] **Step 4: Verify the shared contract**

```bash
uv run ruff check profile_flashqla/src/profile_flashqla profile_flashqla/tests
uv run ty check
```

Expected: both commands exit 0, and `ty` reports diagnostics from `profile_flashqla` if the shared contract or a consumer is broken.

---

### Task 3: Capture and measure GDN CUDA Graph replays

**Files:**

- Modify: `profile_flashqla/pyproject.toml`
- Modify: `uv.lock`
- Create: `profile_flashqla/src/profile_flashqla/gdn_profile.py`

**Interfaces:**

- Produces: `ProfileBackend` with `FLASH_QLA` and `FLA_TRITON`.
- Produces: `main(backend: ProfileBackend, argv: Sequence[str] | None = None) -> int`.
- Consumes: `GdnInput` and `run_forward` from `.utils`.

- [ ] **Step 1: Declare fixed workload controls and the five-option CLI**

```python
HEAD_DIM = 128
DTYPE = torch.bfloat16
EAGER_WARMUP = 10
GRAPH_WARMUP = 10


class ProfileBackend(StrEnum):
    FLASH_QLA = "flash_qla"
    FLA_TRITON = "fla_triton"


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    batch_size: int
    seq_len: int
    num_heads: int
    seed: int
    iterations: int
```

The parser exposes only `--batch-size`, `--seq-len`, `--num-heads`, `--seed`, and `--iterations`. Validate all values except seed as positive integers. Do not add `--device` or `--warmup`.

- [ ] **Step 2: Validate the fixed device and log runtime metadata**

Select `cuda:0`, require CUDA availability and capability `(9, 0)`, then log these exact keys through Loguru's default stderr handler:

```text
backend batch_size seq_len num_heads head_dim value_head_dim dtype device seed eager_warmup graph_warmup iterations
```

Set `FLA_FLASH_QLA` before the first `run_forward` call.

- [ ] **Step 3: Create deterministic input once**

Build Q/K/V with shape `(batch_size, seq_len, num_heads, 128)` and BF16. Build gate/beta with shape `(batch_size, seq_len, num_heads)` and FP32, and initial state with shape `(batch_size, num_heads, 128, 128)` and FP32. Use seed 42 by default and scale `128**-0.5`.

- [ ] **Step 4: Implement the graph lifecycle**

Use this exact measurement ordering:

```python
with torch.inference_mode():
    current_stream = torch.cuda.current_stream(device)
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(current_stream)

    with torch.cuda.stream(warmup_stream):
        for _ in range(EAGER_WARMUP):
            _ = run_forward(gdn_input)

    current_stream.wait_stream(warmup_stream)

    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        _static_output, _static_final_state = run_forward(gdn_input)

    for _ in range(GRAPH_WARMUP):
        cuda_graph.replay()

    torch.cuda.synchronize(device)
    torch.cuda.profiler.start()
    try:
        with torch.cuda.nvtx.range("gdn_forward"):
            for iteration in range(config.iterations):
                with torch.cuda.nvtx.range(f"iteration_{iteration}"):
                    cuda_graph.replay()
        torch.cuda.synchronize(device)
    finally:
        torch.cuda.profiler.stop()
```

The captured outputs remain referenced for the graph lifetime. Numerical comparison remains outside the profiling runner.

- [ ] **Step 5: Run reduced direct workloads**

```bash
uv run python profile_flashqla/scripts/profile_gdn_flash_qla.py \
  --seq-len 1024 --num-heads 4 --iterations 1
uv run python profile_flashqla/scripts/profile_gdn_fla_triton.py \
  --seq-len 1024 --num-heads 4 --iterations 1
```

Expected: both commands exit 0 on SM90, log distinct backend values to stderr, and create no report without an attached profiler.

---

### Task 4: Add selectors and graph-aware Nsight documentation

**Files:**

- Create: `profile_flashqla/scripts/profile_gdn_flash_qla.py`
- Create: `profile_flashqla/scripts/profile_gdn_fla_triton.py`
- Modify: `profile_flashqla/README.md`
- Modify: `profile_flashqla/.gitignore`

**Interfaces:**

- Consumes: `ProfileBackend` and `main` from `profile_flashqla.gdn_profile`.
- Produces: two executable selectors and four explicit report prefixes.

- [ ] **Step 1: Create thin selectors**

The FlashQLA script exits with `main(ProfileBackend.FLASH_QLA)`; the FLA Triton script exits with `main(ProfileBackend.FLA_TRITON)`. Neither script adds its own CLI or profiling behavior.

- [ ] **Step 2: Ignore raw reports**

Add this package-relative entry to `profile_flashqla/.gitignore`:

```gitignore
profiles/
```

- [ ] **Step 3: Document Nsight Systems graph capture**

Both commands include:

```text
--trace=cuda,nvtx
--sample=none
--cuda-graph-trace=node
--capture-range=cudaProfilerApi
--capture-range-end=stop
```

Use `flash_qla_b1_t16384_h16_nsys` and `fla_triton_b1_t16384_h16_nsys` output prefixes and pass `--iterations 10`.

- [ ] **Step 4: Document Nsight Compute graph profiling**

Both commands include:

```text
--set basic
--graph-profiling node
--nvtx
--nvtx-include 'gdn_forward/'
--profile-from-start=off
```

Use `--export` with `flash_qla_b1_t16384_h16_ncu` and `fla_triton_b1_t16384_h16_ncu`, and pass `--iterations 1`. Document `--set detailed` as opt-in and note that GPU performance-counter permissions are required.

- [ ] **Step 5: Verify help and ignore behavior**

```bash
uv run python profile_flashqla/scripts/profile_gdn_flash_qla.py --help
uv run python profile_flashqla/scripts/profile_gdn_fla_triton.py --help
git check-ignore -v profile_flashqla/profiles/probe.nsys-rep
```

Expected: both help outputs show only the five configurable arguments, and the sample report matches `profile_flashqla/.gitignore`.

---

### Task 5: Run final acceptance gates

**Files:**

- Verify only; do not add generated reports.

**Interfaces:**

- Consumes: all prior tasks.
- Produces: test, static-analysis, smoke-run, profiler, and Git-hygiene evidence.

- [ ] **Step 1: Run repository gates**

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run ty check
git diff --check
```

Expected: all commands exit 0. Pytest may emit known dependency deprecation warnings but has no failed tests.

- [ ] **Step 2: Run both reduced direct workloads**

Run the two Task 3 commands and confirm each performs 10 eager warmups, one graph capture, 10 graph replay warmups, and one measured replay.

- [ ] **Step 3: Capture Nsight Systems reports**

Run the two README Nsight Systems commands. Confirm each report contains one `gdn_forward` range, 10 `iteration_N` ranges, and 10 measured `cudaGraphLaunch` calls; graph capture and warmups remain outside the measured range.

- [ ] **Step 4: Attempt Nsight Compute reports**

Run the two README Nsight Compute commands. A successful environment creates one graph-node report per backend. `ERR_NVGPUCTRPERM` is an environment permission failure and must be reported explicitly rather than treated as an application regression.

- [ ] **Step 5: Prove raw reports remain untracked**

```bash
git status --short
git check-ignore -v \
  profile_flashqla/profiles/flash_qla_b1_t16384_h16_nsys.nsys-rep \
  profile_flashqla/profiles/fla_triton_b1_t16384_h16_ncu.ncu-rep
```

Expected: generated reports do not appear in Git status and both paths match `profile_flashqla/.gitignore`.
