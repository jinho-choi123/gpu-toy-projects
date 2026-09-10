# Implementing FlashAttention 1 with Triton

## Description

[FlashAttention 1](https://arxiv.org/pdf/2205.14135) is an IO-aware exact attention algorithm
that reduces HBM reads and writes. Conventional attention implementations materialize the
`O(N²)` attention score matrix in memory. FlashAttention 1 computes the matrix in tiles on the
fly instead, reducing the memory required for attention from `O(N²)` to `O(N)`.

## Usage

Import the fixed-length and variable-length APIs from the package root:

```python
from flash_attention_1_triton import flash_attention_func, flash_attention_varlen_func

output = flash_attention_func(
    q,
    k,
    v,
    causal=False,
    softmax_scale=None,
)

varlen_output = flash_attention_varlen_func(
    q_packed,
    k_packed,
    v_packed,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    causal=False,
    softmax_scale=None,
)
```

The fixed-length API implements forward attention with a Triton kernel. Calls that
require autograd raise a backward-not-implemented error. The variable-length API
is still an unimplemented stub.

Earlier API drafts exported `flash_attention` and `flash_attention_varlen`. These names were
replaced by `flash_attention_func` and `flash_attention_varlen_func`, respectively.

## Forward kernel

`src/flash_attention_1_triton/_flash_attention_kernel.py` implements the paper's
Algorithm 1: an outer K/V tile loop
and an inner Q tile loop, with one Triton program per `(batch, head)` pair.

`_flash_attention_func.py` supplies input validation, a `(batch, heads)` launch grid,
element strides, sequence lengths, scale, causal flag, and fixed 32-by-32 tiles with
four warps. The launcher initializes contiguous FP32 state in GPU memory:

- `O_ptr`: output/state `[batch, query_length, heads, head_dim]`, initialized to zero.
- `M_ptr`: row maxima `[batch, query_length, heads]`, initialized to negative infinity.
- `L_ptr`: row sums with the same layout, initialized to zero.

Each program owns every query row of its batch/head pair. Write the final normalized
output to `O_ptr`; the launcher converts it to the input dtype. This initial launch
configuration prioritizes following the paper; small batch/head counts limit parallelism.

For tensors with `requires_grad=True`, call the API inside `torch.no_grad()` to run
forward. Backward remains a separate implementation task.

Run forward correctness, launcher validation, and annotation checks:

```bash
uv run --locked pytest tests/test_flash_attention_func.py tests/test_forward_scaffold.py tests/test_annotations.py -k "not test_output_and_gradients" -q --capture=fd
```

Forward-only and combined output/gradient tests reuse the same input cases.
The combined tests remain the acceptance criteria for backward implementation.

## Working Directory

Run all commands below from the project directory. From the repository root:

```bash
cd flash_attention_1_triton
uv sync --locked
```

## Development checks

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
```

## Tests

```bash
uv run --locked pytest tests
```

The tests compare outputs and first-order Q/K/V gradients against explicit FP32
PyTorch attention. They cover FP16/BF16, head dimensions 32/64/128, bottom-right
causal masks, custom scales, boundary lengths, strided inputs, and packed batches.
FP16 uses `atol=1e-3, rtol=1e-2`; BF16 uses `atol=1e-2, rtol=5e-2` for outputs
and gradients. Causal queries longer than their keys must raise `ValueError`.

API tests require CUDA and skip when it is unavailable; BF16 cases also skip on
unsupported GPUs. The reference checks run on CPU. To select a GPU, prefix the
command with `CUDA_VISIBLE_DEVICES=<index>`.

Backward and variable-length attention are currently unimplemented, so their
acceptance tests intentionally fail with `NotImplementedError`; they are not marked
as expected failures. Fixed-length forward and launcher validation tests can pass now.

Tests show case names and live Loguru messages by default (pytest `-v -s`). Logs
cover case start/end, input creation, FP32 reference calculation, API calls, output
checks, backward passes, and Q/K/V gradient comparisons with maximum absolute errors
and tolerances. `END` marks fixture cleanup, not a passing result; pytest reports
pass/fail/skip separately. Unimplemented APIs stop at the API-call stage.

For quiet output with logs shown only on failure, run:

```bash
uv run --locked pytest tests -q --capture=fd
```

## Type annotations

Every function argument and return value must have a type annotation. Ruff's
`ANN` rules enforce this; `ty check` covers both `src/` and `tests/`.
Tensor arguments and returns must use jaxtyping with an explicit dtype and shape,
for example `Float32[Tensor, "batch query_length heads head_dim"]`. Use unions for
supported dtype alternatives and named axes for dimensions. Shared helpers may use
variadic axes to express the fixed-length and packed layouts they accept.

`tests/test_annotations.py` checks Tensor signatures (including aliases and nested
containers) and rejects bare `Tensor` and dtype-unspecified `Shaped` annotations.
The Tensor rule test also runs as a local pre-commit hook alongside Ruff and ty.
Triton kernel arguments use `tl.tensor` and `tl.constexpr`, since the JIT receives GPU
pointers and specialization constants. Only the kernel module is excluded from the
package's beartype import hook so that Python type-checking wrappers do not enter JIT code.
These are annotation checks; adding annotations alone does not enable runtime
shape validation. Public APIs retain their existing jaxtyping/beartype validation.

```bash
uv run --locked ruff check .
uv run --locked ty check
uv run --locked pytest tests/test_annotations.py
```

## Docstrings

Use Google-style docstrings with an `Args` entry explaining each parameter and a
`Returns` entry describing the result. Use `Yields` for generator fixtures. For
functions returning `None`, describe the check or side effect. Dtype text inside
docstrings is optional; jaxtyping dtype and shape annotations remain required.

Ruff explicitly selects `D417` (missing parameter descriptions within an existing
Args section), `DOC201` (missing Returns), and `DOC402` (missing Yields). The DOC
rules require preview mode; only explicitly selected preview rules are enabled.
Ruff does not require an Args section when it is absent, and exempts None-only
returns/yields and stubs from the corresponding DOC rules.

## Fixed-length benchmark

Compare `flash_attention_func` with CUDA SDPA forced to `SDPBackend.MATH`, which
materializes the attention matrix. Run one shape/dtype combination per invocation:

```bash
uv run --locked python -m flash_attention_1_triton.benchmark \
  --batch-size 2 --query-length 1024 --key-length 1024 \
  --heads 8 --head-dim 64 --dtype float16 --mode forward --cuda-graph
```

- `--mode forward` measures inference forward without autograd; `backward` measures
  Q/K/V gradients using a retained, precomputed forward graph; `both` measures a fresh
  forward and its backward together. Gradients do not accumulate between iterations.
- Omit `--cuda-graph` for ordinary execution; include it to capture one operation and
  time replays. Both implementations use the same setting.
- Dtypes: `float16` / `bfloat16`; head dimensions: `32` / `64` / `128`.
  `--key-length` defaults to `--query-length`. Attention is non-causal, with no dropout
  and the default `1 / sqrt(head_dim)` scale.
- `--warmup` (default 25) and `--iterations` (default 100) control repetition counts.
  Results report median CUDA-event milliseconds and `MATH / Triton` speedup.
  Input creation, correctness checks, compilation/warmup, and graph capture are excluded.
  Ordinary execution can include GPU idle gaps from host dispatch; graph mode reduces
  that overhead. No cache flush is performed between samples.
- SDPA MATH retains FP32 intermediates for FP16/BF16 inputs. The comparison measures
  these implementations as provided, including their precision differences and API
  layout views; it does not isolate materialization as the only source of speedup.
- Outputs and, for training modes, gradients are checked before timing. While the
  Triton API raises `NotImplementedError`, only MATH latency is reported and speedup
  is explicitly unavailable. Other errors are not suppressed.

Select the GPU with `CUDA_VISIBLE_DEVICES=<index>`. Varlen benchmarking is excluded.
