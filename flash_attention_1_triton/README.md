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

This branch defines the APIs only. Both functions raise `NotImplementedError` until their
FlashAttention implementations are added.

Earlier API drafts exported `flash_attention` and `flash_attention_varlen`. These names were
replaced by `flash_attention_func` and `flash_attention_varlen_func`, respectively.

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
