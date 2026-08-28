# Implementing FlashAttention 1 with Triton

## Description

[FlashAttention 1](https://arxiv.org/pdf/2205.14135) is an IO-aware exact attention algorithm
that reduces HBM reads and writes. Conventional attention implementations materialize the
`O(N²)` attention score matrix in memory. FlashAttention 1 computes the matrix in tiles on the
fly instead, reducing the memory required for attention from `O(N²)` to `O(N)`.

## Working Directory

Run all commands below from the project directory. From the repository root:

```bash
cd flash_attention_1_triton
```

## Run Tests

Run the tests with:

```bash
uv run pytest tests/
```

## Run the Benchmark

Run the benchmark with:

```bash
uv run pytest benchmark/benchmark.py
```

## Run the Profiler

Run the profiler with:

```bash
uv run pytest profile/profile.py
```
