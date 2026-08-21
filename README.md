# GPU Toy Projects

GPU kernel experiments built with PyTorch, Triton, and NVIDIA CuTe DSL.

## Minimum requirements

| Component | Requirement |
| --- | --- |
| GPU | NVIDIA H100 (SM90, compute capability 9.0) |
| OS | Linux (Ubuntu 24.04) |
| NVIDIA driver | 575.51.03 or newer |
| CUDA toolkit | 12.9.1 |
| Python | 3.12 or 3.13, including `python3-dev` |
| Package manager | `uv` |

RAM and VRAM minimums are not defined. The current verification script requires an H100.

## Install

```bash
git clone https://github.com/jinho-choi123/gpu-toy-projects.git
cd gpu-toy-projects
```

### Dev Container (recommended)

Requires Docker, NVIDIA Container Toolkit, VS Code, and the Dev Containers extension.

1. Open the repository in VS Code.
2. Run **Dev Containers: Reopen in Container**.
3. Install the locked dependencies:

```bash
uv sync --locked
```

### Native Linux

On a host that already meets the minimum requirements:

```bash
uv sync --locked
```

## Verify the GPU stack

```bash
uv run --locked python scripts/verify_gpu_stack.py
```

Successful verification includes:

```text
Triton smoke passed: result=42
CuTe smoke passed: result=42
GPU stack verification passed
```

A smoke-result mismatch logs the expected and actual values. Any failed check exits with a non-zero status.

## Troubleshooting

- **CUDA is not available:** check `nvidia-smi` and GPU access inside the container.
- **H100 capability mismatch:** the smoke test currently requires compute capability 9.0.
- **Version mismatch:** rebuild/reopen the Dev Container, then run `uv sync --locked` again.
