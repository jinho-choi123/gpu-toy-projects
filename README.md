# GPU Toy Projects

GPU kernel experiments built with PyTorch, Triton, and NVIDIA CuTe DSL.

## Minimum requirements

| Component       | Requirement                                |
| --------------- | ------------------------------------------ |
| GPU             | NVIDIA H100 (SM90, compute capability 9.0) |
| OS              | Linux (Ubuntu 24.04)                       |
| NVIDIA driver   | 575.51.03 or newer                         |
| CUDA toolkit    | 12.9.1                                     |
| Python          | 3.13 (managed by `uv`)                     |
| Package manager | `uv`                                       |

RAM and VRAM requirements vary by project.

## Install

```bash
git clone https://github.com/jinho-choi123/gpu-toy-projects.git
cd gpu-toy-projects
```

Each subproject owns its own `pyproject.toml`, `uv.lock`, and virtual
environment. Run `uv` commands from the subproject directory.

### Dev Container (recommended)

Requires Docker, NVIDIA Container Toolkit, VS Code, and the Dev Containers extension.

1. Open the repository in VS Code.
2. Run **Dev Containers: Reopen in Container**.
3. Enter the project you want to use and install its locked dependencies as
   shown below.

### Native Linux

On a host that already meets the GPU, OS, driver, and CUDA requirements, enter
the project you want to use and install its locked dependencies as shown below.
`uv` downloads a compatible Python version when needed.

## Project setup

From the repository root, set up FlashAttention 1:

```bash
cd flash_attention_1_triton
uv sync --locked
```

Or set up FlashQLA profiling:

```bash
cd profile_flashqla
uv sync --locked
```

Continue from that project directory and follow its README:
[FlashAttention 1](flash_attention_1_triton/README.md) or
[FlashQLA profiling](profile_flashqla/README.md).

## Troubleshooting

- **CUDA is not available:** check `nvidia-smi` and GPU access inside the container.
- **H100 capability mismatch:** these projects target compute capability 9.0.
- **Version mismatch:** rebuild/reopen the Dev Container, enter the affected
  project directory, then run `uv sync --locked` again.
