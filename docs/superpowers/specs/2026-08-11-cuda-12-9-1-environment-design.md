# CUDA 12.9.1 Unified Environment Design

## Goal

Use one CUDA 12.9.1 development environment for both PyTorch/Triton and NVIDIA CuTe DSL work.

## Decisions

- Set the devcontainer base image to `nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04`.
- Install `python3-dev`; Triton compiles a small CPython CUDA driver extension on first GPU use and therefore needs `Python.h`.
- Rename container-facing `cuda128` identifiers to `cuda129`.
- Support CPython 3.12 through 3.14 by declaring `requires-python = ">=3.12,<3.15"` across the workspace.
- Keep `torch==2.11.0`, but resolve Linux wheels from the official `cu129` index so the locked build is `torch==2.11.0+cu129`.
- Add `nvidia-cutlass-dsl==4.6.1` as a runtime dependency. This release supports CUDA 12 and its official changelog identifies CUDA 12.9 as the optimal code-generation toolkit.
- Keep Triton supplied by the pinned PyTorch distribution; do not add an independent Triton pin that could conflict with Torch.
- Regenerate `uv.lock` from project metadata instead of editing CUDA component versions by hand.
- Update the existing harness research note so it records the accepted unified CUDA 12.9 decision instead of recommending split CUDA 12.8/12.9 lanes.

## Compatibility Contract

The supported baseline after this change is:

| Component | Required value |
|---|---|
| Container toolkit | CUDA 12.9.1 |
| OS | Linux, with Ubuntu 24.04 in the devcontainer |
| Python | CPython 3.12 through 3.14 (`>=3.12,<3.15`) |
| Python build headers | `python3-dev` |
| PyTorch | 2.11.0+cu129 |
| Triton | Version selected by PyTorch 2.11.0+cu129 |
| CuTe DSL | 4.6.1, CUDA 12 variant |
| NVIDIA driver | At least the CuTe CUDA 12.9 requirement; the current 580.178.04 host satisfies it |

The container toolkit, PyTorch bundled CUDA runtime, and CuTe DSL package are separate compatibility surfaces. Verification must report all three rather than assuming the base-image tag determines every Python wheel.

## Verification

Static verification checks the exact image, container name, Python range, Linux-only PyTorch source marker, index name/URL, and dependency declarations. `uv lock --check` proves metadata and lock consistency. A Linux synced environment must import Torch, Triton, and `cutlass.cute`, report Torch CUDA 12.9, see the H100 GPU, and compile and execute one minimal Triton kernel and one minimal CuTe kernel. Python 3.15 and non-Linux dependency dry-runs must be rejected by the declared compatibility contract. The base image's `nvcc` 12.9.1 can only be verified after rebuilding the devcontainer.

## Non-goals

- Adding the full pre-commit, pytest taxonomy, sanitizer, benchmark, or CI harness.
- Upgrading PyTorch beyond 2.11.0.
- Independently upgrading Triton beyond the version selected by PyTorch.
- Supporting CUDA 12.8 and 12.9 simultaneously.
- Supporting Windows or macOS.
