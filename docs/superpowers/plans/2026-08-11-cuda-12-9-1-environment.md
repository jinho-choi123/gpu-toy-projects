# CUDA 12.9.1 Unified Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the repository's development and Python GPU stack from CUDA 12.8 to a unified CUDA 12.9 baseline that can run Triton and CuTe DSL.

**Architecture:** The devcontainer owns the system CUDA toolkit, while `pyproject.toml` owns the PyTorch CUDA wheel index and CuTe DSL package. `uv.lock` is regenerated from those declarations, and the research note is updated to match the accepted single-lane decision.

**Tech Stack:** NVIDIA CUDA 12.9.1, Ubuntu 24.04, uv workspace, PyTorch 2.11.0+cu129, Triton, NVIDIA CuTe DSL 4.6.1.

## Global Constraints

- The base image must be exactly `nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04`.
- The devcontainer must install `python3-dev` so Triton can build its CPython driver extension.
- The workspace must support CPython 3.12 through 3.14 (`>=3.12,<3.15`) on Linux only; Windows and macOS are unsupported.
- PyTorch must remain `2.11.0` and use the official `https://download.pytorch.org/whl/cu129` index on Linux only.
- CuTe DSL must be pinned to `nvidia-cutlass-dsl==4.6.1`.
- Triton must remain owned by the PyTorch dependency graph.
- The existing uncommitted research document must be preserved and updated, not discarded.

---

### Task 1: Update the declared CUDA environment

**Files:**
- Modify: `.devcontainer/Dockerfile:1`
- Modify: `.devcontainer/devcontainer.json:2,13`
- Modify: `pyproject.toml:10-46`

**Interfaces:**
- Consumes: The current CUDA 12.8 devcontainer and `pytorch-cu128` explicit uv index.
- Produces: A Linux-only CUDA 12.9.1 devcontainer, `pytorch-cu129` index, and pinned CuTe DSL dependency.
  The image also provides Python development headers required by Triton runtime compilation.

- [ ] **Step 1: Run a failing declaration check**

```bash
test "$(sed -n '1p' .devcontainer/Dockerfile)" = "FROM nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04"
rg -q 'gpu-toy-projects-cuda129' .devcontainer/devcontainer.json
rg -q 'pytorch-cu129' pyproject.toml
rg -q 'nvidia-cutlass-dsl==4.6.1' pyproject.toml
rg -q '^[[:space:]]+python3-dev' .devcontainer/Dockerfile
```

Expected: FAIL because the repository still declares CUDA 12.8/cu128 and has no CuTe DSL dependency.

- [ ] **Step 2: Apply the minimal declaration changes**

Set the Docker `FROM` line to the required CUDA 12.9.1 image, install `python3-dev` for Triton runtime compilation, replace both container name occurrences with `cuda129`, add `nvidia-cutlass-dsl==4.6.1`, and replace the PyTorch uv index name and URL with `pytorch-cu129` and `/whl/cu129`.

- [ ] **Step 3: Re-run the declaration check**

Run the Step 1 commands again.

Expected: all commands exit 0 on the supported Linux and CPython 3.12–3.14 contract.

- [ ] **Step 4: Commit**

```bash
git add .devcontainer/Dockerfile .devcontainer/devcontainer.json pyproject.toml
git commit -m "build: move GPU stack to CUDA 12.9"
```

### Task 2: Regenerate and validate the dependency lock

**Files:**
- Modify: `uv.lock`

**Interfaces:**
- Consumes: The CUDA 12.9 PyTorch index and CuTe DSL declaration from Task 1.
- Produces: A lockfile resolving Torch cu129, its CUDA 12.9 components, Triton, and CuTe DSL 4.6.1.

- [ ] **Step 1: Verify the old lock is rejected**

```bash
uv lock --check
```

Expected: FAIL because `pyproject.toml` now declares `>=3.12,<3.15` and Linux-only environment/source metadata, while `uv.lock` retains the former `>=3.12` Python contract and platform split.

- [ ] **Step 2: Regenerate the lockfile**

```bash
uv lock
```

Expected: resolution succeeds for the Linux-only explicit cu129 index.

- [ ] **Step 3: Verify locked package identities**

```bash
uv lock --check
rg -q 'version = "2.11.0\+cu129"' uv.lock
rg -q 'https://download.pytorch.org/whl/cu129' uv.lock
rg -q 'name = "nvidia-cutlass-dsl"' uv.lock
rg -q 'version = "4.6.1"' uv.lock
```

Expected: all commands exit 0.

- [ ] **Step 4: Commit**

```bash
git add uv.lock
git commit -m "build: lock CUDA 12.9 dependencies"
```

### Task 3: Update decision documentation and verify runtime

**Files:**
- Modify: `docs/research/2026-08-11-gpu-kernel-harness.md`
- Verify: `.devcontainer/Dockerfile`, `.devcontainer/devcontainer.json`, `pyproject.toml`, `uv.lock`

**Interfaces:**
- Consumes: The completed CUDA 12.9 declarations and lockfile.
- Produces: Non-contradictory documentation and fresh static/runtime evidence.

- [ ] **Step 1: Update the research decision**

Replace the historical split-lane recommendation with the accepted common CUDA 12.9.1 environment. Preserve the original compatibility rationale and state that separate Triton/CuTe test selections remain useful even though they share one toolkit.

- [ ] **Step 2: Scan for stale configuration identifiers and unsupported platform claims**

```bash
git ls-files .devcontainer pyproject.toml uv.lock | xargs rg -n 'cuda128|cu128|pytorch-cu128|12\.8\.1-devel'
rg -n 'Linux and Windows|win32|Windows wheels' pyproject.toml docs/research docs/superpowers/specs
```

Expected: no stale CUDA identifiers and no claim that Windows wheels resolve; Windows may appear only as unsupported.

- [ ] **Step 3: Sync and verify Python packages**

```bash
uv sync --locked
uv run --locked python -c 'import cutlass, cutlass.cute, torch, triton; print(cutlass.__version__, cutlass.CUDA_VERSION, torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())'
```

Expected: on supported Linux and CPython 3.12–3.14, CuTe DSL 4.6.1 imports; Torch reports `2.11.0+cu129` and CUDA `12.9`; Triton imports; CUDA is available with H100 capability `(9, 0)`.

- [ ] **Step 4: Run repository verification**

```bash
uv lock --check
uv run --locked ruff check .
uv run --locked ty check
git diff --check
```

Expected: all commands exit 0. `pytest` is not a completion gate yet because the baseline repository contains no tests.

- [ ] **Step 5: Record the devcontainer rebuild boundary**

The current shell remains inside the old CUDA 12.8 container. Report that `nvcc --version` can show CUDA 12.9.1 only after rebuilding/reopening the devcontainer from the updated Dockerfile.

- [ ] **Step 6: Commit**

```bash
git add docs/research/2026-08-11-gpu-kernel-harness.md docs/superpowers/specs/2026-08-11-cuda-12-9-1-environment-design.md docs/superpowers/plans/2026-08-11-cuda-12-9-1-environment.md
git commit -m "docs: record unified CUDA 12.9 environment"
```
