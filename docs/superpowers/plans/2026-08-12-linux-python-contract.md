# Linux-Only Python Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Constrain the CUDA workspace to CPython 3.12–3.13 and make uv resolve the GPU dependency graph only for Linux.

**Architecture:** Project metadata is the source of truth: both workspace packages declare the same Python interval, `[tool.uv]` limits resolution to Linux, and the PyTorch cu129 source applies only on Linux. The lockfile is regenerated from metadata, while the CUDA environment and harness documents record the same contract.

**Tech Stack:** Python 3.12–3.13, uv 0.12.3, PyTorch 2.11.0+cu129, Triton 3.6.0, NVIDIA CuTe DSL 4.6.1, Linux CUDA 12.9.1.

## Global Constraints

- Declare `requires-python = ">=3.12,<3.14"` in the root and every workspace member.
- Declare `environments = ["sys_platform == 'linux'"]` under `[tool.uv]`.
- Apply the `pytorch-cu129` source only when `sys_platform == 'linux'`; do not retain a `win32` source branch.
- Keep `torch==2.11.0`, `nvidia-cutlass-dsl==4.6.1`, Triton ownership, and the CUDA 12.9.1 devcontainer unchanged.
- Regenerate `uv.lock`; never edit resolved package entries by hand.
- Preserve the existing no-tests baseline explicitly instead of reporting pytest exit 5 as a passing suite.

---

### Task 1: Constrain workspace metadata and documentation

**Files:**
- Modify: `pyproject.toml:9,30-42`
- Modify: `flash_attention_triton/pyproject.toml:9`
- Modify: `docs/research/2026-08-11-gpu-kernel-harness.md:4,22-25`
- Modify: `docs/superpowers/plans/2026-08-11-cuda-12-9-1-environment.md:12-18,122-132`

**Interfaces:**
- Consumes: The approved compatibility contract in `docs/superpowers/specs/2026-08-11-cuda-12-9-1-environment-design.md`.
- Produces: Exact Python and Linux uv metadata that `uv lock` consumes in Task 2.

- [ ] **Step 1: Run the failing metadata contract probe**

```bash
python3 -c 'import tomllib; root=tomllib.load(open("pyproject.toml", "rb")); member=tomllib.load(open("flash_attention_triton/pyproject.toml", "rb")); assert root["project"]["requires-python"] == ">=3.12,<3.14"; assert member["project"]["requires-python"] == ">=3.12,<3.14"; assert root["tool"]["uv"]["environments"] == ["sys_platform == '\''linux'\''"]; assert root["tool"]["uv"]["sources"]["torch"] == [{"index": "pytorch-cu129", "marker": "sys_platform == '\''linux'\''"}]'
```

Expected: FAIL while either workspace manifest still declares an upper bound broader than `<3.14`.

- [ ] **Step 2: Apply the minimal metadata change**

Set both `requires-python` declarations to `>=3.12,<3.14`. Add the following to `[tool.uv]` and narrow the existing Torch source marker:

```toml
[tool.uv]
default-groups = ["dev"]
environments = ["sys_platform == 'linux'"]

[tool.uv.sources]
torch = [
    { index = "pytorch-cu129", marker = "sys_platform == 'linux'" },
]
```

- [ ] **Step 3: Re-run the metadata contract probe**

Run the Step 1 command again.

Expected: exit 0.

- [ ] **Step 4: Update affected decision documents**

Record CPython 3.12–3.13 and Linux-only support in the research note and the original CUDA implementation plan. Remove claims that cu129 resolves Windows wheels; retain Windows only where documenting it as unsupported.

- [ ] **Step 5: Prove the previous lock is stale**

```bash
uv lock --check
```

Expected: FAIL because project metadata changed while `uv.lock` still declares an upper bound broader than `<3.14`.

### Task 2: Regenerate and verify the Linux-only lock

**Files:**
- Modify: `uv.lock`
- Add: `scripts/verify_gpu_stack.py`

**Interfaces:**
- Consumes: The root/member Python interval, uv Linux environment, and Linux-only cu129 source from Task 1.
- Produces: A consistent lockfile that installs Torch 2.11.0+cu129 only for Linux and excludes Python 3.14 from the project contract.

- [ ] **Step 1: Regenerate the lockfile**

```bash
uv lock
```

Expected: resolution succeeds without manually editing package records.

- [ ] **Step 2: Verify metadata and lock identities**

```bash
uv lock --check
python3 -c 'import tomllib; lock=tomllib.load(open("uv.lock", "rb")); assert {part.strip() for part in lock["requires-python"].split(",")} == {">=3.12", "<3.14"}'
rg -q 'version = "2.11.0\+cu129"' uv.lock
rg -q 'https://download.pytorch.org/whl/cu129' uv.lock
```

Expected: all commands exit 0 regardless of uv's harmless whitespace normalization of the Python interval.

- [ ] **Step 3: Verify Linux sync and runtime imports**

```bash
uv sync --locked
uv run --locked python scripts/verify_gpu_stack.py
```

Expected: the environment is synchronized; the H100 preflight, Triton result assertion, and explicit CuTe kernel launch succeed.

- [ ] **Step 4: Verify unsupported contracts are rejected**

```bash
uv sync --locked --dry-run --python-platform x86_64-pc-windows-msvc
uv sync --locked --dry-run --python 3.14
```

Expected: both commands fail. The first must report that the lock/project environment does not support Windows; the second must report that Python 3.14 is incompatible with the project-level `requires-python = ">=3.12,<3.14"` contract, independent of wheel availability.

### Task 3: Validate, publish, and re-review PR #3

**Files:**
- Verify: all files changed since `main`
- GitHub write: PR #3 top-level modification summary comment
- GitHub write: PR #3 top-level Standards/Spec re-review comment

**Interfaces:**
- Consumes: The verified metadata and lockfile from Tasks 1–2.
- Produces: Two follow-up commits on `setup-harness`, an updated remote branch, and auditable PR comments.

- [ ] **Step 1: Run repository quality checks**

```bash
uv run --locked ruff check .
uv run --locked ty check
git diff --check
uv run --locked pytest -q
```

Expected: Ruff, ty, and diff check exit 0. Pytest exits 5 with `no tests ran`; record this as the unchanged baseline, not a passing suite.

- [ ] **Step 2: Commit implementation files**

```bash
git add pyproject.toml flash_attention_triton/pyproject.toml uv.lock scripts/verify_gpu_stack.py docs/research/2026-08-11-gpu-kernel-harness.md docs/superpowers/specs/2026-08-11-cuda-12-9-1-environment-design.md docs/superpowers/plans/2026-08-11-cuda-12-9-1-environment.md docs/superpowers/plans/2026-08-12-linux-python-contract.md
git diff --cached --check
git commit -m "build: constrain GPU workspace to Linux"
```

Expected: commit succeeds with only the selected implementation, lock, and documentation files.

- [ ] **Step 3: Push the PR branch**

```bash
git push origin setup-harness
```

Expected: `origin/setup-harness` advances to local `HEAD`.

- [ ] **Step 4: Post the modification summary**

Post a top-level PR #3 comment listing the Python upper bound, uv Linux environment, Linux-only cu129 source, lock regeneration, and exact validation commands/results.

- [ ] **Step 5: Re-run two-axis review**

Use `code-review` against `main...HEAD`. Run Standards and Spec reviews independently, validate every finding locally, and preserve their separate headings and counts.

- [ ] **Step 6: Post and verify the re-review**

Post the two-axis result as a second new top-level PR #3 comment. Fetch both new comments from GitHub and verify their IDs, author, and bodies. Confirm `git status -sb` is clean and local `HEAD` equals `origin/setup-harness`.
