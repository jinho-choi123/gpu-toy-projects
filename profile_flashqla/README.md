# Profile FlashQLA

Profile the GDN `chunk_gated_delta_rule` forward path with FlashQLA or FLA
Triton.

Requires Python 3.13, an NVIDIA SM90 GPU (H100/H200), Nsight Systems, and
Nsight Compute. Correctness is verified separately in `tests/test_flashqla.py`.

## Setup

Run from the repository root:

```bash
uv sync
mkdir -p profile_flashqla/profiles
```

## Workload

Defaults: `B=1`, `T=16384`, `H=16`, `K=V=128`, BF16 Q/K/V, FP32
gate/beta/state, `cuda:0`, seed 42, 10 warmups, and 1 measured forward.

The workload runs under `torch.inference_mode()` with final-state output and
in-kernel Q/K L2 normalization enabled. Inputs are allocated once and reused.

Override `--batch-size`, `--seq-len`, `--num-heads`, `--seed`, `--warmup`,
or `--iterations`. Run either script with `--help` for details.

## Direct execution

```bash
uv run python profile_flashqla/scripts/profile_gdn_flash_qla.py
uv run python profile_flashqla/scripts/profile_gdn_fla_triton.py
```

## Nsight Systems

FlashQLA:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=profile_flashqla/profiles/flash_qla_b1_t16384_h16_nsys \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_flash_qla.py \
  --iterations 10
```

FLA Triton:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=profile_flashqla/profiles/fla_triton_b1_t16384_h16_nsys \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_fla_triton.py \
  --iterations 10
```

Capture contains `gdn_forward` with one `iteration_N` range per forward.
Input allocation, JIT compilation, and warmup occur before capture.

## Nsight Compute

FlashQLA:

```bash
ncu \
  --set basic \
  --nvtx \
  --nvtx-include gdn_forward \
  --profile-from-start=off \
  --output=profile_flashqla/profiles/flash_qla_b1_t16384_h16_ncu \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_flash_qla.py \
  --iterations 1
```

FLA Triton:

```bash
ncu \
  --set basic \
  --nvtx \
  --nvtx-include gdn_forward \
  --profile-from-start=off \
  --output=profile_flashqla/profiles/fla_triton_b1_t16384_h16_ncu \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_fla_triton.py \
  --iterations 1
```

Use `--set detailed` for deeper analysis. Keep `--iterations 1` because NCU
may replay each captured kernel.
