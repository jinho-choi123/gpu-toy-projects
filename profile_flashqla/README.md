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
gate/beta/state, `cuda:0`, seed 42, and 1 measured graph replay.

The workload always uses CUDA Graphs: 10 eager warmups on a side stream,
one graph capture, and 10 graph replay warmups before measurement.

Override `--batch-size`, `--seq-len`, `--num-heads`, `--seed`, or
`--iterations`. Run either script with `--help` for details.

## Direct execution

```bash
uv run python profile_flashqla/scripts/profile_gdn_flash_qla.py
uv run python profile_flashqla/scripts/profile_gdn_fla_triton.py
```

## CUDA memory snapshot

Each run records one warmed-up eager forward and writes a snapshot to
`profile_flashqla/profiles/{backend}_b{B}_t{T}_h{H}_memory_snapshot.pickle`.
Open [PyTorch Memory Visualizer](https://pytorch.org/memory_viz), then drag and
drop the snapshot file to inspect allocation and free events over time.

For a numeric comparison, use `memory_forward_peak_delta_mib` from the command
output. Memory recording ends before CUDA Graph capture and the Nsight profiling
range, so it is excluded from the Nsight measurement.

## Nsight Systems

FlashQLA:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cuda-graph-trace=node \
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
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=profile_flashqla/profiles/fla_triton_b1_t16384_h16_nsys \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_fla_triton.py \
  --iterations 10
```

Capture contains `gdn_forward` with one `iteration_N` range per measured graph replay.
Input allocation, JIT compilation, and warmup occur before capture.

## Nsight Compute

FlashQLA:

```bash
ncu \
  --set basic \
  --graph-profiling node \
  --nvtx \
  --nvtx-include 'gdn_forward/' \
  --profile-from-start=off \
  --export=profile_flashqla/profiles/flash_qla_b1_t16384_h16_ncu \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_flash_qla.py \
  --iterations 1
```

FLA Triton:

```bash
ncu \
  --set basic \
  --graph-profiling node \
  --nvtx \
  --nvtx-include 'gdn_forward/' \
  --profile-from-start=off \
  --export=profile_flashqla/profiles/fla_triton_b1_t16384_h16_ncu \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_fla_triton.py \
  --iterations 1
```

If NCU reports `ERR_NVGPUCTRPERM`, create a root-owned temporary directory
for its lock file and run the FlashQLA profile with elevated privileges:

```bash
sudo install -d -m 700 /tmp/ncu-root
# FlashQLA
sudo env TMPDIR=/tmp/ncu-root /usr/local/cuda/bin/ncu \
  --set basic \
  --graph-profiling node \
  --nvtx \
  --nvtx-include 'gdn_forward/' \
  --profile-from-start=off \
  --export=profile_flashqla/profiles/flash_qla_b1_t16384_h16_ncu \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_flash_qla.py \
  --iterations 1

# FLA Triton
sudo env TMPDIR=/tmp/ncu-root /usr/local/cuda/bin/ncu \
  --set basic \
  --graph-profiling node \
  --nvtx \
  --nvtx-include 'gdn_forward/' \
  --profile-from-start=off \
  --export=profile_flashqla/profiles/fla_triton_b1_t16384_h16_ncu \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_fla_triton.py \
  --iterations 1
```

Use `--set detailed` for deeper analysis. Keep `--iterations 1` because NCU
may replay each captured kernel.

## Sequence length 65536

### Nsight Systems

FlashQLA:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=profile_flashqla/profiles/flash_qla_b1_t65536_h16_nsys \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_flash_qla.py \
  --seq-len 65536 \
  --iterations 10
```

FLA Triton:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=profile_flashqla/profiles/fla_triton_b1_t65536_h16_nsys \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_fla_triton.py \
  --seq-len 65536 \
  --iterations 10
```

### Nsight Compute (`sudo`)

```bash
sudo install -d -m 700 /tmp/ncu-root

# FlashQLA
sudo env TMPDIR=/tmp/ncu-root /usr/local/cuda/bin/ncu \
  --set basic \
  --graph-profiling node \
  --nvtx \
  --nvtx-include 'gdn_forward/' \
  --profile-from-start=off \
  --export=profile_flashqla/profiles/flash_qla_b1_t65536_h16_ncu \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_flash_qla.py \
  --seq-len 65536 \
  --iterations 1

# FLA Triton
sudo env TMPDIR=/tmp/ncu-root /usr/local/cuda/bin/ncu \
  --set basic \
  --graph-profiling node \
  --nvtx \
  --nvtx-include 'gdn_forward/' \
  --profile-from-start=off \
  --export=profile_flashqla/profiles/fla_triton_b1_t65536_h16_ncu \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_fla_triton.py \
  --seq-len 65536 \
  --iterations 1
```
