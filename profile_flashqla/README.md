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
gate/beta/state, `cuda:0`, seed 42, `strong_decay_head_ratio=1.0`, and 1
measured graph replay.

Use `--strong-decay-head-ratio` to control the gate mix. The value must be in
the inclusive range `[0.0, 1.0]`. The effective strong-decay head count is
`N = int(H * ratio + 0.5)`, which rounds to the nearest whole head with ties
rounded up. Exactly the first `N` heads use per-token `logsigmoid(randn)` gates;
all remaining heads use gate `0` for every token.

- `1.0`: all heads use strong decay.
- `0.0`: all heads use weak decay.
- A value between `0.0` and `1.0`: the first `N` heads use strong decay and the
  remaining heads use weak decay.

The full random gate tensor is generated before weak-decay heads are set to
zero. With the same seed and shape, changing only this ratio therefore preserves
the random inputs generated afterward, including beta and the initial state.

The workload always uses CUDA Graphs: 10 eager warmups on a side stream,
one graph capture, and 10 graph replay warmups before measurement.

Override `--batch-size`, `--seq-len`, `--num-heads`,
`--strong-decay-head-ratio`, `--seed`, or `--iterations`. Use the same ratio for
both backends when comparing them. Run either script with `--help` for details.

## Direct execution

```bash
uv run python profile_flashqla/scripts/profile_gdn_flash_qla.py
uv run python profile_flashqla/scripts/profile_gdn_fla_triton.py
```

For example, profile an even strong/weak head mix with:

```bash
uv run python profile_flashqla/scripts/profile_gdn_flash_qla.py \
  --strong-decay-head-ratio 0.5
uv run python profile_flashqla/scripts/profile_gdn_fla_triton.py \
  --strong-decay-head-ratio 0.5
```

## Full profiling sweep

Run both backends with Nsight Systems and Nsight Compute for sequence lengths
16K, 32K, and 64K and strong-decay head ratios `0.0`, `0.5`, and `1.0`:

```bash
./profile_flashqla/scripts/sweep_profiles.sh
```

This runs 36 profiling jobs sequentially. Nsight Systems uses 10 measured graph
replays per job, while Nsight Compute uses one. Completed jobs (both the Nsight
report and memory snapshot exist) are skipped, so the same command can resume an
interrupted sweep. If a profiler finished just before an interruption, the next
run finalizes its pending snapshot before continuing. Memory snapshots are
preserved separately as
`{backend}_b{B}_t{T}_h{H}_sdh{N}_{profiler}_memory_snapshot.pickle`, so the
Nsight Systems and Nsight Compute runs do not overwrite each other.

Preview what the sweep would run without running a profiler:

```bash
./profile_flashqla/scripts/sweep_profiles.sh --dry-run
```

If Nsight Compute requires elevated permissions, use `--sudo-ncu`. The script
creates `/tmp/ncu-root` with root-only permissions and uses it as NCU's temporary
directory. It restores ownership of the NCU report and memory snapshot to the
calling user after each job:

```bash
./profile_flashqla/scripts/sweep_profiles.sh --sudo-ncu
```

Use `--force` to overwrite existing reports and snapshots. Options can be
combined, for example `--dry-run --sudo-ncu --force`. A snapshot without its
matching report is treated as an incomplete or direct-run artifact and is not
overwritten unless `--force` is supplied.

## CUDA memory snapshot

Each direct Python run records one warmed-up eager forward. Its snapshot is
written to
`profile_flashqla/profiles/{backend}_b{B}_t{T}_h{H}_sdh{N}_memory_snapshot.pickle`,
where `N` is the effective strong-decay head count.
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
  --output=profile_flashqla/profiles/flash_qla_b1_t16384_h16_sdh16_nsys \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_flash_qla.py \
  --strong-decay-head-ratio 1.0 \
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
  --output=profile_flashqla/profiles/fla_triton_b1_t16384_h16_sdh16_nsys \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_fla_triton.py \
  --strong-decay-head-ratio 1.0 \
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
  --export=profile_flashqla/profiles/flash_qla_b1_t16384_h16_sdh16_ncu \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_flash_qla.py \
  --seq-len 16384 \
  --strong-decay-head-ratio 1.0 \
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
  --export=profile_flashqla/profiles/fla_triton_b1_t16384_h16_sdh16_ncu \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_fla_triton.py \
  --seq-len 16384 \
  --strong-decay-head-ratio 1.0 \
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
  --export=profile_flashqla/profiles/flash_qla_b1_t16384_h16_sdh16_ncu \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_flash_qla.py \
  --seq-len 16384 \
  --strong-decay-head-ratio 1.0 \
  --iterations 1

# FLA Triton
sudo env TMPDIR=/tmp/ncu-root /usr/local/cuda/bin/ncu \
  --set basic \
  --graph-profiling node \
  --nvtx \
  --nvtx-include 'gdn_forward/' \
  --profile-from-start=off \
  --export=profile_flashqla/profiles/fla_triton_b1_t16384_h16_sdh16_ncu \
  .venv/bin/python profile_flashqla/scripts/profile_gdn_fla_triton.py \
  --seq-len 16384 \
  --strong-decay-head-ratio 1.0 \
  --iterations 1
```

Use `--set detailed` for deeper analysis. Keep `--iterations 1` because NCU
may replay each captured kernel.
