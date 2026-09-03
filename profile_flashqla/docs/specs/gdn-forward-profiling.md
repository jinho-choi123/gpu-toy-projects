# GDN Forward Profiling Specification

## Objective

Provide reproducible, forward-only Nsight Systems and Nsight Compute entry
points and a fixed profiling sweep for comparing CUDA Graph replays of the
FlashQLA and FLA Triton GDN forward implementations across strong- and
weak-decay gate mixes. The scripts generate profiles; numerical correctness
remains the responsibility of `tests/test_flashqla.py`.

## Terminology

- **GDN forward workload**: one `chunk_gated_delta_rule` forward execution with equivalent inputs and outputs across implementations.
- **Measured graph replay**: one `cuda_graph.replay()` of a graph that captured the GDN forward workload after eager warmup.
- **Long-context workload**: batch size 1, sequence length 16,384, 16 heads, and key/value head dimensions of 128.
- **Strong-decay head**: a head whose gate uses per-token `logsigmoid(randn)` values.
- **Weak-decay head**: a head whose gate is `0` for every token.

## Scope

- Capture and measure forward-only CUDA Graph replays; do not execute backward or an optimizer step.
- Compare the two implementations under identical inputs and runtime settings.
- Produce raw `.nsys-rep` and `.ncu-rep` reports and CUDA memory snapshots under `profile_flashqla/profiles/`.
- Keep raw profile reports and memory snapshots out of Git.
- Do not add numerical validation to profiling scripts.
- Do not add automated tests for the profiling scripts.
- Do not optimize either implementation as part of this work.

## Workload

| Setting | Default |
| --- | --- |
| Batch size | 1 |
| Sequence length | 16,384 |
| Number of heads | 16 |
| Strong-decay head ratio | 1.0 |
| Key head dimension | 128, fixed |
| Value head dimension | 128, fixed |
| Q/K/V dtype | BF16, fixed |
| Gate, beta, and initial-state dtype | FP32, fixed |
| Device | `cuda:0`, fixed |
| Seed | 42 |
| Eager warmup | 10 iterations, fixed |
| Graph replay warmup | 10 iterations, fixed |
| Final state | enabled |
| Q/K L2 normalization in kernel | enabled |
| Execution mode | `torch.inference_mode()` |

For `H` heads and a configured strong-decay head ratio, compute the effective
strong-decay head count as `N = int(H * ratio + 0.5)`. This rounds to the
nearest whole head with ties rounded upward. The first `N` heads are
strong-decay heads, and all remaining heads are weak-decay heads.

Generate the complete FP32 random gate tensor and apply `logsigmoid` before
setting the weak-decay head suffix to zero. This preserves random-number
consumption for beta and the initial state when only the ratio changes. Create
all inputs once outside profiler capture and reuse the same tensors for eager
warmup, memory profiling, graph capture, graph replay warmup, and measured graph
replays. The tensors do not require gradients.

## Implementation selection

Both entry points call FLA's public `chunk_gated_delta_rule` API through the shared `profile_flashqla.utils.run_forward` helper. Set `FLA_FLASH_QLA=1` for the FlashQLA entry point and `FLA_FLASH_QLA=0` for the FLA Triton entry point. Existing dispatch tests are the backend-selection guard; the profiling path does not duplicate that verification.

The supported profiling hardware is SM90 only. Before allocating the workload, reject unavailable CUDA and capabilities other than `(9, 0)` on the fixed `cuda:0` device.

## CLI

Expose only these options:

- `--batch-size`, default `1`
- `--seq-len`, default `16384`
- `--num-heads`, default `16`
- `--strong-decay-head-ratio`, default `1.0`
- `--seed`, default `42`
- `--iterations`, default `1`

Batch size, sequence length, number of heads, and iterations must be positive.
The strong-decay head ratio must be in the inclusive range `[0.0, 1.0]` and
uses the effective-head rounding rule defined under Workload. Device, dtype,
head dimensions, and both warmup counts are deliberately fixed. FlashQLA
requires BF16/FP16 inputs and 128-dimensional key/value heads; this workload
fixes BF16 and 128 to prevent configuration-induced backend fallback.

## Capture protocol

Under `torch.inference_mode()`:

1. Run 10 eager forward warmups on a side CUDA stream.
2. Make the current stream wait for the warmup stream.
3. Capture one `run_forward(gdn_input)` call in a `torch.cuda.CUDAGraph`.
4. Run 10 graph replay warmups and synchronize `cuda:0`.
5. Start capture through the CUDA Profiler API.
6. Wrap all measured graph replays in an outer `gdn_forward` NVTX range and each `cuda_graph.replay()` in `iteration_N`.
7. Synchronize before stopping the profiler.

Graph capture, input allocation, eager warmup, eager memory profiling, and
graph replay warmup are outside the measured profiler range.

- Nsight Systems: ten measured graph replays, CUDA and NVTX tracing, CPU sampling disabled, and CUDA Graph nodes traced.
- Nsight Compute: one measured graph replay, graph profiling by node, and the `basic` set by default; document `detailed` as an opt-in deeper analysis.
- The scripts may also run directly without an attached Nsight tool.

## Runtime information

Keep Loguru's default stderr handler. Before eager warmup, log the backend,
workload shape, configured strong-decay head ratio, effective strong-decay head
count, fixed dtype and head dimensions, fixed device, seed, `eager_warmup`,
`graph_warmup`, and measured iteration count. Do not create a sidecar metadata
file.

## Files

```text
profile_flashqla/
  README.md
  pyproject.toml
  scripts/
    profile_gdn_flash_qla.py
    profile_gdn_fla_triton.py
    sweep_profiles.sh
  src/profile_flashqla/
    gdn_profile.py
    utils/
      __init__.py
      gdn.py
  tests/
    test_flashqla.py
```

The two Python scripts are thin implementation selectors. `gdn_profile.py`
owns CLI parsing, deterministic input creation, SM90 validation, logging, eager
warmup, memory profiling, CUDA Graph capture/replay, synchronization, NVTX
ranges, and profiler lifecycle. `sweep_profiles.sh` owns the fixed profiling
matrix, profiler commands, artifact naming, completed-job detection, and
snapshot finalization. `utils/gdn.py` owns the shared `GdnInput` contract and
pure `run_forward` helper used by both the runner and correctness tests.

## Profiling sweep

`scripts/sweep_profiles.sh` runs this fixed matrix sequentially:

- Backends: `flash_qla` and `fla_triton`
- Sequence lengths: `16384`, `32768`, and `65536`
- Strong-decay head ratios: `0.0`, `0.5`, and `1.0`, corresponding to
  `sdh0`, `sdh8`, and `sdh16` for the fixed 16-head sweep
- Profilers: Nsight Systems and Nsight Compute

The Cartesian product contains exactly 36 jobs. Nsight Systems uses 10 measured
graph replays per job, and Nsight Compute uses one. The script supports
`--dry-run` to print the matrix without executing profilers, `--sudo-ncu` to
run Nsight Compute with its root-owned temporary directory, and `--force` to
overwrite existing artifacts.

Without `--force`, a job is complete and skipped only when both its profiler
report and profiler-specific memory snapshot exist. If the report exists, the
profiler-specific snapshot is absent, and the unsuffixed source snapshot exists,
the script finalizes that snapshot and continues. A report with no snapshot, or
a snapshot with no matching report, stops the sweep and requires an explicit
`--force` rerun. This is artifact-aware restart behavior; it does not guarantee
automatic recovery from every profiler interruption.

## Correctness test

Keep the existing small correctness test and the unmarked `test_flash_qla_matches_fla_triton_long_context` test. Both use the shared `GdnInput` and `run_forward` contract. The long-context test uses the canonical workload and the existing output/final-state comparison ratio of `0.008`.

## Report names

Artifact names encode the effective strong-decay head count as `sdh{N}`,
rather than the raw ratio. Direct profiler commands use explicit prefixes
without automatic timestamps or overwrite behavior:

```text
profile_flashqla/profiles/flash_qla_b1_t16384_h16_sdh16_nsys
profile_flashqla/profiles/fla_triton_b1_t16384_h16_sdh16_nsys
profile_flashqla/profiles/flash_qla_b1_t16384_h16_sdh16_ncu
profile_flashqla/profiles/fla_triton_b1_t16384_h16_sdh16_ncu
```

Direct Python runs write
`{backend}_b{B}_t{T}_h{H}_sdh{N}_memory_snapshot.pickle`. The sweep preserves
that snapshot per profiler as
`{backend}_b{B}_t{T}_h{H}_sdh{N}_{profiler}_memory_snapshot.pickle`. Users
choose a new suffix for subsequent manual profiler runs; the sweep uses its
deterministic names and the restart rules above.

## Acceptance criteria

1. Existing backend dispatch and small numerical comparison tests pass on SM90 through the shared forward helper.
2. The unmarked long-context comparison test passes on SM90.
3. Both entry points run directly with a reduced CLI workload and accept
   `--strong-decay-head-ratio` values in the inclusive range `[0.0, 1.0]`;
   values outside that range are rejected.
4. The effective strong-decay head count uses `N = int(H * ratio + 0.5)`, the
   first `N` gate heads use `logsigmoid(randn)`, and all remaining gate heads
   are zero.
5. With an identical seed and shape, changing only the strong-decay head ratio
   does not change the generated beta or initial-state tensors.
6. Runtime logging includes both the configured ratio and effective
   strong-decay head count.
7. The default Nsight Systems commands create two comparable GPU/NVTX
   timelines, one per implementation, with exactly ten measured graph replays
   grouped by `iteration_N`.
8. The default Nsight Compute commands attempt `basic` graph-node reports for
   both implementations, while documented `detailed` commands provide optional
   deeper analysis; GPU performance-counter permissions remain an environment
   prerequisite.
9. Graph capture, input allocation, memory profiling, and both warmup phases
   remain outside the measured profiler range.
10. Report and memory-snapshot names contain `sdh{N}` and uniquely identify the
    effective gate mix; sweep snapshots also identify the profiler.
11. A sweep dry run enumerates exactly 36 jobs: two backends by three sequence
    lengths by three ratios by two profilers.
12. A repeated sweep skips only complete report/snapshot pairs, finalizes a
    pending source snapshot only when its matching report exists, and requires
    `--force` for conflicting or orphaned artifacts.
13. No numerical validation kernel appears in a profile capture.
14. No raw Nsight report or memory snapshot is tracked by Git.
