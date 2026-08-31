# GDN Forward Profiling Specification

## Objective

Provide reproducible, forward-only Nsight Systems and Nsight Compute entry points for comparing CUDA Graph replays of the FlashQLA and FLA Triton GDN forward implementations. The scripts generate profiles; numerical correctness remains the responsibility of `tests/test_flashqla.py`.

## Terminology

- **GDN forward workload**: one `chunk_gated_delta_rule` forward execution with equivalent inputs and outputs across implementations.
- **Measured graph replay**: one `cuda_graph.replay()` of a graph that captured the GDN forward workload after eager warmup.
- **Long-context workload**: batch size 1, sequence length 16,384, 16 heads, and key/value head dimensions of 128.

## Scope

- Capture and measure forward-only CUDA Graph replays; do not execute backward or an optimizer step.
- Compare the two implementations under identical inputs and runtime settings.
- Produce raw `.nsys-rep` and `.ncu-rep` reports under `profile_flashqla/profiles/`.
- Keep raw profile reports out of Git.
- Do not add numerical validation to profiling scripts.
- Do not add automated tests for the profiling scripts.
- Do not optimize either implementation as part of this work.

## Workload

| Setting | Default |
| --- | --- |
| Batch size | 1 |
| Sequence length | 16,384 |
| Number of heads | 16 |
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

Create inputs once outside profiler capture and reuse the same tensors for eager warmup, graph capture, graph replay warmup, and measured graph replays. The tensors use the same distributions as `tests/test_flashqla.py` and do not require gradients.

## Implementation selection

Both entry points call FLA's public `chunk_gated_delta_rule` API through the shared `profile_flashqla.utils.run_forward` helper. Set `FLA_FLASH_QLA=1` for the FlashQLA entry point and `FLA_FLASH_QLA=0` for the FLA Triton entry point. Existing dispatch tests are the backend-selection guard; the profiling path does not duplicate that verification.

The supported profiling hardware is SM90 only. Before allocating the workload, reject unavailable CUDA and capabilities other than `(9, 0)` on the fixed `cuda:0` device.

## CLI

Expose only these options:

- `--batch-size`, default `1`
- `--seq-len`, default `16384`
- `--num-heads`, default `16`
- `--seed`, default `42`
- `--iterations`, default `1`

Batch size, sequence length, number of heads, and iterations must be positive. Device, dtype, head dimensions, and both warmup counts are deliberately fixed. FlashQLA requires BF16/FP16 inputs and 128-dimensional key/value heads; this workload fixes BF16 and 128 to prevent configuration-induced backend fallback.

## Capture protocol

Under `torch.inference_mode()`:

1. Run 10 eager forward warmups on a side CUDA stream.
2. Make the current stream wait for the warmup stream.
3. Capture one `run_forward(gdn_input)` call in a `torch.cuda.CUDAGraph`.
4. Run 10 graph replay warmups and synchronize `cuda:0`.
5. Start capture through the CUDA Profiler API.
6. Wrap all measured graph replays in an outer `gdn_forward` NVTX range and each `cuda_graph.replay()` in `iteration_N`.
7. Synchronize before stopping the profiler.

Graph capture, input allocation, eager warmup, and graph replay warmup are outside the measured profiler range.

- Nsight Systems: ten measured graph replays, CUDA and NVTX tracing, CPU sampling disabled, and CUDA Graph nodes traced.
- Nsight Compute: one measured graph replay, graph profiling by node, and the `basic` set by default; document `detailed` as an opt-in deeper analysis.
- The scripts may also run directly without an attached Nsight tool.

## Runtime information

Keep Loguru's default stderr handler. Before eager warmup, log the backend, workload shape, fixed dtype and head dimensions, fixed device, seed, `eager_warmup`, `graph_warmup`, and measured iteration count. Do not create a sidecar metadata file.

## Files

```text
profile_flashqla/
  README.md
  pyproject.toml
  scripts/
    profile_gdn_flash_qla.py
    profile_gdn_fla_triton.py
  src/profile_flashqla/
    gdn_profile.py
    utils/
      __init__.py
      gdn.py
  tests/
    test_flashqla.py
```

The two scripts are thin implementation selectors. `gdn_profile.py` owns CLI parsing, deterministic input creation, SM90 validation, logging, eager warmup, CUDA Graph capture/replay, synchronization, NVTX ranges, and profiler lifecycle. `utils/gdn.py` owns the shared `GdnInput` contract and pure `run_forward` helper used by both the runner and correctness tests.

## Correctness test

Keep the existing small correctness test and the unmarked `test_flash_qla_matches_fla_triton_long_context` test. Both use the shared `GdnInput` and `run_forward` contract. The long-context test uses the canonical workload and the existing output/final-state comparison ratio of `0.008`.

## Report names

README examples use explicit prefixes without automatic timestamps or overwrite behavior:

```text
profile_flashqla/profiles/flash_qla_b1_t16384_h16_nsys
profile_flashqla/profiles/fla_triton_b1_t16384_h16_nsys
profile_flashqla/profiles/flash_qla_b1_t16384_h16_ncu
profile_flashqla/profiles/fla_triton_b1_t16384_h16_ncu
```

Users choose a new suffix for subsequent runs.

## Acceptance criteria

1. Existing backend dispatch and small numerical comparison tests pass on SM90 through the shared forward helper.
2. The unmarked long-context comparison test passes on SM90.
3. Both entry points run directly with a reduced CLI workload.
4. The default Nsight Systems commands create two comparable GPU/NVTX timelines, one per implementation, with exactly ten measured graph replays grouped by `iteration_N`.
5. The default Nsight Compute commands attempt `basic` graph-node reports for both implementations, while documented `detailed` commands provide optional deeper analysis; GPU performance-counter permissions remain an environment prerequisite.
6. Graph capture, input allocation, and both warmup phases remain outside the measured profiler range.
7. No numerical validation kernel appears in a profile capture.
8. No raw Nsight report is tracked by Git.
