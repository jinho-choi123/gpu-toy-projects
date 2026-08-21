# FlashAttention-1 Evaluation

This context defines the language used to compare a hand-written FlashAttention-1 operator with a straightforward PyTorch implementation.

## Language

**FlashAttention operator**:
The attention operator under evaluation, with custom forward and backward computation written for this project.
_Avoid_: Triton version, optimized version

**Reference attention**:
The straightforward PyTorch attention implementation that serves as both the same-dtype correctness oracle and the sole performance baseline.
_Avoid_: Naive Torch, baseline implementation

**Benchmark**:
A repeated experiment that compares latency or peak memory for the FlashAttention operator and reference attention under the same input conditions.
_Avoid_: Profile, timing test

**Profile**:
A hardware-counter capture that describes how one operator execution uses memory bandwidth and compute resources.
_Avoid_: Benchmark, performance test

**Benchmark result**:
The terminal and machine-readable record produced by a benchmark, including raw measurements and the comparison between operators.
_Avoid_: Performance test result

**Dense attention input**:
An attention input whose batch members share one sequence length and whose token positions are all valid.
_Avoid_: Padded input, ragged input

**Supported case**:
A dense self-attention configuration within the project's declared modes, numeric formats, and head dimensions.
_Avoid_: Arbitrary attention input

**Numerical agreement**:
Acceptance of a FlashAttention result when its difference from reference attention stays within the declared tolerance for reorderings of reduced floating-point operations.
_Avoid_: Exact equality, bitwise equality

**Steady-state latency**:
GPU execution time measured by replaying a captured graph after compilation and warm-up, with L2 flushing outside the timed interval.
_Avoid_: First-call latency, compile time, hot-cache latency

**Hardware utilization**:
Memory or compute usage measured from GPU hardware counters and expressed as an achieved rate or percentage of peak.
_Avoid_: Effective throughput, theoretical utilization

**Effective throughput**:
Algorithmic work assigned to an attention case divided by its steady-state latency, independent of the instructions actually executed by the GPU.
_Avoid_: Hardware utilization, actual hardware FLOP rate

**Profile case**:
A representative supported case selected for detailed hardware-counter capture rather than inclusion in the full benchmark sweep.
_Avoid_: Benchmark case

**Raw profile**:
The native Nsight Systems or Nsight Compute report retained as the source of truth for a profile run.
_Avoid_: Profile summary, benchmark result

**Profile summary**:
The single tabular projection derived from raw profiles for comparison and analysis.
_Avoid_: Raw profile

**Graph pool footprint**:
The additional resident GPU memory owned by a captured CUDA Graph, reported separately from an operator's ordinary execution peak.
_Avoid_: Algorithm peak memory

**Run manifest**:
The machine-readable record of the environment and settings that produced a set of benchmark results or profiles.
_Avoid_: Benchmark results, profile summary
