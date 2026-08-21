# Separate hardware utilization from effective throughput

Representative profiles use Nsight Compute counters as the source of truth for DRAM bandwidth and compute utilization, while the full benchmark matrix may report conventional effective TFLOP/s from algorithmic work divided by graph-replay latency. Names and schemas keep these values separate so a derived workload rate is never presented as an actual hardware counter.
