# Measure with PyTorch CUDA Graphs

Latency benchmarks and Nsight profiles replay graphs captured directly with `torch.cuda.CUDAGraph`; Triton benchmarking helpers are not used. Benchmarks flush L2 outside the timed interval, profiles replay a graph containing one operator invocation, and memory reporting separates ordinary algorithm peak from the graph-private pool footprint. This removes Python launch overhead without mistaking graph buffer reuse for the operator's memory requirement.
