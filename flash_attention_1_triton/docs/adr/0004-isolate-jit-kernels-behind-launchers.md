# Isolate JIT kernels behind Python launchers

User-owned `@triton.jit` kernels remain private behind project-owned forward and backward launchers. The launchers fix the saved-state contract to an output plus FP32 log-sum-exp values and insulate the public autograd API, tests, benchmarks, and profiles from kernel launch signatures and tuning parameters.
