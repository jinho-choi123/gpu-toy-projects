# Isolate benchmark configurations in worker processes

The benchmark coordinator runs each input configuration in a fresh worker process, where both providers and both phases are measured before the worker exits. The process startup and compilation cost is excluded from latency results; in return, CUDA Graph pools, allocator state, and failures cannot contaminate later configurations.
