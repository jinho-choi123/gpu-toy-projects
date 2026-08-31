# GDN Profiling

This context defines the workload used to compare Gated DeltaNet forward implementations under GPU profiling.

## Language

**GDN forward workload**:
A single forward execution of the `chunk_gated_delta_rule` operation with equivalent inputs and outputs across implementations.
_Avoid_: GDN model, end-to-end GDN

**Long-context workload**:
The canonical GDN forward workload with batch size 1, sequence length 16,384, 16 heads, and key and value head dimensions of 128.
_Avoid_: Large test, 16K test
