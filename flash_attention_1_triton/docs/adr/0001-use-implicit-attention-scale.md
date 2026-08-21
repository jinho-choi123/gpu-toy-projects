# Use an implicit attention scale

The public FlashAttention API always scales scores by `1 / sqrt(head_dim)` and does not accept a caller-provided scale. This deliberately trades compatibility with arbitrary scaling conventions for a smaller API and a single unambiguous correctness and performance contract.
