# ruff: noqa: F722
"""Variable-length FlashAttention 1 interface."""

from beartype import beartype
from jaxtyping import BFloat16, Float16, Int32, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=beartype)
def flash_attention_varlen_func(
    q: (
        Float16[Tensor, "total_queries heads head_dim"]
        | BFloat16[Tensor, "total_queries heads head_dim"]
    ),
    k: (
        Float16[Tensor, "total_keys heads head_dim"] | BFloat16[Tensor, "total_keys heads head_dim"]
    ),
    v: (
        Float16[Tensor, "total_keys heads head_dim"] | BFloat16[Tensor, "total_keys heads head_dim"]
    ),
    *,
    cu_seqlens_q: Int32[Tensor, " batch_plus_one"],
    cu_seqlens_k: Int32[Tensor, " batch_plus_one"],
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> (
    Float16[Tensor, "total_queries heads head_dim"]
    | BFloat16[Tensor, "total_queries heads head_dim"]
):
    """Compute variable-length FlashAttention 1 over packed sequences.

    Args:
        q: Packed query tensor shaped ``[total_queries, heads, head_dim]``.
        k: Packed key tensor shaped ``[total_keys, heads, head_dim]``.
        v: Packed value tensor with the same shape as ``k``.
        cu_seqlens_q: CUDA int32 cumulative query lengths shaped ``[batch + 1]``.
        cu_seqlens_k: CUDA int32 cumulative key lengths shaped ``[batch + 1]``.
        max_seqlen_q: Exact maximum query length in the batch.
        max_seqlen_k: Exact maximum key length in the batch.
        causal: Whether to apply a bottom-right-aligned causal mask.
        softmax_scale: Positive finite scale applied before softmax. Defaults to
            ``1 / sqrt(head_dim)``.

    Returns:
        A contiguous tensor shaped like ``q`` and with the same dtype and device.

    Raises:
        NotImplementedError: Always, until the FlashAttention implementation is added.

    Note:
        ``q``, ``k``, and ``v`` must be CUDA FP16 or BF16 tensors on the same device.
        They must have the same number of heads, a head dimension of 32, 64, or 128,
        and a contiguous final dimension. Each encoded sequence must be non-empty.
        The operation supports first-order gradients with respect to ``q``, ``k``,
        and ``v``.
    """
    raise NotImplementedError("flash_attention_varlen_func is not implemented yet")
