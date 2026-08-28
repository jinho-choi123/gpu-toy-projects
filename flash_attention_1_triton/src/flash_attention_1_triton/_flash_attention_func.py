# ruff: noqa: F722
"""Fixed-length FlashAttention 1 interface."""

from beartype import beartype
from jaxtyping import BFloat16, Float16, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=beartype)
def flash_attention_func(
    q: (
        Float16[Tensor, "batch query_length heads head_dim"]
        | BFloat16[Tensor, "batch query_length heads head_dim"]
    ),
    k: (
        Float16[Tensor, "batch key_length heads head_dim"]
        | BFloat16[Tensor, "batch key_length heads head_dim"]
    ),
    v: (
        Float16[Tensor, "batch key_length heads head_dim"]
        | BFloat16[Tensor, "batch key_length heads head_dim"]
    ),
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> (
    Float16[Tensor, "batch query_length heads head_dim"]
    | BFloat16[Tensor, "batch query_length heads head_dim"]
):
    """Compute fixed-length FlashAttention 1.

    Args:
        q: Query tensor shaped ``[batch, query_length, heads, head_dim]``.
        k: Key tensor shaped ``[batch, key_length, heads, head_dim]``.
        v: Value tensor with the same shape as ``k``.
        causal: Whether to apply a bottom-right-aligned causal mask. When ``True``,
            ``query_length`` must not exceed ``key_length``.
        softmax_scale: Positive finite scale applied before softmax. Defaults to
            ``1 / sqrt(head_dim)``.

    Returns:
        A contiguous tensor shaped like ``q`` and with the same dtype and device.

    Raises:
        ValueError: If ``causal=True`` and ``query_length > key_length``.
        NotImplementedError: Always, until the FlashAttention implementation is added.

    Note:
        ``q``, ``k``, and ``v`` must be CUDA tensors on the same device with the same
        FP16 or BF16 dtype. They must have the same number of heads, a head dimension
        of 32, 64, or 128, and a contiguous final dimension. The operation supports
        first-order gradients with respect to ``q``, ``k``, and ``v``.
    """
    raise NotImplementedError("flash_attention_func is not implemented yet")
