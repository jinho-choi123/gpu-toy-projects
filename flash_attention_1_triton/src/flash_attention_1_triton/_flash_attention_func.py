# ruff: noqa: F722
"""Fixed-length FlashAttention 1 interface."""

import math
from collections.abc import Callable
from typing import cast

import torch
from beartype import beartype
from jaxtyping import BFloat16, Float16, jaxtyped
from torch import Tensor

from . import _flash_attention_kernel


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
        q (Tensor): Query tensor shaped ``[batch, query_length, heads, head_dim]``.
        k (Tensor): Key tensor shaped ``[batch, key_length, heads, head_dim]``.
        v (Tensor): Value tensor with the same shape as ``k``.
        causal (bool): Whether to apply a bottom-right-aligned causal mask. When ``True``,
            ``query_length`` must not exceed ``key_length``.
        softmax_scale (float | None): Positive finite scale applied before softmax. Defaults to
            ``1 / sqrt(head_dim)``.

    Returns:
        Tensor: Contiguous attention output shaped like ``q``, with the same dtype
            and device as ``q``.

    Raises:
        ValueError: If device, dtype, dimensions, strides, or scale are unsupported.
        NotImplementedError: If gradients are requested.

    Note:
        ``q``, ``k``, and ``v`` must be CUDA tensors on the same device with the same
        FP16 or BF16 dtype. They must have the same number of heads, a head dimension
        of 32, 64, or 128, positive dimensions, and a contiguous final dimension.
        This launcher is forward-only. Use ``torch.no_grad()`` when inputs require
        gradients; backward is not implemented.
    """
    if not all(x.is_cuda for x in (q, k, v)):
        raise ValueError("q, k, and v must be CUDA tensors")
    if k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    batch, query_length, heads, head_dim = q.shape
    key_length = k.shape[1]
    if min(batch, query_length, key_length, heads) <= 0:
        raise ValueError("All dimensions must be positive")
    if head_dim not in (32, 64, 128):
        raise ValueError("head_dim must be 32, 64, or 128")
    if any(x.stride(-1) != 1 for x in (q, k, v)):
        raise ValueError("The final dimension of q, k, and v must be contiguous")
    if causal and query_length > key_length:
        raise ValueError("Causal query_length must not exceed key_length")
    scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be positive and finite")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise NotImplementedError("FlashAttention backward is not implemented; use torch.no_grad()")

    # Keep the iteratively updated output and row statistics in FP32 in HBM.
    output = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    row_max = torch.full(
        (batch, query_length, heads), -math.inf, dtype=torch.float32, device=q.device
    )
    row_sum = torch.zeros((batch, query_length, heads), dtype=torch.float32, device=q.device)
    # ponytail: fixed tiles and batch/head parallelism; tune after the kernel is correct.
    with torch.cuda.device(q.device):
        # JIT launch accepts Python values; tl.tensor/constexpr describe the compiled kernel.
        launch = cast(
            Callable[..., None], _flash_attention_kernel._flash_attention_forward[(batch, heads)]
        )
        launch(
            q,
            k,
            v,
            output,
            row_max,
            row_sum,
            scale,
            Q_STRIDES=q.stride(),
            K_STRIDES=k.stride(),
            V_STRIDES=v.stride(),
            O_STRIDES=output.stride(),
            STATE_STRIDES=row_max.stride(),
            QUERY_LENGTH=query_length,
            KEY_LENGTH=key_length,
            HEAD_DIM=head_dim,
            BLOCK_Q=32,
            BLOCK_K=32,
            CAUSAL=causal,
            num_warps=4,
        )
    return output.to(dtype=q.dtype)
