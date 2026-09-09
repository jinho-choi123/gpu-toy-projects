"""Packed API tests with independent attention for each sequence."""

import pytest
import torch
from loguru import logger
from reference import assert_output_and_gradients, attention_varlen, make_inputs

from flash_attention_1_triton import flash_attention_varlen_func

pytestmark = pytest.mark.usefixtures("cuda")


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize(
    "q_lengths,k_lengths,heads,causal,scale,strided",
    [
        pytest.param([1], [1], 1, False, None, False, id="single-token"),
        pytest.param([1, 63, 65], [1, 63, 65], 3, False, None, False, id="uneven-batch"),
        pytest.param([64, 1], [64, 1], 2, True, None, False, id="causal-length-64"),
        pytest.param([65, 1, 63], [65, 1, 63], 3, True, 0.3, True, id="causal-strided"),
        pytest.param([127, 1], [129, 65], 2, False, 0.2, False, id="cross-attention"),
        pytest.param([128, 1, 65], [257, 65, 127], 3, True, None, False, id="causal-bottom-right"),
        pytest.param([1, 1], [65, 3], 2, True, 0.3, False, id="causal-single-queries"),
        pytest.param([129, 1], [65, 3], 2, False, None, True, id="longer-query-strided"),
    ],
)
def test_output_and_gradients(
    dtype: torch.dtype,
    head_dim: int,
    q_lengths: list[int],
    k_lengths: list[int],
    heads: int,
    causal: bool,
    scale: float | None,
    strided: bool,
) -> None:
    """Match per-sequence outputs and gradients without mixing packed sequences.

    Args:
        dtype (torch.dtype): Input dtype, torch.float16 or torch.bfloat16.
        head_dim (int): Per-head dimension: 32, 64, or 128.
        q_lengths (list[int]): Non-empty query sequence lengths in packing order.
        k_lengths (list[int]): Corresponding non-empty key and value sequence lengths.
        heads (int): Shared Q/K/V head count.
        causal (bool): Enable a bottom-right causal mask independently for each sequence.
        scale (float | None): Custom positive logit scale or the API default.
        strided (bool): Test packed inputs with gaps between rows.

    Returns:
        None: Complete if packed output and Q/K/V gradients match independent references.
    """
    inputs = make_inputs(
        [(sum(lengths), heads, head_dim) for lengths in (q_lengths, k_lengths, k_lengths)],
        dtype,
        strided=strided,
    )
    logger.info("Preparing independent Flash Attention reference inputs")
    reference_inputs = tuple(x.detach().requires_grad_() for x in inputs)
    logger.info("Building CUDA cumulative lengths")
    cu_q, cu_k = (
        torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
        for lengths in (q_lengths, k_lengths)
    )
    expected = attention_varlen(
        *reference_inputs,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        causal=causal,
        softmax_scale=scale,
    )
    logger.info("Calling flash_attention_varlen_func")
    output = flash_attention_varlen_func(
        *inputs,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        causal=causal,
        softmax_scale=scale,
    )
    assert_output_and_gradients(output, expected, inputs, reference_inputs)


def test_causal_rejects_one_longer_query(dtype: torch.dtype) -> None:
    """Validate each sequence even when batch maxima and total lengths look valid.

    Args:
        dtype (torch.dtype): Input dtype, torch.float16 or torch.bfloat16.

    Returns:
        None: Complete only if the API rejects the second sequence's longer causal query.
    """
    inputs = make_inputs([(5, 1, 32)] * 3, dtype)
    logger.info("Calling flash_attention_varlen_func: expecting ValueError for causal query > key")
    with pytest.raises(ValueError):
        flash_attention_varlen_func(
            *inputs,
            cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda"),
            cu_seqlens_k=torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda"),
            max_seqlen_q=3,
            max_seqlen_k=3,
            causal=True,
        )
