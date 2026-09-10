"""Fixed-length API output, gradient, layout, and causal contract tests."""

import pytest
import torch
from loguru import logger
from reference import assert_output, assert_output_and_gradients, attention, make_inputs

from flash_attention_1_triton import flash_attention_func

pytestmark = pytest.mark.usefixtures("cuda")


attention_cases = pytest.mark.parametrize(
    "batch,heads,q_len,k_len,causal,scale,strided",
    [
        pytest.param(1, 1, 1, 1, False, None, False, id="single-token"),
        pytest.param(2, 3, 63, 63, False, None, False, id="length-63"),
        pytest.param(2, 2, 64, 64, True, None, False, id="causal-length-64"),
        pytest.param(2, 3, 65, 65, True, 0.3, True, id="causal-strided-length-65"),
        pytest.param(2, 2, 127, 129, False, 0.2, False, id="cross-attention"),
        pytest.param(2, 3, 128, 257, True, None, False, id="causal-bottom-right"),
        pytest.param(1, 2, 1, 65, True, 0.3, False, id="causal-single-query"),
        pytest.param(2, 2, 129, 65, False, None, True, id="longer-query-strided"),
    ],
)


@attention_cases
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_forward(
    dtype: torch.dtype,
    head_dim: int,
    batch: int,
    heads: int,
    q_len: int,
    k_len: int,
    causal: bool,
    scale: float | None,
    strided: bool,
) -> None:
    """Match forward output against PyTorch Flash Attention without autograd.

    Args:
        dtype (torch.dtype): Input dtype, torch.float16 or torch.bfloat16.
        head_dim (int): Per-head dimension: 32, 64, or 128.
        batch (int): Number of independent sequences.
        heads (int): Shared Q/K/V head count.
        q_len (int): Query sequence length.
        k_len (int): Key and value sequence length.
        causal (bool): Enable the bottom-right causal mask.
        scale (float | None): Custom positive logit scale or the API default.
        strided (bool): Test inputs with gaps between rows.

    Returns:
        None: Complete if the output matches the Flash Attention reference.
    """
    inputs = make_inputs(
        [(batch, length, heads, head_dim) for length in (q_len, k_len, k_len)],
        dtype,
        strided=strided,
    )
    with torch.no_grad():
        expected = attention(*inputs, causal=causal, softmax_scale=scale)
        output = flash_attention_func(*inputs, causal=causal, softmax_scale=scale)
    assert not output.requires_grad
    assert_output(output, expected, inputs[0])


@attention_cases
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_output_and_gradients(
    dtype: torch.dtype,
    head_dim: int,
    batch: int,
    heads: int,
    q_len: int,
    k_len: int,
    causal: bool,
    scale: float | None,
    strided: bool,
) -> None:
    """Match PyTorch Flash Attention including scale, causal alignment, and strided inputs.

    Args:
        dtype (torch.dtype): Input dtype, torch.float16 or torch.bfloat16.
        head_dim (int): Per-head dimension: 32, 64, or 128.
        batch (int): Number of independent sequences.
        heads (int): Shared Q/K/V head count.
        q_len (int): Query sequence length.
        k_len (int): Key and value sequence length.
        causal (bool): Enable the bottom-right causal mask.
        scale (float | None): Custom positive logit scale or the API default.
        strided (bool): Test inputs with gaps between rows.

    Returns:
        None: Complete if the output and Q/K/V gradients match the Flash Attention reference.
    """
    inputs = make_inputs(
        [(batch, length, heads, head_dim) for length in (q_len, k_len, k_len)],
        dtype,
        strided=strided,
    )
    logger.info("Preparing independent Flash Attention reference inputs")
    reference_inputs = tuple(x.detach().requires_grad_() for x in inputs)
    expected = attention(*reference_inputs, causal=causal, softmax_scale=scale)
    logger.info("Calling flash_attention_func")
    output = flash_attention_func(*inputs, causal=causal, softmax_scale=scale)
    assert_output_and_gradients(output, expected, inputs, reference_inputs)


def test_causal_rejects_longer_query(dtype: torch.dtype) -> None:
    """Reject a causal query longer than the keys with the documented error.

    Args:
        dtype (torch.dtype): Input dtype, torch.float16 or torch.bfloat16.

    Returns:
        None: Complete only if the API raises ValueError for query length 3 and key length 2.
    """
    inputs = make_inputs([(1, length, 1, 32) for length in (3, 2, 2)], dtype)
    logger.info("Calling flash_attention_func: expecting ValueError for causal query > key")
    with pytest.raises(ValueError):
        flash_attention_func(*inputs, causal=True)
