"""Validate the forward launcher before the attention kernel is implemented."""

import pytest
import torch

from flash_attention_1_triton import flash_attention_func


def test_rejects_cpu() -> None:
    """Reject CPU tensors before attempting a GPU launch.

    Returns:
        None: Complete if CPU inputs raise ValueError.
    """
    q = torch.empty((1, 2, 1, 32), dtype=torch.float16)
    with pytest.raises(ValueError, match="CUDA"):
        flash_attention_func(q, q, q)


@pytest.mark.usefixtures("cuda")
@pytest.mark.parametrize(
    "case",
    ["dtype", "device", "head_dim", "stride", "empty", "scale_zero", "scale_nan"],
)
def test_rejects_invalid_inputs(case: str) -> None:
    """Reject unsupported inputs before reaching the unfinished kernel.

    Args:
        case (str): Invalid input property to exercise.

    Returns:
        None: Complete if validation raises ValueError.
    """
    q = torch.empty((1, 2, 1, 32), dtype=torch.float16, device="cuda")
    k = v = q
    scale = None
    if case == "dtype":
        k = q.to(torch.bfloat16)
    elif case == "device":
        k = q.cpu()
    elif case == "head_dim":
        q = k = v = q[..., :16]
    elif case == "stride":
        q = k = v = torch.empty((1, 2, 1, 64), dtype=q.dtype, device=q.device)[..., ::2]
    elif case == "empty":
        q = q[:, :0]
    elif case == "scale_zero":
        scale = 0.0
    elif case == "scale_nan":
        scale = float("nan")
    with pytest.raises(ValueError):
        flash_attention_func(q, k, v, softmax_scale=scale)


@pytest.mark.parametrize("input_index", [0, 1, 2])
@pytest.mark.usefixtures("cuda")
def test_rejects_backward_but_allows_no_grad(input_index: int) -> None:
    """Distinguish unsupported training from the unfinished forward kernel.

    Args:
        input_index (int): Q, K, or V input that requires gradients.

    Returns:
        None: Complete if training is rejected and no_grad reaches the kernel guard.
    """
    inputs = [torch.empty((1, 2, 1, 32), dtype=torch.float16, device="cuda") for _ in range(3)]
    inputs[input_index].requires_grad_()
    with torch.enable_grad(), pytest.raises(NotImplementedError, match="backward"):
        flash_attention_func(*inputs)
    with torch.no_grad(), pytest.raises(NotImplementedError, match="kernel"):
        flash_attention_func(*inputs)


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_supported_inputs_reach_kernel_guard(
    dtype: torch.dtype, head_dim: int, causal: bool
) -> None:
    """Accept strided cross-attention inputs without returning unwritten output.

    Args:
        dtype (torch.dtype): Supported input dtype.
        head_dim (int): Supported per-head width.
        causal (bool): Whether to enable bottom-right causal masking.

    Returns:
        None: Complete if valid inputs reach the explicit kernel guard.
    """
    q = torch.empty((2, 6, 3, head_dim), dtype=dtype, device="cuda")[:, ::2]
    k = torch.empty((2, 10, 3, head_dim), dtype=dtype, device="cuda")[:, ::2]
    with pytest.raises(NotImplementedError, match="kernel"):
        flash_attention_func(q, k, k, causal=causal, softmax_scale=0.3)
