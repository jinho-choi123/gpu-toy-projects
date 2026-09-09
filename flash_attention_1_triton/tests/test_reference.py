"""Hand-computed checks for the test oracle, runnable without CUDA."""

import math

import pytest
import torch
from reference import attention, attention_varlen


@pytest.mark.parametrize(
    "causal,values,v_gradient",
    [(False, [4.0, 4.0], [2 / 3, 2 / 3, 2 / 3]), (True, [2.0, 4.0], [5 / 6, 5 / 6, 1 / 3])],
)
def test_reference_mask_and_gradients(
    causal: bool, values: list[float], v_gradient: list[float]
) -> None:
    """Check hand-computed bottom-right outputs and gradients for uniform scores.

    Args:
        causal (bool): Enable the bottom-right mask in the FP32 reference.
        values (list[float]): Expected scalar output for each of the two query positions.
        v_gradient (list[float]): Expected scalar gradient for each of the three value positions.

    Returns:
        None: Complete if the reference output and Q/K/V gradients match the hand calculation.
    """
    q = torch.zeros(1, 2, 1, 1, requires_grad=True)
    k = torch.zeros(1, 3, 1, 1, requires_grad=True)
    v = torch.tensor([1.0, 3.0, 8.0]).reshape(1, 3, 1, 1).requires_grad_()
    output = attention(q, k, v, causal=causal)
    torch.testing.assert_close(output.flatten(), torch.tensor(values))
    dq, dk, dv = torch.autograd.grad(output.sum(), (q, k, v))
    torch.testing.assert_close(dq, torch.zeros_like(q))
    torch.testing.assert_close(dk, torch.zeros_like(k))
    torch.testing.assert_close(dv.flatten(), torch.tensor(v_gradient))


@pytest.mark.parametrize("scale", [None, 1.0])
def test_reference_scale_and_nonzero_gradients(scale: float | None) -> None:
    """Check a closed-form sigmoid output and nonzero Q/K gradients.

    Args:
        scale (float | None): Use an explicit scale of 1.0 or the default head_dim**-0.5.

    Returns:
        None: Complete if the FP32 reference output and gradients match the sigmoid formulas.
    """
    q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]], requires_grad=True)
    k = torch.tensor([[[[0.0] * 4], [[2.0, 0.0, 0.0, 0.0]]]], requires_grad=True)
    v = torch.tensor([[[[0.0] * 4], [[1.0, 0.0, 0.0, 0.0]]]], requires_grad=True)
    output = attention(q, k, v, softmax_scale=scale)
    effective_scale = 0.5 if scale is None else scale
    probability = 1 / (1 + math.exp(-2 * effective_scale))
    slope = effective_scale * probability * (1 - probability)
    torch.testing.assert_close(output.flatten(), torch.tensor([probability, 0.0, 0.0, 0.0]))
    dq, dk, dv = torch.autograd.grad(output.sum(), (q, k, v))
    torch.testing.assert_close(dq.flatten(), torch.tensor([2 * slope, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(
        dk.flatten(), torch.tensor([-slope, 0.0, 0.0, 0.0, slope, 0.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        dv.flatten(), torch.tensor([1 - probability] * 4 + [probability] * 4)
    )


@pytest.mark.parametrize(
    "causal,values,v_gradient",
    [
        (False, [4.0, 4.0, 15.0], [2 / 3, 2 / 3, 2 / 3, 0.5, 0.5]),
        (True, [2.0, 4.0, 15.0], [5 / 6, 5 / 6, 1 / 3, 0.5, 0.5]),
    ],
)
def test_varlen_reference_packing_and_gradients(
    causal: bool, values: list[float], v_gradient: list[float]
) -> None:
    """Check packed sequence isolation, output order, and the gradient graph.

    Args:
        causal (bool): Enable an independent bottom-right mask for each sequence.
        values (list[float]): Hand-computed outputs for the three packed queries.
        v_gradient (list[float]): Hand-computed gradients for the five packed values.

    Returns:
        None: Complete if packed outputs and Q/K/V gradients match the hand calculation.
    """
    q = torch.zeros(3, 1, 1, requires_grad=True)
    k = torch.zeros(5, 1, 1, requires_grad=True)
    v = torch.tensor([1.0, 3.0, 8.0, 10.0, 20.0]).reshape(5, 1, 1).requires_grad_()
    output = attention_varlen(q, k, v, q_lengths=[2, 1], k_lengths=[3, 2], causal=causal)
    assert output.shape == q.shape
    assert output.is_contiguous()
    torch.testing.assert_close(output.flatten(), torch.tensor(values))
    dq, dk, dv = torch.autograd.grad(output.sum(), (q, k, v))
    torch.testing.assert_close(dq, torch.zeros_like(q))
    torch.testing.assert_close(dk, torch.zeros_like(k))
    torch.testing.assert_close(dv.flatten(), torch.tensor(v_gradient))
