"""Check benchmark operations against explicit materialized attention."""

import pytest
import torch

from flash_attention_1_triton.benchmark import make_step, math_attention, measure


@pytest.mark.parametrize("mode", ["forward", "backward", "both"])
def test_benchmark_step(mode: str) -> None:
    """Check outputs or gradients for each timed operation.

    Args:
        mode: Operation to check.
    """
    torch.manual_seed(0)
    q = torch.randn(2, 3, 2, 32, requires_grad=mode != "forward")
    k = torch.randn(2, 5, 2, 32, requires_grad=mode != "forward")
    v = torch.randn_like(k, requires_grad=mode != "forward")
    grad = torch.randn_like(q)
    scores = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1) / 32**0.5
    expected = (scores.softmax(-1) @ v.transpose(1, 2)).transpose(1, 2)
    step = make_step(lambda: math_attention(q, k, v), (q, k, v), grad, mode)
    if mode != "forward":
        expected = torch.autograd.grad(expected, (q, k, v), grad)
    for _ in range(2):
        actual = step()
        torch.testing.assert_close(actual, expected)
        assert all(t.grad is None for t in (q, k, v))
    if mode == "forward":
        assert isinstance(actual, torch.Tensor)
        assert not actual.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["forward", "backward", "both"])
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_math_timing(mode: str, cuda_graph: bool) -> None:
    """Exercise real CUDA timing and graph capture for every operation.

    Args:
        mode: Operation to time.
        cuda_graph: Whether to capture and replay the operation.
    """
    q, k, v = (
        torch.randn(
            1, 16, 2, 32, device="cuda", dtype=torch.float16, requires_grad=mode != "forward"
        )
        for _ in range(3)
    )
    grad = torch.ones_like(q)
    assert (
        measure(
            lambda: make_step(lambda: math_attention(q, k, v), (q, k, v), grad, mode),
            cuda_graph,
            warmup=2,
            iterations=3,
        )
        > 0
    )
