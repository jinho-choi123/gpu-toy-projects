# ruff: noqa: F722, T201
"""Benchmark fixed-length FlashAttention against materialized SDPA MATH on CUDA."""

import argparse
from collections.abc import Callable
from statistics import median

import torch
import torch.nn.functional as F
from jaxtyping import BFloat16, Float16, Float32
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from flash_attention_1_triton import flash_attention_func

type AttentionTensor = (
    Float16[Tensor, "batch length heads dim"]
    | BFloat16[Tensor, "batch length heads dim"]
    | Float32[Tensor, "batch length heads dim"]
)
type Result = AttentionTensor | tuple[AttentionTensor, ...]


def math_attention(q: AttentionTensor, k: AttentionTensor, v: AttentionTensor) -> AttentionTensor:
    """Run SDPA MATH with the package's batch/sequence/head/dimension layout.

    Args:
        q: Queries.
        k: Keys.
        v: Values.

    Returns:
        Attention output in the input layout.
    """
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0
        ).transpose(1, 2)


def make_step(
    forward: Callable[[], AttentionTensor],
    inputs: tuple[AttentionTensor, ...],
    grad: AttentionTensor,
    mode: str,
) -> Callable[[], Result]:
    """Build a timed operation without accumulating leaf gradients.

    Args:
        forward: Attention call using preallocated inputs.
        inputs: Differentiable Q/K/V tensors.
        grad: Fixed upstream gradient.
        mode: Forward, backward, or both.

    Returns:
        Repeatable operation. Backward mode retains one precomputed forward graph.
    """
    if mode == "forward":
        return torch.no_grad()(forward)
    if mode == "backward":
        output = forward()
        return lambda: torch.autograd.grad(output, inputs, grad, retain_graph=True)
    if mode == "both":
        return lambda: torch.autograd.grad(forward(), inputs, grad)
    raise ValueError(f"Unknown mode: {mode}")


def measure(
    setup: Callable[[], Callable[[], Result]],
    cuda_graph: bool,
    warmup: int,
    iterations: int,
) -> float:
    """Measure median CUDA-event latency, excluding warmup and graph capture.

    Args:
        setup: Build the operation on the measurement stream, including any saved forward.
        cuda_graph: Capture once and time graph replays when enabled.
        warmup: Number of untimed warmup calls.
        iterations: Number of timed samples.

    Returns:
        Median milliseconds per operation.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step = setup()
        for _ in range(warmup):
            step()
        stream.synchronize()
        graph = None
        if cuda_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured_output = step()
            # Keep captured outputs alive until all replays finish.
            assert captured_output is not None
            graph.replay()
            stream.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            if graph is None:
                step()
            else:
                graph.replay()
            end.record()
        stream.synchronize()
    return median(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))


def positive_int(value: str) -> int:
    """Parse a positive CLI dimension or repetition count.

    Args:
        value: Command-line text.

    Returns:
        Positive integer.
    """
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> None:
    """Parse one benchmark case, check agreement, and print latencies and speedup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--query-length", type=positive_int, default=1024)
    parser.add_argument(
        "--key-length", type=positive_int, default=None, help="defaults to query length"
    )
    parser.add_argument("--heads", type=positive_int, default=8)
    parser.add_argument("--head-dim", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--mode", choices=["forward", "backward", "both"], default="forward")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--warmup", type=positive_int, default=25)
    parser.add_argument("--iterations", type=positive_int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("this GPU does not support bfloat16")
    key_length = args.key_length or args.query_length
    torch.manual_seed(0)
    shapes = [
        (args.batch_size, length, args.heads, args.head_dim)
        for length in (args.query_length, key_length, key_length)
    ]
    q, k, v = (
        torch.randn(
            shape,
            device="cuda",
            dtype=getattr(torch, args.dtype),
            requires_grad=args.mode != "forward",
        )
        for shape in shapes
    )
    inputs = (q, k, v)
    grad = torch.randn_like(q)
    print(f"GPU: {torch.cuda.get_device_name()} | torch: {torch.__version__}")
    print(
        f"Q={tuple(q.shape)} K/V={tuple(k.shape)} dtype={args.dtype} "
        f"mode={args.mode} cuda_graph={args.cuda_graph}"
    )
    print("Non-causal; default scale; SDPA MATH uses FP32 intermediates.")
    reference_step = make_step(lambda: math_attention(q, k, v), inputs, grad, args.mode)
    triton_available = False
    try:
        candidate_step = make_step(lambda: flash_attention_func(q, k, v), inputs, grad, args.mode)
        actual = candidate_step()
    except NotImplementedError as error:
        print(f"Triton: unavailable ({error}); speedup unavailable")
    else:
        atol, rtol = (1e-3, 1e-2) if args.dtype == "float16" else (1e-2, 5e-2)
        torch.testing.assert_close(actual, reference_step(), atol=atol, rtol=rtol)
        # Check forward output as well when the selected step only returns gradients.
        if args.mode != "forward":
            with torch.no_grad():
                torch.testing.assert_close(
                    flash_attention_func(q, k, v), math_attention(q, k, v), atol=atol, rtol=rtol
                )
        del actual
        triton_available = True
        print("Correctness: passed")
    del reference_step
    if triton_available:
        del candidate_step
    math_ms = measure(
        lambda: make_step(lambda: math_attention(q, k, v), inputs, grad, args.mode),
        args.cuda_graph,
        args.warmup,
        args.iterations,
    )
    print(f"SDPA MATH: {math_ms:.6f} ms (median, {args.iterations} samples)")
    if triton_available:
        triton_ms = measure(
            lambda: make_step(lambda: flash_attention_func(q, k, v), inputs, grad, args.mode),
            args.cuda_graph,
            args.warmup,
            args.iterations,
        )
        print(f"Triton:    {triton_ms:.6f} ms")
        print(f"Speedup (MATH / Triton): {math_ms / triton_ms:.3f}x")


if __name__ == "__main__":
    main()
