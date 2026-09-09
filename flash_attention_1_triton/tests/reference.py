"""Small, explicit FP32 attention oracle and shared numerical assertions."""

import torch
from jaxtyping import BFloat16, Float16, Float32
from loguru import logger
from torch import Tensor

type LowPrecision = (
    Float16[Tensor, "*batch length heads head_dim"]
    | BFloat16[Tensor, "*batch length heads head_dim"]
)
type ReferenceTensor = Float32[Tensor, "*batch length heads head_dim"]


def attention(
    q: Float32[Tensor, "batch query_length heads head_dim"],
    k: Float32[Tensor, "batch key_length heads head_dim"],
    v: Float32[Tensor, "batch key_length heads head_dim"],
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> Float32[Tensor, "batch query_length heads head_dim"]:
    """Compute FP32 attention in [batch, sequence, heads, dimension] layout.

    Args:
        q (Tensor, float32): Queries shaped [batch, query_length, heads, head_dim].
        k (Tensor, float32): Keys shaped [batch, key_length, heads, head_dim].
        v (Tensor, float32): Values with the same shape and device as k.
        causal (bool): Apply a bottom-right mask; query_length must not exceed key_length.
        softmax_scale (float | None): Positive finite logit scale; None uses head_dim**-0.5.

    Returns:
        Tensor (float32): Contiguous output shaped like q on the same device, with an
            autograd graph for first-order Q/K/V gradients.
    """
    logger.info(
        "Reference: QK^T -> mask -> softmax -> V | q_shape={} k_shape={} causal={} scale={}",
        tuple(q.shape),
        tuple(k.shape),
        causal,
        softmax_scale,
    )
    q, k, v = (x.float().transpose(1, 2) for x in (q, k, v))
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        query_length, key_length = scores.shape[-2:]
        rows = torch.arange(query_length, device=q.device)[:, None]
        columns = torch.arange(key_length, device=q.device)[None, :]
        scores = scores.masked_fill(columns > rows + key_length - query_length, -torch.inf)
    return (scores.softmax(dim=-1) @ v).transpose(1, 2).contiguous()


def attention_varlen(
    q: Float32[Tensor, "total_queries heads head_dim"],
    k: Float32[Tensor, "total_keys heads head_dim"],
    v: Float32[Tensor, "total_keys heads head_dim"],
    *,
    q_lengths: list[int],
    k_lengths: list[int],
    causal: bool = False,
    softmax_scale: float | None = None,
) -> Float32[Tensor, "total_queries heads head_dim"]:
    """Compute packed FP32 attention independently for each sequence.

    Args:
        q (Tensor): Packed queries shaped [total_queries, heads, head_dim].
        k (Tensor): Packed keys shaped [total_keys, heads, head_dim].
        v (Tensor): Packed values with the same shape and device as k.
        q_lengths (list[int]): Positive query lengths in packing order, summing to total_queries.
        k_lengths (list[int]): Corresponding positive key lengths, summing to total_keys.
        causal (bool): Apply bottom-right masks; each query length must not exceed its key length.
        softmax_scale (float | None): Positive finite logit scale; None uses head_dim**-0.5.

    Returns:
        Tensor: Contiguous FP32 output shaped like q, in packing order, with an autograd
            graph connecting every sequence's output to its Q/K/V inputs.
    """
    logger.info("Splitting packed reference: q_lengths={} k_lengths={}", q_lengths, k_lengths)
    sequences = [
        tensor.split(lengths)
        for tensor, lengths in zip((q, k, v), (q_lengths, k_lengths, k_lengths), strict=True)
    ]
    outputs = [
        attention(
            *(x.unsqueeze(0) for x in sequence), causal=causal, softmax_scale=softmax_scale
        ).squeeze(0)
        for sequence in zip(*sequences, strict=True)
    ]
    logger.info("Concatenating {} reference sequence outputs", len(outputs))
    return torch.cat(outputs)


def make_inputs(
    shapes: list[tuple[int, ...]], dtype: torch.dtype, *, strided: bool = False
) -> tuple[LowPrecision, ...]:
    """Create reproducible CUDA leaves, optionally with gaps between rows.

    Args:
        shapes (list[tuple[int, ...]]): Q/K/V shapes in order: [batch, length, heads,
            head_dim] for fixed-length inputs or [total_length, heads, head_dim] for packed inputs.
        dtype (torch.dtype): Input dtype, either torch.float16 or torch.bfloat16.
        strided (bool): Leave gaps between rows while keeping the final dimension contiguous.

    Returns:
        tuple[Tensor, ...] (float16 | bfloat16): CUDA Q/K/V leaves in shapes order with
            requires_grad=True and the requested dtype. Seed 0 makes generation reproducible.
    """
    logger.info("Creating Q/K/V: shapes={} dtype={} strided={}", shapes, dtype, strided)
    generator = torch.Generator(device="cuda").manual_seed(0)
    inputs = []
    for shape in shapes:
        storage_shape = (*shape[:-1], shape[-1] * 2) if strided else shape
        tensor = torch.randn(storage_shape, dtype=dtype, device="cuda", generator=generator)
        if strided:
            tensor = tensor[..., : shape[-1]]
        inputs.append(tensor.requires_grad_())
    return tuple(inputs)


def assert_output_and_gradients(
    output: LowPrecision,
    expected: ReferenceTensor,
    inputs: tuple[LowPrecision, ...],
    reference_inputs: tuple[ReferenceTensor, ...],
) -> None:
    """Check the output contract and all three vector-Jacobian products.

    Args:
        output (Tensor, float16 | bfloat16): API output shaped like the input q.
        expected (Tensor, float32): Independent reference output with the same shape and device.
        inputs (tuple[Tensor, ...], float16 | bfloat16): Original Q/K/V leaves in fixed-length
            [batch, length, heads, head_dim] or packed [total_length, heads, head_dim] layout.
        reference_inputs (tuple[Tensor, ...], float32): Independent Q/K/V leaves matching
            inputs in shape and device and connected to expected's autograd graph.

    Returns:
        None: Log numerical errors and return only if output and gradient checks pass.

    Raises:
        AssertionError: If the output contract or a numerical comparison fails.
    """
    logger.info("Checking output shape, dtype, device, and contiguity")
    q = inputs[0]
    assert output.shape == q.shape
    assert output.dtype == q.dtype
    assert output.device == q.device
    assert output.is_contiguous()
    atol, rtol = (1e-3, 1e-2) if q.dtype == torch.float16 else (1e-2, 5e-2)
    logger.info(
        "Output: shape={} dtype={} device={} max_abs_error={:.6g} atol={} rtol={}",
        tuple(output.shape),
        output.dtype,
        output.device,
        (output.detach().float() - expected.detach()).abs().max().item(),
        atol,
        rtol,
    )
    torch.testing.assert_close(output.float(), expected, atol=atol, rtol=rtol)

    logger.info("Output comparison passed; creating upstream gradient")
    generator = torch.Generator(device=q.device).manual_seed(1)
    upstream = torch.randn(output.shape, dtype=q.dtype, device=q.device, generator=generator)
    logger.info("Computing API Q/K/V gradients")
    actual_grads = torch.autograd.grad(output, inputs, upstream)
    logger.info("Computing FP32 reference Q/K/V gradients")
    expected_grads = torch.autograd.grad(expected, reference_inputs, upstream.float())
    for name, actual, reference in zip(("q", "k", "v"), actual_grads, expected_grads, strict=True):
        logger.info(
            "d{}: max_abs_error={:.6g} atol={} rtol={}",
            name,
            (actual.detach().float() - reference.detach()).abs().max().item(),
            atol,
            rtol,
        )
        torch.testing.assert_close(
            actual.float(),
            reference,
            atol=atol,
            rtol=rtol,
            msg=lambda msg, name=name: f"d{name}: {msg}",
        )
    logger.info("Q/K/V gradient comparisons passed")
