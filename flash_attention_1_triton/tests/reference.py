"""PyTorch Flash Attention adapters and shared numerical assertions."""

import torch
from jaxtyping import BFloat16, Float16, Int32
from loguru import logger
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from torch.nn.attention.varlen import varlen_attn
from torch.nn.functional import scaled_dot_product_attention

type LowPrecision = (
    Float16[Tensor, "*batch length heads head_dim"]
    | BFloat16[Tensor, "*batch length heads head_dim"]
)


def attention(
    q: LowPrecision,
    k: LowPrecision,
    v: LowPrecision,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> LowPrecision:
    """Call PyTorch Flash SDPA with the fixed-length API's layout and options.

    Args:
        q (Tensor): Queries shaped [batch, query_length, heads, head_dim].
        k (Tensor): Keys shaped [batch, key_length, heads, head_dim].
        v (Tensor): Values with the same shape as k.
        causal (bool): Apply a bottom-right-aligned causal mask.
        softmax_scale (float | None): Logit scale; None uses head_dim**-0.5.

    Returns:
        Tensor: Contiguous output shaped like q, preserving dtype and autograd.
    """
    bias = causal_lower_right(q.shape[1], k.shape[1]) if causal else None
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = scaled_dot_product_attention(
            *(x.transpose(1, 2) for x in (q, k, v)),
            attn_mask=bias,
            scale=softmax_scale,
        )
    return output.transpose(1, 2).contiguous()


def attention_varlen(
    q: LowPrecision,
    k: LowPrecision,
    v: LowPrecision,
    *,
    cu_seqlens_q: Int32[Tensor, " batch_plus_one"],
    cu_seqlens_k: Int32[Tensor, " batch_plus_one"],
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> LowPrecision:
    """Call PyTorch Flash varlen attention with the packed API's options.

    Args:
        q (Tensor): Packed queries shaped [total_queries, heads, head_dim].
        k (Tensor): Packed keys shaped [total_keys, heads, head_dim].
        v (Tensor): Values with the same shape as k.
        cu_seqlens_q (Tensor): CUDA int32 cumulative query lengths.
        cu_seqlens_k (Tensor): CUDA int32 cumulative key lengths.
        max_seqlen_q (int): Maximum query sequence length.
        max_seqlen_k (int): Maximum key sequence length.
        causal (bool): Apply a bottom-right-aligned causal mask.
        softmax_scale (float | None): Logit scale; None uses head_dim**-0.5.

    Returns:
        Tensor: Contiguous output shaped like q, preserving dtype and autograd.
    """
    output = varlen_attn(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scale=softmax_scale,
        window_size=(-1, 0) if causal else (-1, -1),
    )
    assert isinstance(output, Tensor)
    return output.contiguous()


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


def assert_output(output: LowPrecision, expected: LowPrecision, q: LowPrecision) -> None:
    """Check the output contract and numerical accuracy.

    Args:
        output (Tensor, float16 | bfloat16): API output to check.
        expected (Tensor, float16 | bfloat16): Independent reference output.
        q (Tensor, float16 | bfloat16): Query defining the output shape, dtype, and device.

    Returns:
        None: Complete if the output contract and numerical comparison pass.

    Raises:
        AssertionError: If the output contract or numerical comparison fails.
    """
    logger.info("Checking output shape, dtype, device, and contiguity")
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
    torch.testing.assert_close(output.float(), expected.float(), atol=atol, rtol=rtol)


def assert_output_and_gradients(
    output: LowPrecision,
    expected: LowPrecision,
    inputs: tuple[LowPrecision, ...],
    reference_inputs: tuple[LowPrecision, ...],
) -> None:
    """Check the output contract and all three vector-Jacobian products.

    Args:
        output (Tensor, float16 | bfloat16): API output shaped like the input q.
        expected (Tensor, float16 | bfloat16): Independent Flash Attention output
            with the same shape and device.
        inputs (tuple[Tensor, ...], float16 | bfloat16): Original Q/K/V leaves in fixed-length
            [batch, length, heads, head_dim] or packed [total_length, heads, head_dim] layout.
        reference_inputs (tuple[Tensor, ...], float16 | bfloat16): Independent Q/K/V leaves matching
            inputs in shape and device and connected to expected's autograd graph.

    Returns:
        None: Log numerical errors and return only if output and gradient checks pass.

    Raises:
        AssertionError: If the output contract or a numerical comparison fails.
    """
    q = inputs[0]
    assert_output(output, expected, q)
    atol, rtol = (1e-3, 1e-2) if q.dtype == torch.float16 else (1e-2, 5e-2)

    logger.info("Output comparison passed; creating upstream gradient")
    generator = torch.Generator(device=q.device).manual_seed(1)
    upstream = torch.randn(output.shape, dtype=q.dtype, device=q.device, generator=generator)
    logger.info("Computing API Q/K/V gradients")
    actual_grads = torch.autograd.grad(output, inputs, upstream)
    logger.info("Computing Flash Attention reference Q/K/V gradients")
    expected_grads = torch.autograd.grad(expected, reference_inputs, upstream)
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
            reference.float(),
            atol=atol,
            rtol=rtol,
            msg=lambda msg, name=name: f"d{name}: {msg}",
        )
    logger.info("Q/K/V gradient comparisons passed")
