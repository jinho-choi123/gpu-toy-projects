"""Shared GDN forward profiling runner."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn.functional as F
from loguru import logger

HEAD_DIM = 128
DTYPE = torch.bfloat16
EAGER_WARMUP = 10
GRAPH_WARMUP = 10


class ProfileBackend(StrEnum):
    """FLA backend selected by a profiling entry point."""

    FLASH_QLA = "flash_qla"
    FLA_TRITON = "fla_triton"


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    """Runtime-configurable dimensions and execution controls."""

    batch_size: int
    seq_len: int
    num_heads: int
    seed: int
    iterations: int


@dataclass(frozen=True, slots=True)
class GdnInput:
    """Inputs reused by every warmup and measured forward."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    initial_state: torch.Tensor
    scale: float


GdnForward = Callable[
    [GdnInput],
    tuple[torch.Tensor, torch.Tensor],
]


def _parse_args(argv: Sequence[str] | None) -> ProfileConfig:
    def _positive_int(raw: str) -> int:
        value = int(raw)
        if value <= 0:
            raise argparse.ArgumentTypeError("value must be positive")
        return value

    parser = argparse.ArgumentParser(description="Profile a GDN forward implementation.")
    _ = parser.add_argument("--batch-size", type=_positive_int, default=1)
    _ = parser.add_argument("--seq-len", type=_positive_int, default=16_384)
    _ = parser.add_argument("--num-heads", type=_positive_int, default=16)
    _ = parser.add_argument("--seed", type=int, default=42)
    _ = parser.add_argument("--iterations", type=_positive_int, default=1)
    args = parser.parse_args(argv)

    return ProfileConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        seed=args.seed,
        iterations=args.iterations,
    )


def main(backend: ProfileBackend, argv: Sequence[str] | None = None) -> int:
    """Run the selected GDN forward workload."""
    # parse the configuration from the command line
    config = _parse_args(argv)

    # VERIFICATION

    # set the FLA_FLASH_QLA env var
    os.environ["FLA_FLASH_QLA"] = "1" if backend == ProfileBackend.FLASH_QLA else "0"

    if not torch.cuda.is_available():
        err_msg = "CUDA is required for GDN profiling."
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        detected_capability = f"SM{capability[0]}{capability[1]}"
        err_msg = " ".join(
            [
                "GDN profiling supports SM90 only;",
                f"found {detected_capability} on {device}.",
            ]
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    torch.cuda.set_device(device=device)

    profiling_info: dict[str, object] = {
        "backend": backend.value,
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "num_heads": config.num_heads,
        "head_dim": HEAD_DIM,
        "value_head_dim": HEAD_DIM,
        "dtype": str(DTYPE).removeprefix("torch."),
        "device": device,
        "seed": config.seed,
        "eager_warmup": EAGER_WARMUP,
        "graph_warmup": GRAPH_WARMUP,
        "iterations": config.iterations,
    }

    logger.info(" ".join(f"{key}={value}" for key, value in profiling_info.items()))

    # INPUT GENERATION
    torch.manual_seed(config.seed)

    input_shape = (
        config.batch_size,
        config.seq_len,
        config.num_heads,
        HEAD_DIM,
    )
    gate_shape = (
        config.batch_size,
        config.seq_len,
        config.num_heads,
    )

    q = torch.randn(*input_shape, device=device, dtype=DTYPE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    g = F.logsigmoid(torch.randn(*gate_shape, device=device, dtype=torch.float32))
    beta = torch.randn(
        *gate_shape,
        device=device,
        dtype=torch.float32,
    ).sigmoid()
    initial_state = torch.randn(
        config.batch_size,
        config.num_heads,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )

    gdn_input = GdnInput(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        scale=HEAD_DIM**-0.5,
    )

    # RUN FORWARD

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    def run_forward(gdn_input: GdnInput) -> tuple[torch.Tensor, torch.Tensor]:
        output, final_state = chunk_gated_delta_rule(  # pyright: ignore[reportCallIssue]
            q=gdn_input.q,  # pyright: ignore[reportCallIssue]
            k=gdn_input.k,  # pyright: ignore[reportCallIssue]
            v=gdn_input.v,  # pyright: ignore[reportCallIssue]
            g=gdn_input.g,  # pyright: ignore[reportCallIssue]
            beta=gdn_input.beta,  # pyright: ignore[reportCallIssue]
            scale=gdn_input.scale,  # pyright: ignore[reportCallIssue]
            initial_state=gdn_input.initial_state,  # pyright: ignore[reportCallIssue]
            output_final_state=True,  # pyright: ignore[reportCallIssue]
            use_qk_l2norm_in_kernel=True,  # pyright: ignore[reportCallIssue]
        )
        if final_state is None:
            raise RuntimeError("GDN forward did not return the requested final state.")
        return output, final_state

    forward: GdnForward = run_forward

    with torch.inference_mode():
        current_stream = torch.cuda.current_stream(device)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(current_stream)

        with torch.cuda.stream(warmup_stream):
            for _ in range(EAGER_WARMUP):
                forward(gdn_input)

        current_stream.wait_stream(warmup_stream)

        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            _static_output, _static_final_state = forward(gdn_input)

        for _ in range(GRAPH_WARMUP):
            cuda_graph.replay()

        torch.cuda.synchronize(device)

        torch.cuda.profiler.start()
        try:
            with torch.cuda.nvtx.range("gdn_forward"):
                for iteration in range(config.iterations):
                    with torch.cuda.nvtx.range(f"iteration_{iteration}"):
                        cuda_graph.replay()

            torch.cuda.synchronize(device)
        finally:
            torch.cuda.profiler.stop()

    return 0
