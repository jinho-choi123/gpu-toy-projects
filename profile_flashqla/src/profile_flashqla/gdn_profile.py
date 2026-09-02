"""Shared GDN forward profiling runner."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

from .utils import GdnInput, run_forward

HEAD_DIM = 128
DTYPE = torch.bfloat16
EAGER_WARMUP = 10
GRAPH_WARMUP = 10
MEMORY_HISTORY_MAX_ENTRIES = 100_000
BYTES_PER_MIB = 1024**2
PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"


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


def _to_mib(num_bytes: int) -> float:
    """Convert bytes to MiB."""
    return num_bytes / BYTES_PER_MIB


def _profile_eager_forward_memory(
    backend: ProfileBackend,
    config: ProfileConfig,
    device: torch.device,
    gdn_input: GdnInput,
) -> None:
    """Capture memory history for one warmed-up eager GDN forward."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = PROFILES_DIR / (
        f"{backend.value}_b{config.batch_size}_t{config.seq_len}"
        f"_h{config.num_heads}_memory_snapshot.pickle"
    )

    # Complete all warmup work and remove unused allocator cache so that the
    # baseline consists of live inputs and persistent runtime allocations.
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)

    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="all",
        max_entries=MEMORY_HISTORY_MAX_ENTRIES,
        device=device,
        clear_history=True,
    )

    try:
        memory_output, memory_final_state = run_forward(gdn_input)
        torch.cuda.synchronize(device)

        # Read these values while both returned tensors are still alive.
        allocated_after = torch.cuda.memory_allocated(device)
        reserved_after = torch.cuda.memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

        returned_tensor_bytes = (
            memory_output.numel() * memory_output.element_size()
            + memory_final_state.numel() * memory_final_state.element_size()
        )

        # Include the output deallocation events in the memory timeline.
        del memory_output, memory_final_state
        torch.cuda.synchronize(device)

        torch.cuda.memory._dump_snapshot(str(snapshot_path))
    finally:
        # This must finish before torch.cuda.profiler.start().
        torch.cuda.memory._record_memory_history(enabled=None, device=device)

    logger.info(
        " ".join(
            [
                f"memory_backend={backend.value}",
                f"memory_baseline_allocated_mib={_to_mib(allocated_before):.2f}",
                f"memory_after_forward_allocated_mib={_to_mib(allocated_after):.2f}",
                f"memory_peak_allocated_mib={_to_mib(peak_allocated):.2f}",
                (f"memory_forward_peak_delta_mib={_to_mib(peak_allocated - allocated_before):.2f}"),
                (
                    "memory_forward_retained_delta_mib="
                    f"{_to_mib(allocated_after - allocated_before):.2f}"
                ),
                (f"memory_temporary_over_end_mib={_to_mib(peak_allocated - allocated_after):.2f}"),
                f"memory_returned_tensors_mib={_to_mib(returned_tensor_bytes):.2f}",
                f"memory_baseline_reserved_mib={_to_mib(reserved_before):.2f}",
                f"memory_after_forward_reserved_mib={_to_mib(reserved_after):.2f}",
                f"memory_peak_reserved_mib={_to_mib(peak_reserved):.2f}",
                (f"memory_peak_reserved_delta_mib={_to_mib(peak_reserved - reserved_before):.2f}"),
                f"memory_snapshot={snapshot_path}",
            ]
        )
    )

    # Do not let the eager memory-profiling run affect graph-private pool setup.
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


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
    with torch.inference_mode():
        current_stream = torch.cuda.current_stream(device)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(current_stream)

        with torch.cuda.stream(warmup_stream):
            for _ in range(EAGER_WARMUP):
                run_forward(gdn_input)

        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize(device)
        # Measure one warmed-up eager forward. Memory recording is stopped
        # completely before CUDA Graph capture and Nsight profiling.
        _profile_eager_forward_memory(
            backend=backend,
            config=config,
            device=device,
            gdn_input=gdn_input,
        )

        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            _static_output, _static_final_state = run_forward(gdn_input)

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
