"""Shared GDN input contract and forward call."""

from dataclasses import dataclass

import torch


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


def run_forward(gdn_input: GdnInput) -> tuple[torch.Tensor, torch.Tensor]:
    """Run GDN forward."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

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
