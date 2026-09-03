"""Verify Flash QLA integration through the FLA GDN API."""

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule.backends.flash_qla import FlashQLABackend
from fla.utils import assert_close

from profile_flashqla.utils import GdnInput, run_forward

_FLASH_QLA_ERROR_RATIO = 0.008


# Helper function
def record_calls[**P, R](
    func: Callable[P, R],
    calls: list[bool],
) -> Callable[P, R]:
    """Return a type-preserving wrapper that records each invocation.

    The wrapper appends ``True`` to ``calls`` before delegating to ``func``.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        calls.append(True)
        return func(*args, **kwargs)

    return wrapper


def make_gdn_input(*, seq_len: int, num_heads: int) -> GdnInput:
    """Create deterministic GDN inputs."""
    assert torch.cuda.is_available(), "CUDA is required for this test"
    capability = torch.cuda.get_device_capability(0)
    assert capability == (9, 0), (
        "The FlashQLA profiling setup requires an SM90 GPU "
        f"(H100/H200); found SM{capability[0]}{capability[1]}."
    )
    batch_size = 1
    head_dim = 128
    device = torch.device("cuda:0")

    torch.manual_seed(42)

    q = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = F.logsigmoid(
        torch.randn(
            batch_size,
            seq_len,
            num_heads,
            device=device,
            dtype=torch.float32,
        )
    )
    beta = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        device=device,
        dtype=torch.float32,
    ).sigmoid()
    initial_state = torch.randn(
        batch_size,
        num_heads,
        head_dim,
        head_dim,
        device=device,
        dtype=torch.float32,
    )

    return GdnInput(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        scale=head_dim**-0.5,
    )


@pytest.fixture
def gdn_input_fixture() -> GdnInput:
    """Create the small deterministic GDN test workload."""
    return make_gdn_input(seq_len=1024, num_heads=4)


@pytest.fixture
def gdn_long_context_input_fixture() -> GdnInput:
    """Create the canonical long-context GDN test workload."""
    return make_gdn_input(seq_len=16_384, num_heads=16)


def test_flash_qla_backend_imports() -> None:
    """Verify that the Flash QLA runtime and FLA adapter are importable."""
    from flash_qla import chunk_gated_delta_rule as flash_qla_forward

    assert callable(flash_qla_forward)
    assert FlashQLABackend.backend_type == "flash_qla"


def test_fla_dispatches_to_flash_qla(
    monkeypatch: pytest.MonkeyPatch, gdn_input_fixture: GdnInput
) -> None:
    """Verify that FLA dispatches to the FlashQLA backend."""
    # List that records whether the FlashQLABackend.chunk_gated_delta_rule was called
    is_called: list[bool] = []

    monkeypatch.setattr(
        FlashQLABackend,
        "chunk_gated_delta_rule",
        record_calls(FlashQLABackend.chunk_gated_delta_rule, is_called),
    )
    monkeypatch.setenv("FLA_FLASH_QLA", "1")

    # Run GDN
    _ = run_forward(gdn_input_fixture)

    assert len(is_called) != 0, "FLA did not dispatch GDN forward to FlashQLA."


def test_fla_dispatches_to_fla_triton(
    monkeypatch: pytest.MonkeyPatch, gdn_input_fixture: GdnInput
) -> None:
    """Verify that FLA dispatches to the FLA Triton backend."""
    # List that records whether the FlashQLABackend.chunk_gated_delta_rule was called
    is_called: list[bool] = []

    monkeypatch.setattr(
        FlashQLABackend,
        "chunk_gated_delta_rule",
        record_calls(FlashQLABackend.chunk_gated_delta_rule, is_called),
    )

    # Set the env variable to disable FlashQLA and force FLA to use its Triton implementation
    monkeypatch.setenv("FLA_FLASH_QLA", "0")

    # Run GDN
    _ = run_forward(gdn_input_fixture)

    assert len(is_called) == 0, "FLA should not dispatch GDN to FlashQLA."


def test_flash_qla_matches_fla_triton(
    monkeypatch: pytest.MonkeyPatch,
    gdn_input_fixture: GdnInput,
) -> None:
    """Verify that FlashQLA matches FLA's Triton implementation for GDN."""
    monkeypatch.setenv("FLA_FLASH_QLA", "0")
    golden_output, golden_final_state = run_forward(gdn_input_fixture)

    monkeypatch.setenv("FLA_FLASH_QLA", "1")
    flash_qla_output, flash_qla_final_state = run_forward(gdn_input_fixture)

    assert_close("output", golden_output, flash_qla_output, ratio=_FLASH_QLA_ERROR_RATIO)
    assert_close(
        "final_state", golden_final_state, flash_qla_final_state, ratio=_FLASH_QLA_ERROR_RATIO
    )


def test_flash_qla_matches_fla_triton_long_context(
    monkeypatch: pytest.MonkeyPatch,
    gdn_long_context_input_fixture: GdnInput,
) -> None:
    """Verify FlashQLA against FLA Triton for the long-context workload."""
    monkeypatch.setenv("FLA_FLASH_QLA", "0")
    golden_output, golden_final_state = run_forward(gdn_long_context_input_fixture)

    monkeypatch.setenv("FLA_FLASH_QLA", "1")
    flash_qla_output, flash_qla_final_state = run_forward(gdn_long_context_input_fixture)

    assert_close("output", golden_output, flash_qla_output, ratio=_FLASH_QLA_ERROR_RATIO)
    assert_close(
        "final_state",
        golden_final_state,
        flash_qla_final_state,
        ratio=_FLASH_QLA_ERROR_RATIO,
    )
