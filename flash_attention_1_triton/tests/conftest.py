"""CUDA requirements and dtypes shared by the API tests."""

from collections.abc import Iterator

import pytest
import torch
from loguru import logger


@pytest.fixture(autouse=True)
def log_case(request: pytest.FixtureRequest) -> Iterator[None]:
    """Record the case parameters and completion even when a test fails.

    Args:
        request (pytest.FixtureRequest): Current test context, including its ID and parameters.

    Yields:
        None: Hand control to the test between START and END log messages.
    """
    params = getattr(request.node, "callspec", None)
    logger.info("START {} | parameters={}", request.node.nodeid, getattr(params, "params", {}))
    try:
        yield
    finally:
        logger.info("END {}", request.node.nodeid)


@pytest.fixture
def cuda() -> None:
    """Skip GPU tests when CUDA is unavailable.

    Returns:
        None: Continue fixture setup when CUDA is available; otherwise skip the test.
    """
    if not torch.cuda.is_available():
        logger.warning("Skipping GPU test: CUDA is unavailable")
        pytest.skip("CUDA is required for FlashAttention")


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request: pytest.FixtureRequest, cuda: None) -> torch.dtype:
    """Exercise both supported dtypes on capable hardware.

    Args:
        request (pytest.FixtureRequest): Parametrized context holding the requested torch dtype.
        cuda (None): Completed CUDA availability fixture.

    Returns:
        torch.dtype: torch.float16 or torch.bfloat16; unsupported CUDA execution fails.
    """
    return request.param
