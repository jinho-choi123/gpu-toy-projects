"""Compile and execute minimal Triton and CuTe kernels on an H100."""

import cutlass  # ty: ignore[unresolved-import]
from cutlass import cute  # ty: ignore[unresolved-import]
from cutlass.cute.runtime import from_dlpack  # ty: ignore[unresolved-import]
from loguru import logger
import torch
import triton
import triton.language as tl


EXPECTED_RESULT = 42.0


def _validate_result(stack: str, actual: float) -> None:
    if actual != EXPECTED_RESULT:
        message = (
            f"{stack} smoke failed: expected={EXPECTED_RESULT:g}, actual={actual:g}"
        )
        logger.error(message)
        raise RuntimeError(message)
    logger.info("{} smoke passed: result={:g}", stack, actual)


@triton.jit
def add_one_kernel(input_pointer, output_pointer):
    offset = tl.program_id(0)
    value = tl.load(input_pointer + offset)
    tl.store(output_pointer + offset, value + 1.0)


@cute.kernel
def smoke_kernel(output: cute.Tensor, value: cutlass.Float32):
    output[0] = value + cutlass.Float32(1.0)


@cute.jit
def launch_smoke(output: cute.Tensor, value: cutlass.Float32):
    smoke_kernel(output, value).launch(grid=(1, 1, 1), block=(1, 1, 1))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    capability = torch.cuda.get_device_capability()
    if capability != (9, 0):
        raise RuntimeError(f"expected H100 capability (9, 0), got {capability}")

    logger.info("device={} capability={}", torch.cuda.get_device_name(), capability)
    logger.info("cutlass={} cutlass-cuda={}", cutlass.__version__, cutlass.CUDA_VERSION)
    logger.info("torch={} torch-cuda={}", torch.__version__, torch.version.cuda)
    logger.info("triton={}", triton.__version__)

    input_tensor = torch.tensor([41.0], device="cuda")
    output_tensor = torch.empty_like(input_tensor)
    add_one_kernel[(1,)](input_tensor, output_tensor)
    torch.cuda.synchronize()
    _validate_result("Triton", output_tensor.item())

    cute_output_tensor = torch.empty_like(input_tensor)
    cute_output = from_dlpack(cute_output_tensor)
    launch_smoke(cute_output, cutlass.Float32(41.0))
    torch.cuda.synchronize()
    _validate_result("CuTe", cute_output_tensor.item())

    logger.success("GPU stack verification passed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("GPU stack verification failed")
        raise
