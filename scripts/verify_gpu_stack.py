"""Compile and execute minimal Triton and CuTe kernels on an H100."""

import cutlass  # ty: ignore[unresolved-import]
from cutlass import cute  # ty: ignore[unresolved-import]
import torch
import triton
import triton.language as tl


@triton.jit
def add_one_kernel(input_pointer, output_pointer):
    offset = tl.program_id(0)
    value = tl.load(input_pointer + offset)
    tl.store(output_pointer + offset, value + 1.0)


@cute.kernel
def smoke_kernel(value: cutlass.Float32):
    cute.printf("cute-smoke={}", value + cutlass.Float32(1.0))


@cute.jit
def launch_smoke(value: cutlass.Float32):
    smoke_kernel(value).launch(grid=(1, 1, 1), block=(1, 1, 1))


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is not available"

    capability = torch.cuda.get_device_capability()
    assert capability == (9, 0), f"expected H100 capability (9, 0), got {capability}"

    print(f"device={torch.cuda.get_device_name()} capability={capability}")
    print(f"cutlass={cutlass.__version__} cutlass-cuda={cutlass.CUDA_VERSION}")
    print(f"torch={torch.__version__} torch-cuda={torch.version.cuda}")
    print(f"triton={triton.__version__}")

    input_tensor = torch.tensor([41.0], device="cuda")
    output_tensor = torch.empty_like(input_tensor)
    add_one_kernel[(1,)](input_tensor, output_tensor)
    torch.cuda.synchronize()
    torch.testing.assert_close(output_tensor.cpu(), torch.tensor([42.0]))
    print(f"triton-smoke={output_tensor.item():g}")

    launch_smoke(cutlass.Float32(41.0))
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
