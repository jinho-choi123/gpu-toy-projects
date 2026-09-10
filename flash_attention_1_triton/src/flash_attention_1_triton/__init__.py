"""Flash Attention 1 Triton package."""

from beartype import BeartypeConf
from beartype.claw import beartype_this_package

# Triton compiles kernel source; Python runtime type-checking wrappers cannot run there.
beartype_this_package(
    conf=BeartypeConf(claw_skip_package_names=("flash_attention_1_triton._flash_attention_kernel",))
)

from ._flash_attention_func import flash_attention_func  # noqa: E402
from ._flash_attention_varlen_func import flash_attention_varlen_func  # noqa: E402

__all__ = ["flash_attention_func", "flash_attention_varlen_func"]
