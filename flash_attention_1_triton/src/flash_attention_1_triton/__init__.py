"""Flash Attention 1 Triton package."""

from beartype.claw import beartype_this_package

beartype_this_package()

from ._flash_attention_func import flash_attention_func  # noqa: E402
from ._flash_attention_varlen_func import flash_attention_varlen_func  # noqa: E402

__all__ = ["flash_attention_func", "flash_attention_varlen_func"]
