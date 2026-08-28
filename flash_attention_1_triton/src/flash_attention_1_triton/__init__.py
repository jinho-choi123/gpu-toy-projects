"""Flash Attention 1 Triton package."""

from beartype.claw import beartype_this_package

beartype_this_package()

from ._flash_attention import flash_attention  # noqa: E402
from ._flash_attention_varlen import flash_attention_varlen  # noqa: E402

__all__ = ["flash_attention", "flash_attention_varlen"]
