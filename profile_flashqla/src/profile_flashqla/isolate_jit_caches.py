"""Provide isolated JIT cache directories."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from tempfile import TemporaryDirectory

_CACHE_ENV_KEYS = (
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TILELANG_CACHE_DIR",
    "TILELANG_TMP_DIR",
)


@contextmanager
def isolate_jit_caches(prefix: str) -> Generator[None]:
    """Point JIT caches at a temporary directory for the context lifetime."""
    previous_values = {key: os.environ.get(key) for key in _CACHE_ENV_KEYS}

    with TemporaryDirectory(prefix=prefix) as cache_root:
        tilelang_cache_dir = os.path.join(cache_root, "tilelang")

        os.environ.update(
            {
                "TORCHINDUCTOR_CACHE_DIR": os.path.join(
                    cache_root,
                    "torchinductor",
                ),
                "TRITON_CACHE_DIR": os.path.join(cache_root, "triton"),
                "TILELANG_CACHE_DIR": tilelang_cache_dir,
                "TILELANG_TMP_DIR": os.path.join(tilelang_cache_dir, "tmp"),
            }
        )

        try:
            yield
        finally:
            for key, previous_value in previous_values.items():
                if previous_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous_value
