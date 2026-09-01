"""Profile GDN forward through FLA's FlashQLA backend."""

import os
from tempfile import TemporaryDirectory

if __name__ == "__main__":
    with TemporaryDirectory(prefix="profile-gdn-flash-qla-") as cache_root:
        tilelang_cache_dir = os.path.join(cache_root, "tilelang")

        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_root, "torchinductor")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, "triton")
        os.environ["TILELANG_CACHE_DIR"] = tilelang_cache_dir
        os.environ["TILELANG_TMP_DIR"] = os.path.join(tilelang_cache_dir, "tmp")

        # Cache 환경 설정 뒤에만 torch/FLA/FlashQLA를 import한다.
        from profile_flashqla.gdn_profile import ProfileBackend, main

        raise SystemExit(main(ProfileBackend.FLASH_QLA))
