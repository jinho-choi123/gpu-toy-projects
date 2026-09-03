"""Profile Qwen3.8-27B GDN cores through FLA Triton."""

import os

from profile_flashqla.isolate_jit_caches import isolate_jit_caches

if __name__ == "__main__":
    with isolate_jit_caches(prefix="profile-qwen38-27b-fla-triton-"):
        # FLA backend 선택은 torch/Transformers/FLA import보다 먼저 끝나야 한다.
        os.environ["FLA_FLASH_QLA"] = "0"

        from profile_flashqla.qwen38_27b_profile import ProfileBackend, main

        raise SystemExit(main(ProfileBackend.FLA_TRITON))
