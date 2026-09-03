"""Profile Qwen3.8-27B GDN cores through FlashQLA."""

import os

from profile_flashqla.isolate_jit_caches import isolate_jit_caches

if __name__ == "__main__":
    with isolate_jit_caches(prefix="profile-qwen38-27b-flash-qla-"):
        # FLA backend 선택은 torch/Transformers/FLA import보다 먼저 끝나야 한다.
        os.environ["FLA_FLASH_QLA"] = "1"

        from profile_flashqla.qwen38_27b_profile import ProfileBackend, main

        raise SystemExit(main(ProfileBackend.FLASH_QLA))
