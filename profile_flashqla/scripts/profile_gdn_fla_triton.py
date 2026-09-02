"""Profile GDN forward through FLA's Triton implementation."""

from profile_flashqla.utils import isolate_jit_caches

if __name__ == "__main__":
    with isolate_jit_caches(prefix="profile-gdn-fla-triton-"):
        # Cache 환경 설정 뒤에만 torch/FLA/FlashQLA를 import한다.
        from profile_flashqla.gdn_profile import ProfileBackend, main

        raise SystemExit(main(ProfileBackend.FLA_TRITON))
