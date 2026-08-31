"""Profile GDN forward through FLA's Triton implementation."""

from profile_flashqla.gdn_profile import ProfileBackend, main

if __name__ == "__main__":
    raise SystemExit(main(ProfileBackend.FLA_TRITON))
