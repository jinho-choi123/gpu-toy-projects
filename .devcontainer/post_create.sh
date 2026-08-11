#!/usr/bin/env bash
set -euo pipefail

if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

# shellcheck source=/dev/null
source "$HOME/.cargo/env"

if ! command -v cargo-binstall >/dev/null 2>&1; then
  curl -L --proto '=https' --tlsv1.2 -sSf \
    https://raw.githubusercontent.com/cargo-bins/cargo-binstall/ead08b90bd7b2e6d81963fb9cf0b7239f66d5db4/install-from-binstall-release.sh \
    | bash
fi

if ! command -v zellij >/dev/null 2>&1; then
  cargo binstall --no-confirm zellij
fi

if [[ ! -d "$HOME/.oh-my-zsh" ]]; then
  RUNZSH=no CHSH=no KEEP_ZSHRC=yes \
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
    "" --unattended
fi

cargo --version
cargo-binstall -V
zellij --version
