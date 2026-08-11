#!/usr/bin/env bash
set -euo pipefail

npx -y skills add https://github.com/obra/superpowers --all
npx -y skills add mattpocock/skills --all
npx -y skills add https://github.com/juliusbrussee/caveman --all
curl -fsSL https://raw.githubusercontent.com/Egonex-AI/Understand-Anything/main/install.sh | bash -s codex
