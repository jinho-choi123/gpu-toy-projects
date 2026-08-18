FROM nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
ARG DEBIAN_FRONTEND=noninteractive
ARG NSIGHT_SYSTEMS_VERSION=2026.2.1.210-3763964

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PATH=/home/${USERNAME}/.local/bin:/home/${USERNAME}/.bun/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        bubblewrap \
        build-essential \
        ca-certificates \
        clang \
        clang-tools \
        curl \
        git \
        gnupg \
        libatomic1 \
        openssh-server \
        perl \
        procps \
        python3 \
        python3-dev \
        python3-pip \
        ripgrep \
        sudo \
        tmux \
        unzip \
        util-linux \
        wget \
        zip \
        zsh \
    && install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && chmod a+r /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
      > /etc/apt/sources.list.d/nodesource.list \
    && curl -fsSL \
      -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
      https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    && chmod a+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh nodejs \
    && curl -fL \
      -o /tmp/nsight-systems-cli.deb \
      "https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_2/NsightSystems-linux-cli-public-${NSIGHT_SYSTEMS_VERSION}.deb" \
    && apt-get install -y --no-install-recommends /tmp/nsight-systems-cli.deb \
    && rm -f /tmp/nsight-systems-cli.deb \
    && rm -rf /var/lib/apt/lists/*

RUN groupmod --new-name ${USERNAME} ubuntu \
    && usermod --login ${USERNAME} --home /home/${USERNAME} --move-home --shell /bin/bash ubuntu \
    && sed -i "s/^ubuntu:/${USERNAME}:/" /etc/subuid /etc/subgid \
    && usermod --uid ${USER_UID} --gid ${USER_GID} ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && echo "${USERNAME}:vscode123" | chpasswd

RUN install -d -m 0755 /etc/ssh/sshd_config.d /run/sshd \
    && printf '%s\n' \
      'Port 3333' \
      'ListenAddress 127.0.0.1' \
      'PasswordAuthentication yes' \
      'PermitRootLogin no' \
      > /etc/ssh/sshd_config.d/00-devcontainer.conf \
    && rm -f /etc/ssh/ssh_host_*

RUN printf '%s\n' \
      '#!/usr/bin/env bash' \
      'exec sudo -E env PATH="$PATH" ncu "$@"' \
      > /usr/local/bin/ncu-root \
    && chmod 0755 /usr/local/bin/ncu-root

USER ${USERNAME}

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN curl -fsSL https://bun.com/install | bash

RUN curl -fsSL https://chatgpt.com/codex/install.sh | sh

# Install pnpm for understand-anything skill.
RUN npm install --global --prefix /home/${USERNAME}/.local pnpm

WORKDIR /workspace

CMD ["zsh"]
