FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    build-essential ninja-build pkg-config \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p /opt/conda \
 && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-lc"]

# Conda TOS 우회: defaults(Anaconda) 채널 제거 + conda-forge만 사용
RUN mkdir -p /etc/conda && \
    printf "channels:\n  - conda-forge\nchannel_priority: strict\ndefault_channels: []\n" > /etc/conda/.condarc

# =========================
# Env: sr (MambaIR/HAT/SwinIR/Swin2SR)
# =========================
RUN conda create -n sr -y --override-channels -c conda-forge python=3.10

# setuptools 82+에서 pkg_resources 제거 → <82로 고정
RUN conda run -n sr python -m pip install -U \
    "pip<25" \
    "setuptools<82" \
    wheel

# torch 2.0.1 cu117
RUN conda run -n sr pip install --index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# common deps (numpy<2 고정)
RUN conda run -n sr pip install \
    "numpy==1.26.4" \
    opencv-python scikit-image scipy \
    pyyaml tqdm requests einops lmdb h5py addict future yapf packaging

# timm/transformers
RUN conda run -n sr pip install \
    timm==0.9.16 transformers==4.38.1 huggingface-hub tensorboard

# mamba deps (build isolation OFF: torch import 필요)
RUN conda run -n sr pip install --no-build-isolation \
    mamba-ssm==1.0.1 causal-conv1d==1.0.0

# HAT deps (build isolation OFF)
RUN conda run -n sr pip install --no-build-isolation \
    basicsr==1.3.4.9

# =========================
# Env: dat (DAT 전용)
# =========================
RUN conda create -n dat -y --override-channels -c conda-forge python=3.8

# dat env도 동일하게 setuptools<82 고정
RUN conda run -n dat python -m pip install -U \
    "pip<25" \
    "setuptools<82" \
    wheel

# torch 1.8.0 cu111 (DAT 전용)
RUN conda run -n dat pip install --index-url https://download.pytorch.org/whl/cu111 \
    torch==1.8.0+cu111 torchvision==0.9.0+cu111

RUN conda run -n dat pip install \
    "numpy==1.23.5" \
    opencv-python scikit-image scipy \
    pyyaml tqdm requests einops lmdb h5py addict future yapf packaging \
    timm==0.4.12

# =========================
# Env: eval (IQA/metrics)
# =========================
RUN conda create -n eval -y --override-channels -c conda-forge python=3.10

RUN conda run -n eval python -m pip install -U \
    "pip<25" \
    "setuptools<82" \
    wheel

# torch 2.0.1 cu117 (eval)
RUN conda run -n eval pip install --index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# pyiqa + common deps (numpy<2)
RUN conda run -n eval pip install \
    "numpy==1.26.4" \
    pillow opencv-python scikit-image scipy \
    timm transformers huggingface-hub einops tqdm requests \
    pyiqa

# =========================
# Env: flux2 
# =========================

RUN conda create -n flux2 -y --override-channels -c conda-forge python=3.10

RUN conda run -n flux2 python -m pip install -U \
    "pip<25" \
    "setuptools<82" \
    wheel

# torch 2.0.1 cu117 (matches CUDA 11.7 base image)
RUN conda run -n flux2 pip install --index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Flux2 LoRA trainer deps
RUN conda run -n flux2 pip install \
    "numpy<2" \
    pillow tqdm packaging pyyaml requests \
    accelerate>=0.31.0 \
    transformers>=4.41.2 \
    peft>=0.11.1 \
    datasets \
    sentencepiece \
    ftfy \
    tensorboard \
    Jinja2 \
    bitsandbytes \
    prodigyopt \
    huggingface-hub \
    wandb \
    safetensors

# diffusers (Flux2 pipeline is on dev/main)
RUN conda run -n flux2 pip install \
    "diffusers @ git+https://github.com/huggingface/diffusers.git"

# =========================
# Convenience wrappers
# =========================
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'source /opt/conda/etc/profile.d/conda.sh' \
'conda run -n sr --no-capture-output "$@"' \
> /usr/local/bin/sr && chmod +x /usr/local/bin/sr

RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'source /opt/conda/etc/profile.d/conda.sh' \
'conda run -n dat --no-capture-output "$@"' \
> /usr/local/bin/dat && chmod +x /usr/local/bin/dat

RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'source /opt/conda/etc/profile.d/conda.sh' \
'conda run -n eval --no-capture-output "$@"' \
> /usr/local/bin/eval && chmod +x /usr/local/bin/eval

RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'source /opt/conda/etc/profile.d/conda.sh' \
'conda run -n flux2 --no-capture-output "$@"' \
> /usr/local/bin/flux2 && chmod +x /usr/local/bin/flux2

CMD ["bash"]
