# syntax=docker/dockerfile:1.7
#
# trashpanda training image, two flavors from one file:
#
#   pascal  PyTorch cu126 wheels  Pascal (sm_60/61) and Volta (sm_70), up to Hopper
#   modern  PyTorch cu130 wheels  Turing (sm_75) through Blackwell; host driver >= 580
#
# Build from the repository root:
#   docker build -f docker/Dockerfile --build-arg FLAVOR=pascal -t trashpanda:dev-pascal .
#   docker build -f docker/Dockerfile --build-arg FLAVOR=modern -t trashpanda:dev-modern .
#
# PyTorch wheels bundle the CUDA runtime, cuBLAS and cuDNN, so the base is plain
# Ubuntu; the NVIDIA Container Toolkit supplies the driver at run time.

ARG FLAVOR=modern
ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE} AS flavor-pascal
ENV TP_FLAVOR=pascal \
    TP_TORCH_INDEX=https://download.pytorch.org/whl/cu126

FROM ${BASE_IMAGE} AS flavor-modern
ENV TP_FLAVOR=modern \
    TP_TORCH_INDEX=https://download.pytorch.org/whl/cu130

FROM flavor-${FLAVOR}

# 2.14 is the last PyTorch line with Pascal wheels (cu126); keep both flavors on
# the same line so they train identically. Override to move the modern image on.
ARG TORCH_VERSION=2.14.*
ARG VCS_REF=unknown
ARG VERSION=0.1.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/hf \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv \
 && pip install --upgrade pip

# Torch first, in its own layer: it is ~3 GB and changes far less often than the code.
RUN pip install "torch==${TORCH_VERSION}" --index-url "${TP_TORCH_INDEX}" \
 && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'archs', torch.cuda.get_arch_list())"

WORKDIR /opt/trashpanda
COPY . .
RUN pip install ".[test,datasets,provenance]" \
 && python -c "import torch, trashpanda; assert torch.version.cuda, 'CPU torch was pulled in'; print('trashpanda OK on torch', torch.__version__)"

# Writable places for an arbitrary --user uid: the HF cache and scratch.
RUN mkdir -p /hf /work && chmod 1777 /hf /work
WORKDIR /work

LABEL org.opencontainers.image.title="trashpanda-${TP_FLAVOR}" \
      org.opencontainers.image.description="trashpanda small-LM training (${TP_FLAVOR} GPUs)" \
      org.opencontainers.image.source="https://github.com/AtA-Systems/trashpanda" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${VERSION}-${TP_FLAVOR}"

CMD ["python", "-m", "trashpanda", "--help"]
