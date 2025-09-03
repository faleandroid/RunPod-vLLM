# ---------------------------------------------------------------------------- #
#               STANDALONE - Optional: Clone HuggingFace models                #
# ---------------------------------------------------------------------------- #
FROM alpine/git:2.47.2 AS clone
COPY builder/clone.sh /clone.sh

# Clone selected HuggingFace repo. Format is: RUN . /clone.sh /model-path https://huggingface.co/author/model branch-name
# RUN . /clone.sh /workspace/models/Qwen3-14B-AWQ https://huggingface.co/Qwen/Qwen3-14B-AWQ main
RUN . /clone.sh /workspace/models/Qwen2.5-VL-7B-Instruct-AWQ https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ main

# ---------------------------------------------------------------------------- #
#                          SHARED - Build final image                          #
# ---------------------------------------------------------------------------- #
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 AS build_final_image
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    VLLM_USE_PRECOMPILED=false \
    FLASHINFER_ENABLE_AOT=1 \
    TOKENIZERS_PARALLELISM=false \
    TORCH_CUDA_ARCH_LIST="12.0;12.0+PTX"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Update and install system packages
RUN apt-get update && apt-get upgrade -y && \
    apt install -y wget git && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Add the deadsnakes PPA and install Python 3.10
# RUN add-apt-repository ppa:deadsnakes/ppa -y && \
#     apt-get install python3.12-dev python3.12-venv python3-pip -y --no-install-recommends && \
#     rm /usr/bin/python3 && \
#     ln -s /usr/bin/python3.12 /usr/bin/python3 && \
#     apt-get autoremove -y && \
#     apt-get clean -y && \
#     rm -rf /var/lib/apt/lists/*

# Download and install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# IMPORTANT: install PyTorch NIGHTLY with CUDA 12.8 (includes sm_120)
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt && \
    pip install flash-attn --no-build-isolation

# FlashAttention compiled for sm_120
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation flash-attn

# vLLM built from source (until wheels include sm_120)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "vllm @ git+https://github.com/vllm-project/vllm.git"

# Cleanup
RUN apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------- #
#                      STANDALONE - Upload or move models                      #
# ---------------------------------------------------------------------------- #
# Use COPY for local files on your disk                                        #
#COPY models/Qwen3-14B-AWQ /workspace/models/Qwen3-14B-AWQ

# Or to move the downloaded HuggingFace files to the final image, if you chose this method
COPY --from=clone /workspace/models /workspace/models

# Or you can use wget to download file(s) from the internet, or individual HuggingFace files only
# Remember that these models usually need their json config files in the same folder for proper functionality
#RUN wget -O /workspace/models/Lamarck-14B-v0.7-bnb-4bit/model.safetensors https://huggingface.co/3wad/Lamarck-14B-v0.7-bnb-4bit/resolve/main/model.safetensors?download=true
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                     NETWORKVOLUME - Link network folder                      #
# ---------------------------------------------------------------------------- #
# When loading models from network volume or external place, you want to serialize them first
# using vllm[tensorizer], so they load much faster. See more about this topic here: https://docs.vllm.ai/en/stable/getting_started/examples/tensorize_vllm_model.html
#RUN pip install vllm[tensorizer]

## TODO: FINISH NETWORK VOLUME PART

# ---------------------------------------------------------------------------- #

RUN mkdir /src
WORKDIR /src
ADD src .

CMD ["python3", "handler.py"]
