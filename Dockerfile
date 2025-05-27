# ---------------------------------------------------------------------------- #
#               STANDALONE - Optional: Clone HuggingFace models                #
# ---------------------------------------------------------------------------- #
#FROM alpine/git:2.47.2 AS clone
#COPY builder/clone.sh /clone.sh

# Clone selected HuggingFace repo
#RUN . /clone.sh /workspace/models/ https://huggingface.co/unsloth/Qwen3-14B-unsloth-bnb-4bit

# ---------------------------------------------------------------------------- #
#                          SHARED - Build final image                          #
# ---------------------------------------------------------------------------- #
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS build_final_image
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    VLLM_USE_PRECOMPILED=true \
    FLASHINFER_ENABLE_AOT=1 \
    TOKENIZERS_PARALLELISM=false
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Update and install system packages
RUN apt-get update && apt-get upgrade -y && \
    apt install -y wget git python3-pip && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt && \
    pip install flashinfer-python==0.2.5 -i https://flashinfer.ai/whl/cu124/torch2.6/

# Cleanup
RUN apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------- #
#                      STANDALONE - Upload or move models                      #
# ---------------------------------------------------------------------------- #
# Use COPY for local files on your disk                                        #
COPY models/Qwen3-14B-unsloth-bnb-4bit /workspace/models/Qwen3-14B-unsloth-bnb-4bit

# Or to move the downloaded HuggingFace files to the final image, if you chose this method
#COPY --from=download /workspace/models /workspace/models

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