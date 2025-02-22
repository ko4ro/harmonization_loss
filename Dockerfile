# FROM mcr.microsoft.com/devcontainers/base:ubuntu-20.04
FROM nvcr.io/nvidia/pytorch:24.03-py3

SHELL [ "bash", "-c" ]

# update apt and install packages
RUN apt update && \
    apt install -yq \
        ffmpeg \
        dkms \
        build-essential

# add user tools
RUN apt install -yq \
        jq \
        jp \
        tree \
        tldr

# add git-lfs and install
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -yq git-lfs && \
    git lfs install

# # KM Environment settings
ENV http_proxy ${http_proxy}
ENV https_proxy ${http_proxy}

# RUN pip install -r requirements.txt
# RUN pip install flash-attn
# RUN pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu
# RUN pip install -e .
