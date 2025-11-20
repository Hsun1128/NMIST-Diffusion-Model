FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        python3-dev \
        git \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set the working directory
WORKDIR /app

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      torch==2.3.1+cu121 \
      torchvision==0.18.1+cu121 \
      torchaudio==2.3.1+cu121 \
      --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir imutils pytorch-fid scikit-learn xgboost \
     pandas tqdm tensorboardX tensorboard opencv-python pillow scipy numpy matplotlib