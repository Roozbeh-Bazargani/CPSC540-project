FROM nvidia/cuda:11.3.1-runtime-ubuntu18.04
MAINTAINER Wanwen Chen

ARG https_proxy
ARG http_proxy

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    curl \
    vim

RUN apt-get install -y \
    python3-numpy \
    python3-pip


RUN pip3 install --upgrade pip

# Install dependency
RUN pip3 --no-cache-dir install \
    numpy \
    opencv-python \
    SimpleITK \
    scikit-image \
    scipy \
    pillow \
    scikit-learn \
    matplotlib


# Install pytorch 1.10 cuda 10.2
RUN pip3 --no-cache-dir install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Fix opencv lib error
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Set the library path to use cuda and cupti
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH


RUN apt-get update

# Install staintools
RUN apt-get -y install libblas-dev liblapack-dev gfortran
RUN pip3 install spams
RUN pip3 install staintools

# Install other dependencies
RUN pip3 install tqdm
RUN pip3 install tensorboard
RUN pip install jupyterlab
RUN pip install notebook