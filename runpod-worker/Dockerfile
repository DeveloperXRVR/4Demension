FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE=/workspace

# Required for building CUDA extensions without a GPU present
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1
ENV MAX_JOBS=2

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget curl \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 \
    cmake build-essential \
    libboost-all-dev libeigen3-dev \
    libflann-dev libfreeimage-dev \
    libmetis-dev libgoogle-glog-dev \
    libgflags-dev libsqlite3-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libceres-dev \
    && rm -rf /var/lib/apt/lists/*

# Install COLMAP
RUN apt-get update && apt-get install -y colmap && rm -rf /var/lib/apt/lists/*

WORKDIR ${WORKSPACE}

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Clone and install Depth Anything V3
RUN git clone https://github.com/ByteDance-Seed/depth-anything-3.git ${WORKSPACE}/depth-anything-3 && \
    cd ${WORKSPACE}/depth-anything-3 && \
    pip3 install --no-cache-dir -e .

# Clone 3D Gaussian Splatting
RUN git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git ${WORKSPACE}/gaussian-splatting

# Build diff-gaussian-rasterization (CUDA extension)
RUN cd ${WORKSPACE}/gaussian-splatting && \
    pip3 install --no-cache-dir -e submodules/diff-gaussian-rasterization

# Build simple-knn (CUDA extension)
RUN cd ${WORKSPACE}/gaussian-splatting && \
    pip3 install --no-cache-dir -e submodules/simple-knn

# Download Depth Anything V3 model weights
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('depth-anything/Depth-Anything-V3-ViT-L', 'model.safetensors', local_dir='${WORKSPACE}/depth-anything-3/checkpoints')"

# Copy handler
COPY handler.py ${WORKSPACE}/handler.py

# RunPod serverless entrypoint
CMD ["python3", "-u", "/workspace/handler.py"]
