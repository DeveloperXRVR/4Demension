FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE=/workspace

# System dependencies (ffmpeg for frame extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${WORKSPACE}

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install Depth Anything V3
RUN git clone https://github.com/ByteDance-Seed/depth-anything-3.git ${WORKSPACE}/depth-anything-3 && \
    cd ${WORKSPACE}/depth-anything-3 && \
    pip install --no-cache-dir --ignore-installed -e .

# Model weights will be downloaded at runtime (first job start)
# This avoids build-time HuggingFace auth issues

# Copy handler
COPY handler.py ${WORKSPACE}/handler.py

# RunPod serverless entrypoint
CMD ["python3", "-u", "/workspace/handler.py"]
