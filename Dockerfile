# ============================================================
# Stage 1: Builder
# ============================================================
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3.10-dev \
    build-essential \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libxrender1 libxext6 libsm6 \
    patchelf \
    gcc \
    cmake \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install jaxlib with CUDA support and jax BEFORE other packages
RUN pip install --no-cache-dir --upgrade \
    jaxlib==0.4.16+cuda12.cudnn89 \
    jax==0.4.16 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy source and install mujoco
COPY . /work/rl
WORKDIR /work/rl
RUN sh install_mujoco.sh

# Install mock envs
WORKDIR /work/rl/ensemble_offline_rl
RUN pip install --no-cache-dir --user -e .


# ============================================================
# Stage 2: Runtime
# ============================================================
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libosmesa6 \
    libgl1-mesa-glx \
    libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy mujoco binaries
COPY --from=builder /root/.mujoco /root/.mujoco

# Copy application source
COPY --from=builder /work/rl /work/rl

ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
ENV MUJOCO_GL=osmesa
ENV PATH=/root/.local/bin:$PATH

WORKDIR /work/rl/ensemble_offline_rl

CMD ["python"]
