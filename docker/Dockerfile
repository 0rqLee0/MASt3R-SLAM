# MASt3R-SLAM (CVPR'25) baseline image. Build from repo root:
#   nice docker build -f docker/Dockerfile -t mast3r-slam:latest .
# Checkpoints are NOT baked in: mount docker/cache -> /opt/MASt3R-SLAM/checkpoints at run time.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.6;8.9" \
    MAX_JOBS=16 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ninja-build libgl1 libglib2.0-0 libegl1 libgles2 \
      libglfw3 libx11-6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install "setuptools==70.0.0" wheel cython ninja "numpy==1.26.4"

WORKDIR /opt/MASt3R-SLAM
COPY . /opt/MASt3R-SLAM
# Build machine has no GPU visible: force CUDA extension build and add sm_89 (RTX 40xx).
RUN sed -i 's/^has_cuda = torch.cuda.is_available()/has_cuda = True/' setup.py && \
    sed -i 's/"-gencode=arch=compute_86,code=sm_86",/"-gencode=arch=compute_86,code=sm_86", "-gencode=arch=compute_89,code=sm_89",/' setup.py

# plane_vio: loader for extract_images.py folders (calib.json + cam0/<ns>.png)
RUN cat docker/plane_vio_loader.py >> mast3r_slam/dataloader.py

# lietorch first (git dep of the main package; heavy CUDA build)
RUN pip install --no-build-isolation "lietorch @ git+https://github.com/princeton-vl/lietorch.git"
RUN pip install --no-build-isolation -e thirdparty/mast3r
# pyimgui 2.0 does not compile with Cython 3
RUN pip install "cython<3" && pip install --no-build-isolation -e thirdparty/in3d
RUN pip install --no-build-isolation -e . && pip install "numpy==1.26.4"
RUN python -c "import torch, lietorch, mast3r_slam_backends, curope; print('ok', torch.__version__)"
# pyrealsense2 (imported by the dataloader) needs libusb
RUN apt-get update && apt-get install -y --no-install-recommends libusb-1.0-0 && rm -rf /var/lib/apt/lists/*
# plane_vio: keyframe buffer (preallocated on GPU, ~7 MB/KF at 512x320) configurable; upstream default 512
RUN sed -i 's/^    keyframes = SharedKeyframes(manager, h, w)$/    keyframes = SharedKeyframes(manager, h, w, buffer=int(__import__("os").environ.get("MAST3R_KF_BUFFER", "512")))/' main.py && \
    grep -q MAST3R_KF_BUFFER main.py
