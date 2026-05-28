# Build image for puzzlepaint/camera_calibration on modern hardware (RTX 30/40 series).
#
# Strategy: stay on CUDA 12.x + Ubuntu 22.04 so that we have a recent-enough nvcc
# for sm_86 (RTX 30xx) and sm_89 (RTX 40xx) while still having gcc-11 available
# for the older C++14 codebase. CUDA 13 drops several APIs the project uses
# (cusolver / cublas headers, OpenGL interop), so we deliberately do NOT use
# the cuda:13.x base image.
#
# The repo source is NOT copied into the image; mount it at /workspace at run
# time so the host stays the source of truth.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# ---- System packages ------------------------------------------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc-11 g++-11 \
        cmake ninja-build pkg-config git wget curl ca-certificates \
        # Project deps (Ubuntu 22.04 versions are compatible)
        libboost-all-dev \
        libeigen3-dev \
        libglew-dev \
        libsuitesparse-dev \
        libpng-dev zlib1g-dev \
        qtbase5-dev qttools5-dev libqt5opengl5-dev libqt5x11extras5-dev \
        # OpenGV build deps
        libopencv-dev \
        # Headless OpenGL (so the program can at least link / start without an X server)
        libgl1-mesa-dev libglu1-mesa-dev \
        # Live webcam input path — applications/camera_calibration/CMakeLists.txt:187
        # links unconditionally against libv4l2 (no CMake guard), so the header
        # must be present even when running only headless / file-based pipelines.
        libv4l-dev \
        # ffmpeg for extracting frames from mp4 inputs
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Force gcc-11 as default for both host + CUDA host compilation.
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# ---- Build OpenGV ---------------------------------------------------------- #
# The README pins commit 306a54e6c6b94e2048f820cdf77ef5281d4b48ad. We use that
# exact commit to avoid drift.
ARG OPENGV_COMMIT=306a54e6c6b94e2048f820cdf77ef5281d4b48ad
RUN cd /tmp && git clone https://github.com/laurentkneip/opengv.git && \
    cd opengv && git checkout ${OPENGV_COMMIT} && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_TESTS=OFF \
          -DBUILD_PYTHON=OFF \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          .. && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/opengv

# A few quality-of-life pieces.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdb less vim python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ---- Entrypoint ------------------------------------------------------------ #
WORKDIR /workspace
# Default to an interactive shell. Use `docker run --rm -it ... bash` so the
# user can run cmake/make manually inside the build/ folder.
CMD ["/bin/bash"]
