#!/usr/bin/env bash
# Configure + build the camera_calibration project inside the container.
# Run this from /workspace (the repo root) after `run_shell.sh` puts you in.
#
# We target sm_89 (RTX 40xx) by default. Add more archs via CUDA_ARCH if needed
# (semicolon-separated, e.g. "75;86;89").
set -euo pipefail

CUDA_ARCH="${CUDA_ARCH:-89}"
BUILD_DIR="${BUILD_DIR:-build_docker}"
JOBS="${JOBS:-$(nproc)}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

CUDA_FLAGS=""
for a in ${CUDA_ARCH//;/ }; do
  CUDA_FLAGS+="-gencode arch=compute_${a},code=sm_${a} "
done

cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CUDA_FLAGS="${CUDA_FLAGS}" \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
      ..

# Limit memory by reducing -j if your machine is small.
make -j"${JOBS}" camera_calibration
