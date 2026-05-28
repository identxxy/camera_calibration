#!/usr/bin/env bash
set -euo pipefail

# Run on t0. This needs sudo because the calibration app depends on system C++
# development packages, CIFS mount helpers, and optionally Docker.

sudo apt-get update
sudo apt-get install -y \
  build-essential gcc-11 g++-11 \
  cmake ninja-build pkg-config git wget curl ca-certificates \
  libboost-all-dev \
  libeigen3-dev \
  libglew-dev \
  libsuitesparse-dev \
  libpng-dev zlib1g-dev \
  qtbase5-dev qttools5-dev libqt5opengl5-dev libqt5x11extras5-dev \
  libopencv-dev \
  libgl1-mesa-dev libglu1-mesa-dev \
  libv4l-dev \
  ffmpeg \
  cifs-utils \
  docker.io

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

sudo systemctl enable --now docker
TARGET_USER="${SUDO_USER:-${USER}}"
sudo usermod -aG docker "${TARGET_USER}"

mkdir -p "${HOME}/src" "${HOME}/cameras_mount" "${HOME}/calib_data"

echo "Base packages installed."
echo "Added ${TARGET_USER} to the docker group."
echo "Log out and back in on t0 for docker group membership to take effect."
echo "Next: run scripts/ops/t0_build_opengv_and_project.sh from the repo."
