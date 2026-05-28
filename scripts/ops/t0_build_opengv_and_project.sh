#!/usr/bin/env bash
set -euo pipefail

# Run on t0 after scripts/ops/t0_install_calib_env.sh.

REPO="${REPO:-${HOME}/camera_calibration}"
PREFIX="${PREFIX:-${HOME}/.local}"
CUDA_ARCH="${CUDA_ARCH:-89}"
OPENGV_COMMIT="${OPENGV_COMMIT:-306a54e6c6b94e2048f820cdf77ef5281d4b48ad}"

mkdir -p "${HOME}/src" "${PREFIX}"

if [ ! -d "${HOME}/src/opengv/.git" ]; then
  git clone https://github.com/laurentkneip/opengv.git "${HOME}/src/opengv"
fi

git -C "${HOME}/src/opengv" fetch --tags
git -C "${HOME}/src/opengv" checkout "${OPENGV_COMMIT}"

cmake -S "${HOME}/src/opengv" -B "${HOME}/src/opengv/build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF \
  -DBUILD_PYTHON=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}"

cmake --build "${HOME}/src/opengv/build" -j"$(nproc)"
cmake --install "${HOME}/src/opengv/build"

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

CUDA_FLAGS=""
for arch in ${CUDA_ARCH//;/ }; do
  CUDA_FLAGS+="-gencode arch=compute_${arch},code=sm_${arch} "
done

cmake -S "${REPO}" -B "${REPO}/build_t0" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="${PREFIX}" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_FLAGS="${CUDA_FLAGS}"

cmake --build "${REPO}/build_t0" --target camera_calibration -j"$(nproc)"

set +e
"${REPO}/build_t0/applications/camera_calibration/camera_calibration" --help >/tmp/camera_calibration_help.txt
HELP_STATUS=$?
set -e
if ! grep -q -- "--apriltag_tower_config" /tmp/camera_calibration_help.txt; then
  echo "camera_calibration --help did not contain the expected tower flag." >&2
  exit 1
fi
if [ "${HELP_STATUS}" -ne 0 ]; then
  echo "Note: camera_calibration --help returned ${HELP_STATUS}; usage output was still generated."
fi
echo "Build OK: ${REPO}/build_t0/applications/camera_calibration/camera_calibration"
