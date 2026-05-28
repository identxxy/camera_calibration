#!/usr/bin/env bash
# Drop into an interactive shell inside camcalib:dev with the repo and any
# user-supplied data directory mounted.
#
# Usage:
#   ./docker/run_shell.sh                    # only the repo mounted
#   ./docker/run_shell.sh /home/vox/d/THAND  # also mounts a data dir at /data
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

DATA_MOUNT=()
if [[ $# -ge 1 ]]; then
  DATA_DIR="$(readlink -f "$1")"
  DATA_MOUNT=(-v "${DATA_DIR}:/data:ro")
fi

# X11 forwarding for --show_visualizations (optional). If $DISPLAY is unset,
# the GUI flag won't work, but the headless calibration pipeline still does.
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
if [[ -n "${DISPLAY:-}" ]]; then
  touch ${XAUTH} 2>/dev/null || true
  xauth nlist "${DISPLAY}" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f ${XAUTH} nmerge - 2>/dev/null || true
fi

docker run --rm -it \
    --gpus all \
    --network host \
    -v "${REPO_ROOT}:/workspace" \
    "${DATA_MOUNT[@]}" \
    -v ${XSOCK}:${XSOCK} \
    -v ${XAUTH}:${XAUTH} \
    -e DISPLAY="${DISPLAY:-}" \
    -e XAUTHORITY=${XAUTH} \
    -e QT_X11_NO_MITSHM=1 \
    -w /workspace \
    camcalib:dev \
    /bin/bash
