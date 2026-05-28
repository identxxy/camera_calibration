#!/usr/bin/env bash
# Build the camera_calibration build environment image.
# Tag: camcalib:dev
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t camcalib:dev -f "${HERE}/Dockerfile" "${HERE}"
