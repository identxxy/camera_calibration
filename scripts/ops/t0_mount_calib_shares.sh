#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for older notes and shell history. Prefer
# t0_mount_camera_shares.sh for new usage.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/t0_mount_camera_shares.sh" "$@"
