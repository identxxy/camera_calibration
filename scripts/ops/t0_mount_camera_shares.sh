#!/usr/bin/env bash
set -euo pipefail

# Mount Windows camera-data SMB shares on t0.
# Required env for all hosts:
#   SMB_PASS='...'
# Optional env:
#   SMB_DOMAIN='.'
#   SMB_SHARE='Dshare'
#   SMB_USER_W1='Administrator'
#   SMB_USER_W2='Administrator'
#   SMB_USER_W3='admin'
#   SMB_USER_W4='admin'

if [ -z "${SMB_PASS:-}" ]; then
  echo "Set SMB_PASS before running this script." >&2
  exit 2
fi

SMB_DOMAIN="${SMB_DOMAIN:-.}"
SMB_SHARE="${SMB_SHARE:-Dshare}"
MOUNT_ROOT="${MOUNT_ROOT:-${HOME}/cameras_mount}"
CRED_ROOT="${CRED_ROOT:-${HOME}/.smbcredentials}"

sudo -v

mkdir -p "${MOUNT_ROOT}" "${CRED_ROOT}"
chmod 700 "${CRED_ROOT}"

mount_one() {
  local host="$1"
  local ip="$2"
  local user="$3"
  local mount_dir="${MOUNT_ROOT}/${host}_D"
  local cred_file="${CRED_ROOT}/${host}"

  mkdir -p "${mount_dir}"
  cat > "${cred_file}" <<EOF
username=${user}
password=${SMB_PASS}
domain=${SMB_DOMAIN}
EOF
  chmod 600 "${cred_file}"

  if mountpoint -q "${mount_dir}"; then
    echo "Already mounted: ${mount_dir}"
    return
  fi

  sudo mount -t cifs "//${ip}/${SMB_SHARE}" "${mount_dir}" \
    -o "credentials=${cred_file},vers=3.1.1,iocharset=utf8,uid=$(id -u),gid=$(id -g),file_mode=0444,dir_mode=0555,ro,noperm"

  echo "Mounted: //${ip}/${SMB_SHARE} -> ${mount_dir}"
}

mount_one w1 192.168.2.1 "${SMB_USER_W1:-Administrator}"
mount_one w2 192.168.2.2 "${SMB_USER_W2:-Administrator}"
mount_one w3 192.168.2.3 "${SMB_USER_W3:-admin}"
mount_one w4 192.168.2.4 "${SMB_USER_W4:-admin}"

find "${MOUNT_ROOT}" -maxdepth 2 -type d | sort
