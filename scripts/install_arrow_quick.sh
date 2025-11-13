#!/usr/bin/env bash
# install_arrow_quick.sh
# Quickly install prebuilt Apache Arrow/Parquet packages from local artifacts.
# Usage: ./install_arrow_quick.sh /path/to/build-dir
# Expects *.deb on Debian/Ubuntu, *.rpm on RHEL/Alma/Rocky.
# Falls back with clear errors if artifacts are missing.

set -euo pipefail

DEP_BUILD_DIR="${1:-/opt/dep/build}"

echo "Installing Apache Arrow/Parquet"
echo "from build directory: ${DEP_BUILD_DIR}"

if [[ ! -d "${DEP_BUILD_DIR}" ]]; then
  echo "ERROR: Build directory '${DEP_BUILD_DIR}' does not exist."
  echo "       Use 'make install-arrow' or create the directory and place artifacts there."
  exit 2
fi

# ---- OS detection ----
OS_ID=""
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  OS_ID="${ID,,}"
elif command -v lsb_release >/dev/null 2>&1; then
  OS_ID="$(lsb_release -is 2>/dev/null | tr '[:upper:]' '[:lower:]')"
else
  echo "ERROR: Cannot detect OS (no /etc/os-release, no lsb_release)."
  echo "       Use 'make install-arrow' (source build) instead."
  exit 1
fi

cd "${DEP_BUILD_DIR}"

# Ensure globs that don't match expand to nothing
shopt -s nullglob

is_root() {
  # POSIX-safe root detection
  [[ "${EUID:-$(id -u)}" -eq 0 ]]
}

case "${OS_ID}" in
  ubuntu|debian)
    debs=(./apache-arrow*.deb)
    if ((${#debs[@]} == 0)); then
      echo "ERROR: No apache-arrow*.deb files found in ${DEP_BUILD_DIR}."
      echo "       Place .deb artifacts here or use 'make install-arrow'."
      exit 3
    fi
    if ! is_root; then
      sudo apt-get update
      sudo apt-get install -y -V "${debs[@]}"
    else
      apt-get update
      apt-get install -y -V "${debs[@]}"
    fi
    ;;

  almalinux|rhel|centos|rocky|ol|fedora)
    rpms=(./apache-arrow*.rpm)
    if ((${#rpms[@]} == 0)); then
      echo "ERROR: No apache-arrow*.rpm files found in ${DEP_BUILD_DIR}."
      echo "       Place .rpm artifacts here or use 'make install-arrow'."
      exit 3
    fi
    if ! is_root; then
      sudo dnf install -y "${rpms[@]}"
    else
      dnf install -y "${rpms[@]}"
    fi
    ;;

  *)
    echo "ERROR: install-arrow-quick is not supported on OS '${OS_ID}'."
    echo "       Use 'make install-arrow' for a source build."
    exit 4
    ;;
esac

echo "✔ Arrow quick install complete."
