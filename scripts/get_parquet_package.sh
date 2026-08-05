#!/usr/bin/env bash
#
# Fetch and build the external Chapel `Parquet` Mason package, then print the
# `chpl` arguments (C++ prerequisite headers/objects, include paths, Arrow link
# flags) and the Parquet module source path needed to compile it into the
# Arkouda server.
#
# This is used by the Makefile in place of `mason modules`, which does not work
# reliably with Chapel 2.4.
#
# Contract: ALL human-readable status is written to stderr. The ONLY thing
# written to stdout is a single line of `chpl` arguments, so a Make recipe can
# capture it with shell command substitution.
#
# Usage:
#   get_parquet_package.sh <install-dir>
#
# Environment overrides:
#   ARKOUDA_PARQUET_REPO      git URL to clone (default: chapel-lang/Parquet)
#   ARKOUDA_PARQUET_REF       branch or tag to check out (default: repo HEAD)
#   ARKOUDA_PARQUET_SRC_DIR   use an existing checkout instead of cloning
#   CHPL_HOME                 used to select Chapel's C++ compiler for prereqs

set -euo pipefail

log() { echo "get_parquet_package: $*" >&2; }

PARQUET_REPO="${ARKOUDA_PARQUET_REPO:-https://github.com/chapel-lang/Parquet}"
PARQUET_REF="${ARKOUDA_PARQUET_REF:-}"

INSTALL_DIR="${1:-${ARKOUDA_PARQUET_INSTALL_DIR:-}}"
if [[ -z "${INSTALL_DIR}" ]]; then
  log "ERROR: no install directory provided (pass it as the first argument)"
  exit 1
fi

if [[ -z "${CHPL_HOME:-}" ]]; then
  if ! command -v chpl >/dev/null 2>&1; then
    log "ERROR: CHPL_HOME is unset and chpl is not available on PATH"
    exit 1
  fi
  CHPL_HOME="$(chpl --print-chpl-home)"
fi

# Allow pointing at an existing checkout (e.g. a Mason clone) to skip cloning.
PARQUET_SRC="${ARKOUDA_PARQUET_SRC_DIR:-${INSTALL_DIR}}"

if [[ -f "${PARQUET_SRC}/Mason.toml" ]]; then
  log "Using existing Parquet checkout at ${PARQUET_SRC}"
else
  log "Cloning ${PARQUET_REPO}${PARQUET_REF:+@${PARQUET_REF}} into ${PARQUET_SRC}"
  mkdir -p "$(dirname "${PARQUET_SRC}")"
  git clone --depth 1 ${PARQUET_REF:+--branch "${PARQUET_REF}"} \
      "${PARQUET_REPO}" "${PARQUET_SRC}" >&2
fi

# Resolve to an absolute path so the emitted flags work from any CWD.
PARQUET_ROOT="$(cd "${PARQUET_SRC}" && pwd -P)"
PREREQ_DIR="${PARQUET_ROOT}/prereqs/cpp"
PARQUET_MODULE="${PARQUET_ROOT}/src/Parquet.chpl"

if [[ ! -d "${PREREQ_DIR}" ]]; then
  log "ERROR: expected C++ prerequisites at ${PREREQ_DIR}, but they are missing"
  exit 1
fi

if [[ ! -f "${PARQUET_MODULE}" ]]; then
  log "ERROR: expected Chapel module at ${PARQUET_MODULE}, but it is missing"
  exit 1
fi

log "Building C++ prerequisites in ${PREREQ_DIR}"
make -s -C "${PREREQ_DIR}" ARKOUDA_CHPL_HOME="${CHPL_HOME:-}" >&2

# Gather the chpl flags the package needs.
FLAGS="$(make -s -C "${PREREQ_DIR}" ARKOUDA_CHPL_HOME="${CHPL_HOME:-}" printchplflags)"
if [[ -z "${FLAGS}" ]]; then
  log "ERROR: the Parquet prerequisite build returned no Chapel flags"
  exit 1
fi

printf '%s %s\n' "${FLAGS}" "${PARQUET_MODULE}"
