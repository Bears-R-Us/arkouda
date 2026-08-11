#!/bin/bash
# Shared helper for the build-*.sh scripts in this directory: detects which
# container CLI to use. Meant to be sourced, not executed directly, e.g.:
#   source "$SCRIPT_DIR/lib-docker-cmd.sh"
# Sets DOCKER_CMD unless it is already set in the environment.

# Auto-detect container CLI: prefer docker, fall back to podman.
# Override anytime by exporting DOCKER_CMD before sourcing this file.
if [ -n "${DOCKER_CMD:-}" ]; then
    :
elif command -v docker >/dev/null 2>&1; then
    DOCKER_CMD="docker"
elif command -v podman >/dev/null 2>&1; then
    DOCKER_CMD="podman"
else
    DOCKER_CMD="docker"
fi
