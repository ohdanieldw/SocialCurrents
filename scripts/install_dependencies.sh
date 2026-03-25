#!/bin/bash
# Install project dependencies using Poetry.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

log() {
    local level="$1"; shift
    printf '[%s] %s\n' "$level" "$*"
}

if ! command -v poetry >/dev/null 2>&1; then
    log ERROR "Poetry is required but not installed. See https://python-poetry.org/docs/#installation"
    exit 1
fi

INSTALL_ARGS=()
if [[ -n "${POETRY_INSTALL_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    INSTALL_ARGS=(${POETRY_INSTALL_ARGS})
fi

log INFO "Installing project dependencies via Poetry..."
poetry install "${INSTALL_ARGS[@]}"
log INFO "Poetry dependencies installed."
