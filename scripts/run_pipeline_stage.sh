#!/bin/bash
# Execute the multimodal pipeline via Poetry.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

log() {
    local level="$1"; shift
    printf '[%s] %s\n' "$level" "$*"
}

if ! command -v poetry >/dev/null 2>&1; then
    log ERROR "Poetry is required but not installed. Run scripts/install_dependencies.sh first."
    exit 1
fi

log INFO "Running multimodal data pipeline..."
poetry run python run_pipeline.py "$@"
log INFO "Pipeline run completed."
