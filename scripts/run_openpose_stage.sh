#!/bin/bash
# Ensure OpenPose binary is built and optionally run the CLI helper.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OPENPOSE_SCRIPT="$ROOT_DIR/scripts/openpose_setup.sh"

log() {
    local level="$1"; shift
    printf '[%s] %s\n' "$level" "$*"
}

if [[ ! -f "$OPENPOSE_SCRIPT" ]]; then
    log WARN "OpenPose setup script not found at scripts/openpose_setup.sh; skipping OpenPose stage."
    exit 0
fi

OPENPOSE_TARGET_VALUE="${OPENPOSE_TARGET:-${1:-data}}"
shift || true

log INFO "Preparing OpenPose assets for target: $OPENPOSE_TARGET_VALUE"
"$OPENPOSE_SCRIPT" "$OPENPOSE_TARGET_VALUE" "$@"
