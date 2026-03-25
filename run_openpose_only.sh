#!/bin/bash
# Helper to run only the OpenPose vision extractor against videos in ./data.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v poetry >/dev/null 2>&1; then
    echo "[ERROR] Poetry is required but not installed. See https://python-poetry.org/docs/#installation" >&2
    exit 1
fi

if [[ "${SKIP_POETRY_INSTALL:-0}" != "1" ]]; then
    echo "[INFO] Installing project dependencies via Poetry..."
    poetry install
fi

DATA_TARGET="${1:-data}"
shift || true

if [[ ! -e "$DATA_TARGET" ]]; then
    echo "[ERROR] Input path not found: $DATA_TARGET" >&2
    exit 1
fi

if [[ -f "$ROOT_DIR/scripts/openpose_setup.sh" ]]; then
    # shellcheck disable=SC1090
    source "$ROOT_DIR/scripts/openpose_setup.sh"
    openpose_setup_main "$DATA_TARGET" "$@"
else
    if [[ -z "${OPENPOSE_BIN:-}" ]]; then
        echo "[ERROR] OPENPOSE_BIN environment variable is not set. Set it to the OpenPose binary (e.g. ~/openpose/build/examples/openpose/openpose.bin)." >&2
        exit 1
    fi

    echo "[INFO] Running OpenPose binary on '$DATA_TARGET'..."
    poetry run python run_openpose_cli.py "$DATA_TARGET" "$@"
    echo "[INFO] OpenPose processing completed."
fi
