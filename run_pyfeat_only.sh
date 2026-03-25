#!/bin/bash
# Helper to run only the Py-Feat vision features against videos in ./data.

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

DATA_DIR="${1:-data}"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "[ERROR] Data directory not found: $DATA_DIR" >&2
    exit 1
fi

# Allow callers to pass additional run_pipeline.py flags after the data dir argument.
shift || true

echo "[INFO] Running Py-Feat vision extraction on videos under '$DATA_DIR'..."
poetry run python run_pipeline.py --data-dir "$DATA_DIR" --features pyfeat_vision "$@"

echo "[INFO] Py-Feat feature extraction completed."
