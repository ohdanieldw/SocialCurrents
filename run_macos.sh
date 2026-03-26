#!/usr/bin/env bash
# run_macos.sh — run the multimodal pipeline inside the pipeline-env conda environment
#
# Usage:
#   bash run_macos.sh -i data/ -o output/
#   bash run_macos.sh --list-features
#   bash run_macos.sh --check-dependencies
#   bash run_macos.sh -i data/ -f basic_audio,librosa_spectral,mediapipe_pose_vision
#
# Run 'bash run_macos.sh --help' for all options.
# If the environment does not exist yet, run: bash setup_macos.sh

set -euo pipefail

ENV_NAME="pipeline-env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── locate conda ─────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Anaconda or Miniconda first." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── check the env exists ─────────────────────────────────────────────────────
if ! conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "ERROR: conda env '$ENV_NAME' not found." >&2
    echo "Run 'bash setup_macos.sh' to create it." >&2
    exit 1
fi

# ── run pipeline ─────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"
conda run --no-capture-output -n "$ENV_NAME" python run_pipeline.py "$@"
