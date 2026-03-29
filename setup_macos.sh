#!/usr/bin/env bash
# setup_macos.sh — one-time setup for the multimodal pipeline on macOS
#
# Usage:
#   bash setup_macos.sh              # creates conda env "pipeline-env"
#   bash setup_macos.sh --name myenv # custom env name

set -euo pipefail

ENV_NAME="pipeline-env"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) ENV_NAME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash setup_macos.sh [--name ENV_NAME]"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

log()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; }

# ── check conda ─────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    err "conda not found. Install Anaconda or Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── check / install ffmpeg ──────────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    log "ffmpeg already installed: $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f1-3)"
else
    if command -v brew &>/dev/null; then
        log "Installing ffmpeg via Homebrew..."
        brew install ffmpeg
    else
        err "ffmpeg not found and Homebrew is not available."
        err "Install Homebrew (https://brew.sh) then re-run, or install ffmpeg manually."
        exit 1
    fi
fi

# ── create conda environment ─────────────────────────────────────────────────
if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    warn "Conda env '$ENV_NAME' already exists — skipping creation."
    warn "To recreate: conda env remove -n $ENV_NAME && bash setup_macos.sh"
else
    log "Creating conda env '$ENV_NAME' (Python $PYTHON_VERSION)..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

run() { conda run --no-capture-output -n "$ENV_NAME" "$@"; }

# ── upgrade pip ──────────────────────────────────────────────────────────────
log "Upgrading pip..."
run pip install --quiet --upgrade pip setuptools wheel

# ── PyTorch (CPU, works on Intel and Apple Silicon macOS) ────────────────────
log "Installing PyTorch (CPU)..."
run pip install --quiet \
    "torch>=2.2,<3.0" \
    "torchaudio>=2.2,<3.0" \
    "torchvision>=0.17,<1.0" \
    --index-url https://download.pytorch.org/whl/cpu

# ── core scientific stack ────────────────────────────────────────────────────
log "Installing core scientific packages..."
run pip install --quiet \
    "numpy>=1.26,<2.0" \
    "pandas>=2.2" \
    "scipy>=1.11,<1.13" \
    "scikit-learn>=1.4.2" \
    "tqdm>=4.66" \
    "pyarrow>=14.0"

# ── audio ────────────────────────────────────────────────────────────────────
log "Installing audio packages..."
run pip install --quiet \
    "librosa>=0.10.1" \
    "soundfile>=0.12.1" \
    "opensmile>=2.5.0" \
    "audiostretchy>=1.3.5" \
    "ffmpeg-python>=0.2.0"

# ── speech / ASR ─────────────────────────────────────────────────────────────
log "Installing speech packages (speechbrain, whisperx)..."
run pip install --quiet "speechbrain>=1.0.0" || warn "speechbrain install failed — speech separation will be unavailable."
run pip install --quiet "openai-whisper" || warn "openai-whisper install failed."
run pip install --quiet "whisperx" || warn "whisperx install failed — WhisperX transcription will be unavailable."

# ── computer vision ──────────────────────────────────────────────────────────
log "Installing computer vision packages..."
run pip install --quiet \
    "opencv-python>=4.9.0,<4.10.0" \
    "mediapipe>=0.10.7" \
    "Pillow>=10.2" \
    "scikit-image>=0.22" \
    "matplotlib>=3.8" \
    "moviepy>=1.0.3"

# ── NLP / transformers ───────────────────────────────────────────────────────
log "Installing NLP packages..."
run pip install --quiet \
    "transformers[torch]>=4.44.2" \
    "sentence-transformers>=2.6.1" \
    "accelerate>=0.29.2" \
    "protobuf>=4.25" \
    "sentencepiece>=0.2.0"

# ── facial analysis ──────────────────────────────────────────────────────────
log "Installing facial analysis packages..."
run pip install --quiet "py-feat>=0.5.0" || warn "py-feat install failed."
run pip install --quiet \
    "git+https://github.com/FacePerceiver/facer.git@ddd35c76ff840174b8a5403ad1c1255e37b8782b" \
    || warn "pyfacer install failed — some facial analysis features will be unavailable."

# ── local packages (editable) ────────────────────────────────────────────────
log "Installing local packages in editable mode..."
run pip install --quiet -e "$SCRIPT_DIR/packages/audio_models"
run pip install --quiet -e "$SCRIPT_DIR/packages/nlp_models"
run pip install --quiet -e "$SCRIPT_DIR/packages/cv_models"
run pip install --quiet -e "$SCRIPT_DIR/packages/core_pipeline"

# ── done ─────────────────────────────────────────────────────────────────────
echo ""
log "Setup complete!"
echo ""
echo "  To run the pipeline:"
echo "    conda activate $ENV_NAME"
echo "    python extract.py -i data/ -o output/"
echo ""
echo "  Or use the convenience script (no manual activation needed):"
echo "    bash run_macos.sh -i data/ -o output/"
echo "    bash run_macos.sh --list-features"
echo "    bash run_macos.sh --check-dependencies"
echo ""
