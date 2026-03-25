#!/bin/bash
# Orchestrate the full multimodal pipeline by chaining stage scripts.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PIPELINE_ARGS=("$@")

# If the first argument looks like a path (no leading dash), treat it as the data directory.
if [[ "${#PIPELINE_ARGS[@]}" -gt 0 ]]; then
    first_arg="${PIPELINE_ARGS[0]}"
    if [[ "$first_arg" != --* ]]; then
        OPENPOSE_TARGET="${OPENPOSE_TARGET:-$first_arg}"
        PIPELINE_ARGS=("--data-dir" "$first_arg" "${PIPELINE_ARGS[@]:1}")
    fi
fi

OPENPOSE_TARGET="${OPENPOSE_TARGET:-data}"
export OPENPOSE_TARGET

STAGES=(
    "scripts/install_dependencies.sh"
    "scripts/run_openpose_stage.sh"
    "scripts/run_pipeline_stage.sh"
)

for stage in "${STAGES[@]}"; do
    if [[ ! -x "$ROOT_DIR/$stage" ]]; then
        echo "[WARN] Stage script $stage is missing or not executable; skipping." >&2
        continue
    fi

    case "$stage" in
        "scripts/run_openpose_stage.sh")
            "$ROOT_DIR/$stage"
            ;;
        "scripts/run_pipeline_stage.sh")
            "$ROOT_DIR/$stage" "${PIPELINE_ARGS[@]}"
            ;;
        *)
            "$ROOT_DIR/$stage"
            ;;
    esac
done
