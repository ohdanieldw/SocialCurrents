#!/bin/bash
# Ensure an OpenPose binary exists and export OPENPOSE_BIN for the pipeline.

set -euo pipefail

# Configurable inputs with sane defaults for WSL.
OPENPOSE_ROOT="${OPENPOSE_ROOT:-$HOME/openpose}"
OPENPOSE_REPO="${OPENPOSE_REPO:-https://github.com/CMU-Perceptual-Computing-Lab/openpose.git}"
OPENPOSE_BUILD_TYPE="${OPENPOSE_BUILD_TYPE:-Release}"
OPENPOSE_BUILD_DIR="$OPENPOSE_ROOT/build"
OPENPOSE_BIN_DEFAULT="$OPENPOSE_BUILD_DIR/examples/openpose/openpose.bin"
OPENPOSE_USE_CUDA="${OPENPOSE_USE_CUDA:-OFF}"
OPENPOSE_GPU_MODE="${OPENPOSE_GPU_MODE:-CPU_ONLY}"
OPENPOSE_MODEL_FOLDER="${OPENPOSE_MODEL_FOLDER:-$OPENPOSE_ROOT/models}"

export OPENPOSE_USE_CUDA
export OPENPOSE_GPU_MODE
export OPENPOSE_MODEL_FOLDER

log() {
    local level="$1"; shift
    printf '[%s] %s\n' "$level" "$*"
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log ERROR "Required command '$cmd' not found in PATH."
        exit 1
    fi
}

check_optional_lib() {
    local cmd="$1"
    local install_msg="$2"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log WARNING "$install_msg"
    fi
}

check_openpose_prereqs() {
    require_cmd git
    require_cmd cmake
    require_cmd make

    check_optional_lib protoc "Protobuf compiler missing; install with 'sudo apt install protobuf-compiler libprotobuf-dev'."
    check_optional_lib pkg-config "pkg-config missing; install with 'sudo apt install pkg-config'."

    if command -v pkg-config >/dev/null 2>&1; then
        if ! pkg-config --exists gflags; then
            log WARNING "gflags development headers not found; install with 'sudo apt install libgflags-dev'."
        fi
        if ! pkg-config --exists libglog; then
            log WARNING "glog development headers not found; install with 'sudo apt install libgoogle-glog-dev'."
        fi
        if ! pkg-config --exists protobuf; then
            log WARNING "protobuf development libraries not detected; install with 'sudo apt install protobuf-compiler libprotobuf-dev'."
        fi
    fi

    if [[ ! -d /usr/include/boost ]]; then
        log WARNING "Boost headers not found; install with 'sudo apt install libboost-filesystem-dev libboost-system-dev libboost-thread-dev'."
    fi

    if ! pkg-config --exists hdf5 >/dev/null 2>&1 && [[ ! -d /usr/include/hdf5 ]]; then
        log WARNING "HDF5 development files not detected; install with 'sudo apt install libhdf5-dev'."
    fi
}

clone_openpose() {
    if [[ -d "$OPENPOSE_ROOT/.git" ]]; then
        log INFO "OpenPose repo already present at $OPENPOSE_ROOT"
        return
    fi

    log INFO "Cloning OpenPose into $OPENPOSE_ROOT"
    git clone --depth 1 "$OPENPOSE_REPO" "$OPENPOSE_ROOT"
}

patch_caffe_protobuf() {
    local caffe_io_file="$OPENPOSE_ROOT/3rdparty/caffe/src/caffe/util/io.cpp"
    if [[ -f "$caffe_io_file" ]] && grep -q "SetTotalBytesLimit(kProtoReadBytesLimit, 536870912)" "$caffe_io_file"; then
        log INFO "Patching Caffe protobuf API usage for compatibility"
        sed -i 's/SetTotalBytesLimit(kProtoReadBytesLimit, 536870912)/SetTotalBytesLimit(kProtoReadBytesLimit)/' "$caffe_io_file"
    fi
}

configure_openpose() {
    mkdir -p "$OPENPOSE_BUILD_DIR"
    pushd "$OPENPOSE_BUILD_DIR" >/dev/null

    log INFO "Configuring OpenPose build (${OPENPOSE_BUILD_TYPE})"
    cmake -DCMAKE_BUILD_TYPE="$OPENPOSE_BUILD_TYPE" \
          -DBUILD_PYTHON=OFF \
          -DUSE_CUDNN=OFF \
          -DUSE_CUDA="$OPENPOSE_USE_CUDA" \
        -DGPU_MODE="$OPENPOSE_GPU_MODE" \
          "$OPENPOSE_ROOT"

    popd >/dev/null
}

build_openpose() {
    pushd "$OPENPOSE_BUILD_DIR" >/dev/null

    local jobs
    if command -v nproc >/dev/null 2>&1; then
        jobs="$(nproc)"
    else
        jobs=1
    fi

    log INFO "Building OpenPose (this can take a while)"
    make -j"$jobs"

    popd >/dev/null
}

ensure_openpose_bin() {
    local desired_bin="${OPENPOSE_BIN:-}" 
    if [[ -n "$desired_bin" ]]; then
        if [[ -x "$desired_bin" ]]; then
            log INFO "OPENPOSE_BIN already configured at $desired_bin"
            OPENPOSE_BIN="$desired_bin"
            export OPENPOSE_BIN
            return
        fi
        log WARNING "OPENPOSE_BIN is set to $desired_bin but the file is missing."
    fi

    if [[ -x "$OPENPOSE_BIN_DEFAULT" ]]; then
        log INFO "Using existing OpenPose binary at $OPENPOSE_BIN_DEFAULT"
        OPENPOSE_BIN="$OPENPOSE_BIN_DEFAULT"
        export OPENPOSE_BIN
        return
    fi

    check_openpose_prereqs

    clone_openpose
    patch_caffe_protobuf
    configure_openpose
    build_openpose

    if [[ ! -x "$OPENPOSE_BIN_DEFAULT" ]]; then
        log ERROR "Expected OpenPose binary $OPENPOSE_BIN_DEFAULT not found after build."
        exit 1
    fi

    OPENPOSE_BIN="$OPENPOSE_BIN_DEFAULT"
    export OPENPOSE_BIN
    log INFO "OPENPOSE_BIN set to $OPENPOSE_BIN"

    if [[ ! -d "$OPENPOSE_MODEL_FOLDER" ]]; then
        log WARNING "OpenPose model folder not found at $OPENPOSE_MODEL_FOLDER"
    fi
}

run_openpose_cli() {
    local target_dir="${1:-data}"
    shift || true
    if [[ ! -e "$target_dir" ]]; then
        log ERROR "OpenPose target path not found: $target_dir"
        exit 1
    fi

    require_cmd poetry

    log INFO "Running OpenPose CLI for $target_dir"
    poetry run python run_openpose_cli.py "$target_dir" "$@"
}

openpose_setup_main() {
    ensure_openpose_bin

    if [[ "${OPENPOSE_AUTO_RUN:-1}" == "1" ]]; then
        run_openpose_cli "$@"
    else
        log INFO "Skipping OpenPose CLI run (OPENPOSE_AUTO_RUN=$OPENPOSE_AUTO_RUN)"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    openpose_setup_main "$@"
fi
