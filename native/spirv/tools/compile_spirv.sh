#!/bin/bash
##
# TurboQuant SPIR-V Kernel Compilation Pipeline
# Compiles OpenCL C kernels to SPIR-V 1.0/1.2 for Adreno GPU via Mesa Rusticl
#
# Requirements: llvm-spirv, spirv-llvm-translator, spirv-tools
# Target: Adreno A7xx/A8xx (Qualcomm SM8475+)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="/data/data/com.termux/files/home/tmp_turboquant/native/opencl/kernels"
BUILD_DIR="${SCRIPT_DIR}/build"
SPIRV_DIR="${SCRIPT_DIR}"

CLANG="${CLANG:-/data/data/com.termux/files/usr/bin/clang}"
LLVM_SPIV="${LLVM_SPIV:-/data/data/com.termux/files/usr/bin/llvm-spirv}"

# SPIR-V target version
SPIRV_TARGET="1.2"
# SPIR-V generator ID (TurboQuant)
SPIRV_GEN_ID="0x13375"

# Kernel files to compile
KERNELS=(
    "tq_mse_score"
    "tq_qjl_score"
    "tq_value_dequant"
    "tq_attention_logits"
    "tq_attention_apply_values"
)

# Default: compile all kernels
COMPILE_ALL=1
KERNEL_FILTER=""

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [KERNEL...]

Compile OpenCL C kernels to SPIR-V for TurboQuant.

Options:
  -a, --all          Compile all kernels (default)
  -k, --kernel KRN   Compile specific kernel (by name, no ext)
  -d, --dump        Dump SPIR-V disassembly after compile
  -v, --verbose     Verbose output
  -h, --help        Show this help

Examples:
  $0 --all                   # Compile all kernels
  $0 tq_mse_score tq_qjl_score  # Compile specific kernels
  $0 --kernel tq_mse_score   # Same using --kernel flag

Output:
  build/tq_<kernel>.spv       — SPIR-V binary
  build/tq_<kernel>.spv.txt   — SPIR-V disassembly (with --dump)

EOF
    exit 0
}

log() { echo "[TQ_SPIRV] $*" >&2; }
err() { echo "[TQ_SPIRV ERROR] $*" >&2; exit 1; }

[[ -d "${KERNEL_DIR}" ]] || err "Kernel dir not found: ${KERNEL_DIR}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

compile_kernel() {
    local name="$1"
    local cl_file="${KERNEL_DIR}/${name}.cl"
    local spv_file="${BUILD_DIR}/${name}.spv"
    local txt_file="${BUILD_DIR}/${name}.spv.txt"
    local opt="${2:-}"

    [[ -f "${cl_file}" ]] || { err "Kernel source not found: ${cl_file}"; }

    log "Compiling ${name}.cl -> ${name}.spv"

    # Step 1: Compile OpenCL C -> LLVM IR
    local ir_file="${BUILD_DIR}/${name}.bc"
    local opt_ir="${BUILD_DIR}/${name}.opt.bc"

    log "  [1/3] clang -emit-llvm -fPIC (OpenCL C -> LLVM IR)"
    "${CLANG}" -emit-llvm -x cl -O3 -fPIC \
        -target spir64-unknown-unknown \
        -Xclang "-cl-std=CL2.0" \
        -D__OPENCL_VERSION__=200 \
        -DKEPLER_CORE=0 \
        -I"${KERNEL_DIR}" \
        "${cl_file}" \
        -o "${ir_file}" \
        -c \
        ${opt} \
        2>&1 | grep -v "^warning:" || true

    [[ -f "${ir_file}" ]] || { err "Failed to generate IR: ${ir_file}"; }

    # Step 2: SPIR-V translation via llvm-spirv
    local spv_binary="${spv_file}"
    log "  [2/3] llvm-spirv (LLVM IR -> SPIR-V 1.2)"
    "${LLVM_SPIV}" --spirv-target-env=CL2.0 \
        -o "${spv_binary}" \
        "${ir_file}" \
        2>&1 | grep -v "^warning:" || true

    [[ -f "${spv_binary}" ]] || { err "Failed to generate SPIR-V: ${spv_binary}"; }

    # Step 3: Optional disassembly dump
    if [[ "${DUMP:-0}" == "1" ]] || [[ "${opt}" == "--dump" ]]; then
        log "  [3/3] spirv-dis (SPIR-V -> text)"
        spirv-dis "${spv_binary}" > "${txt_file}" || true
    fi

    # Cleanup intermediate
    rm -f "${ir_file}" "${opt_ir}"

    log "  Done: ${name}.spv ($(stat -c%s "${spv_binary}" | numfmt --to=iec))"
}

DUMP=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--all) COMPILE_ALL=1; shift ;;
        -k|--kernel) KERNEL_FILTER="$2"; shift 2 ;;
        -d|--dump) DUMP=1; shift ;;
        -v|--verbose) VERBOSE=1; shift ;;
        -h|--help) usage ;;
        -*) err "Unknown flag: $1" ;;
        *) COMPILE_ALL=0; KERNEL_FILTER="${KERNEL_FILTER} $1"; shift ;;
    esac
done

if [[ "${COMPILE_ALL}" == "1" ]]; then
    log "Compiling all ${#KERNELS[@]} kernels..."
    for k in "${KERNELS[@]}"; do compile_kernel "$k"; done
else
    for k in ${KERNEL_FILTER}; do
        compile_kernel "$k"
    done
fi

log "All SPIR-V kernels compiled to ${BUILD_DIR}/"
ls -lh "${BUILD_DIR}"/*.spv 2>/dev/null || true