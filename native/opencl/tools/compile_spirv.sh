#!/usr/bin/env bash
# Compile TurboQuant OpenCL kernels to SPIR-V modules.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Usage:
#   compile_spirv.sh [--all] [--llvm-version N]
#
# Environment:
#   CLANG_BIN       - clang binary (default: clang-18 for CI, auto-detect)
#   LLVM_SPIRV_BIN  - llvm-spirv binary (default: llvm-spirv-18 for CI)
#   KERNEL_DIR      - kernel source dir
#   OUT_DIR         - output dir
#
# For Termux (aarch64): uses local clang-21/llvm-spirv-21
# For CI (x86_64): uses LLVM 18 from apt.llvm.org

set -euo pipefail

# Default LLVM version — 22 (latest stable on apt.llvm.org/noble)
# Termux uses 21.1.8, CI uses LLVM 22 for SPIR-V 1.3 + OpenCL 3.0
DEFAULT_LLVM_VERSION="22"

# Auto-detect Termux LLVM 21 vs CI LLVM 18
if [[ -x "/data/data/com.termux/files/usr/bin/clang" ]]; then
    CLANG_BIN="${CLANG_BIN:-/data/data/com.termux/files/usr/bin/clang}"
    LLVM_SPIRV_BIN="${LLVM_SPIRV_BIN:-/data/data/com.termux/files/usr/bin/llvm-spirv}"
    LLVM_VERSION=21
elif [[ -x "/usr/bin/clang-${DEFAULT_LLVM_VERSION}" ]]; then
    CLANG_BIN="${CLANG_BIN:-clang-${DEFAULT_LLVM_VERSION}}"
    LLVM_SPIRV_BIN="${LLVM_SPIRV_BIN:-llvm-spirv-${DEFAULT_LLVM_VERSION}}"
    LLVM_VERSION="${DEFAULT_LLVM_VERSION}"
else
    CLANG_BIN="${CLANG_BIN:-clang}"
    LLVM_SPIRV_BIN="${LLVM_SPIRV_BIN:-llvm-spirv}"
    LLVM_VERSION=0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="${KERNEL_DIR:-${SCRIPT_DIR}/../driver-pack/kernels}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/../../spirv/build}"

# Parse args
case "${1:-}" in
  --all)           shift ;;
  --llvm-version)  LLVM_VERSION="${2:-18}"; CLANG_BIN="clang-${LLVM_VERSION}"; LLVM_SPIRV_BIN="llvm-spirv-${LLVM_VERSION}"; shift 2 ;;
esac

log() { echo "[TQ_SPIRV v${LLVM_VERSION}] $*"; }
err() { echo "[TQ_SPIRV ERROR] $*" >&2; exit 1; }

mkdir -p "$OUT_DIR"

log "Compiling kernels from ${KERNEL_DIR} to ${OUT_DIR}"
log "Toolchain: $CLANG_BIN + $LLVM_SPIRV_BIN"

# Check tools exist
for tool in "$CLANG_BIN" "$LLVM_SPIRV_BIN"; do
    if [[ ! -x "$tool" ]]; then
        err "Tool not found: $tool (install llvm-${LLVM_VERSION})"
    fi
done

log ""

# Compile kernels
SUCCESS=0
FAILED=0

for kernel in "$KERNEL_DIR"/*.cl; do
    [[ -f "$kernel" ]] || continue

    base="$(basename "$kernel" .cl)"
    bc="${OUT_DIR}/${base}.bc"
    spv="${OUT_DIR}/${base}.spv"
    txt="${OUT_DIR}/${base}.spv.txt"

    log "Compiling: ${base}.cl"

    # Step 1: OpenCL C -> LLVM IR (SPIR 64)
    if "$CLANG_BIN" \
        -emit-llvm -x cl -O3 -fPIC \
        -target spir64-unknown-unknown \
        -Xclang "-cl-std=CL2.0" \
        -D__OPENCL_VERSION__=200 \
        -DKEPLER_CORE=0 \
        -I"${KERNEL_DIR}" \
        "$kernel" \
        -o "$bc" -c 2>&1 | grep -v "^warning:"; then
        log "  ✓ LLVM IR generated"
    else
        err "Failed to generate IR for ${base}"
    fi

    # Step 2: LLVM IR -> SPIR-V 1.2
    if "$LLVM_SPIRV_BIN" -o "$spv" "$bc" 2>&1 | grep -v "^warning:"; then
        log "  ✓ SPIR-V generated"
    else
        err "Failed to generate SPIR-V for ${base}"
    fi

    # Step 3: spirv-dis for text dump
    if command -v spirv-dis >/dev/null 2>&1; then
        if spirv-dis "$spv" > "$txt" 2>/dev/null; then
            log "  ✓ Disassembly: $txt"
        fi
    fi

    # Step 4: spirv-val for validation
    if command -v spirv-val >/dev/null 2>&1; then
        if spirv-val "$spv"; then
            log "  ✓ Valid SPIR-V 1.2"
        else
            log "  ⚠ SPIR-V validation failed"
        fi
    fi

    rm -f "$bc"

    size=$(stat -c%s "$spv" 2>/dev/null || stat -f%z "$spv" 2>/dev/null)
    log "  ✓ ${base}.spv (${size} bytes)"
    log ""
    ((SUCCESS++)) || true
done

log "═══════════════════════════════════════════"
log "Compilation complete: $SUCCESS succeeded, $FAILED failed"
log "Output: ${OUT_DIR}"
log "═══════════════════════════════════════════"
ls -lh "${OUT_DIR}"/*.spv 2>/dev/null || true

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi