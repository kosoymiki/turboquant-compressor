#!/usr/bin/env bash
# Compile TurboQuant OpenCL kernels to SPIR-V modules.
# SPDX-License-Identifier: GPL-3.0-or-later
# LLVM Clang 21.1.8 + llvm-spirv 21.1.8 pipeline
set -euo pipefail

KERNEL_DIR="${1:-/data/data/com.termux/files/home/tmp_turboquant/native/opencl/kernels}"
OUT_DIR="${2:-/data/data/com.termux/files/home/tmp_turboquant/native/spirv/build}"

if [ -z "$KERNEL_DIR" ]; then
  echo "Usage: compile_spirv.sh <kernel-dir> <out-dir>" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

CLANG_BIN="${CLANG:-/data/data/com.termux/files/usr/bin/clang}"
LLVM_SPIRV_BIN="${LLVM_SPIRV:-/data/data/com.termux/files/usr/bin/llvm-spirv}"

SPIRV_TARGET="${SPIRV_TARGET:-1.2}"

log() { echo "[TQ_SPIRV] $*"; }
err() { echo "[TQ_SPIRV ERROR] $*" >&2; exit 1; }

log "Compiling kernels from ${KERNEL_DIR} to ${OUT_DIR}"

for kernel in "$KERNEL_DIR"/*.cl; do
  base="$(basename "$kernel" .cl)"
  bc="${OUT_DIR}/${base}.bc"
  spv="${OUT_DIR}/${base}.spv"
  txt="${OUT_DIR}/${base}.spv.txt"

  log "  ${base}.cl -> ${base}.spv"

  # Step 1: OpenCL C -> LLVM IR (SPIR 64)
  "$CLANG_BIN" \
    -emit-llvm -x cl -O3 -fPIC \
    -target spir64-unknown-unknown \
    -Xclang "-cl-std=CL2.0" \
    -D__OPENCL_VERSION__=200 \
    -DKEPLER_CORE=0 \
    -I"${KERNEL_DIR}" \
    "$kernel" \
    -o "$bc" -c \
    2>&1 | grep -v "^warning:" || true

  if [ ! -f "$bc" ]; then
    err "Failed to generate IR for ${base}"
  fi

  # Step 2: LLVM IR -> SPIR-V
  "$LLVM_SPIRV_BIN" -o "$spv" "$bc" \
    2>&1 | grep -v "^warning:" || true

  if [ ! -f "$spv" ]; then
    err "Failed to generate SPIR-V for ${base}"
  fi

  # Step 3: spirv-dis for text dump
  if command -v spirv-dis >/dev/null 2>&1; then
    spirv-dis "$spv" > "$txt" 2>/dev/null || true
  fi

  # Step 4: spirv-val for validation
  if command -v spirv-val >/dev/null 2>&1; then
    spirv-val "$spv" 2>&1 || log "  WARNING: spirv-val failed for ${base}"
  fi

  rm -f "$bc"

  size=$(stat -c%s "$spv" 2>/dev/null | numfmt --to=iec || stat -f%z "$spv" 2>/dev/null | tr -d '\n')
  log "  Done: ${base}.spv (${size})"
done

log "All SPIR-V kernels compiled to ${OUT_DIR}"
ls -lh "${OUT_DIR}"/*.spv 2>/dev/null || true