#!/usr/bin/env bash
# Compile TurboQuant OpenCL kernels to SPIR-V modules.
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

KERNEL_DIR="${1:-}"
OUT_DIR="${2:-}"

if [ -z "$KERNEL_DIR" ] || [ -z "$OUT_DIR" ]; then
  echo "Usage: compile_spirv.sh <kernel-dir> <out-dir>" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

if command -v clang >/dev/null 2>&1; then
  CLANG_BIN="$(command -v clang)"
elif [ -n "${LLVM_BIN:-}" ] && [ -x "${LLVM_BIN}/clang" ]; then
  CLANG_BIN="${LLVM_BIN}/clang"
else
  echo "[compile_spirv] clang not found" >&2
  exit 1
fi

LLVM_SPIRV_BIN=""
if command -v llvm-spirv >/dev/null 2>&1; then
  LLVM_SPIRV_BIN="$(command -v llvm-spirv)"
elif [ -n "${LLVM_BIN:-}" ] && [ -x "${LLVM_BIN}/llvm-spirv" ]; then
  LLVM_SPIRV_BIN="${LLVM_BIN}/llvm-spirv"
fi

for kernel in "$KERNEL_DIR"/*.cl; do
  base="$(basename "$kernel" .cl)"
  bc="$OUT_DIR/$base.bc"
  spv="$OUT_DIR/$base.spv"
  "$CLANG_BIN" \
    -target spirv64 \
    -x cl \
    -cl-std=CL3.0 \
    -Xclang -finclude-default-header \
    -emit-llvm -c "$kernel" -o "$bc"

  if [ -n "$LLVM_SPIRV_BIN" ]; then
    "$LLVM_SPIRV_BIN" "$bc" -o "$spv"
  else
    "$CLANG_BIN" -target spirv64 -c "$kernel" -o "$spv"
  fi
  rm -f "$bc"
done

echo "[compile_spirv] wrote modules to $OUT_DIR"
