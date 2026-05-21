#!/usr/bin/env bash
# Build TurboQuant OpenCL native sidecar in Termux
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NATIVE_DIR="$ROOT_DIR/native/opencl"
TQ_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
BUILD_DIR="${TQ_OPENCL_BUILD_DIR:-$TQ_CACHE_HOME/turboquant/native-opencl-build}"
LOCK_DIR="${TQ_CACHE_HOME}/turboquant/opencl-serial.lock"

acquire_lock() {
  mkdir -p "${TQ_CACHE_HOME}/turboquant"
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    sleep 1
  done
  trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT INT TERM
}

acquire_lock

echo "[build-opencl-native] Installing dependencies..."
pkg install -y clang cmake make opencl-headers ocl-icd 2>/dev/null || true

echo "[build-opencl-native] Configuring..."
mkdir -p "$BUILD_DIR"
cmake -S "$NATIVE_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  2>&1

echo "[build-opencl-native] Building..."
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)" 2>&1

if cmake --build "$BUILD_DIR" --target tq_spirv_kernels >/dev/null 2>&1; then
  echo "[build-opencl-native] SPIR-V modules generated in $BUILD_DIR/spirv"
else
  echo "[build-opencl-native] WARN: tq_spirv_kernels target unavailable or failed"
fi

if [ -f "$BUILD_DIR/tq_opencl_cli" ]; then
  echo "[build-opencl-native] SUCCESS: $BUILD_DIR/tq_opencl_cli"
else
  echo "[build-opencl-native] FAIL: binary not produced"
  exit 1
fi
