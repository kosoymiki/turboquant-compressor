#!/usr/bin/env bash
# Build TurboQuant OpenCL native sidecar in Termux
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NATIVE_DIR="$ROOT_DIR/native/opencl"
BUILD_DIR="$NATIVE_DIR/build"

echo "[build-opencl-native] Installing dependencies..."
pkg install -y clang cmake make opencl-headers ocl-icd 2>/dev/null || true

echo "[build-opencl-native] Configuring..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  2>&1

echo "[build-opencl-native] Building..."
cmake --build . -j"$(nproc 2>/dev/null || echo 4)" 2>&1

if [ -f "$BUILD_DIR/tq_opencl_cli" ]; then
  echo "[build-opencl-native] SUCCESS: $BUILD_DIR/tq_opencl_cli"
  "$BUILD_DIR/tq_opencl_cli" probe || true
else
  echo "[build-opencl-native] FAIL: binary not produced"
  exit 1
fi
