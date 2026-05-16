#!/usr/bin/env bash
# TurboQuant OpenCL probe for Termux/Android
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

echo "=== TurboQuant OpenCL Probe ==="

echo "[platform]"
uname -a 2>/dev/null || true
getprop ro.board.platform 2>/dev/null || true
getprop ro.hardware 2>/dev/null || true
getprop ro.soc.model 2>/dev/null || true
getprop ro.product.model 2>/dev/null || true

echo "[opencl-libraries]"
FOUND=0
for p in \
  /system/vendor/lib64/libOpenCL.so \
  /vendor/lib64/libOpenCL.so \
  /system/lib64/libOpenCL.so \
  /vendor/lib64/libOpenCL_adreno.so \
  /system/vendor/lib64/libOpenCL_adreno.so \
  "${PREFIX:-/data/data/com.termux/files/usr}/lib/libOpenCL.so" \
  "${PREFIX:-/data/data/com.termux/files/usr}/opt/vendor/lib/libOpenCL.so"
do
  if [ -e "$p" ]; then
    echo "FOUND $p"
    FOUND=1
  fi
done
if [ "$FOUND" -eq 0 ]; then
  echo "NONE"
fi

echo "[clinfo]"
if command -v clinfo >/dev/null 2>&1; then
  timeout 5 clinfo --list 2>/dev/null || echo "clinfo failed or timed out"
else
  echo "clinfo not installed"
fi

echo "[done]"
