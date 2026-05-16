#!/usr/bin/env bash
# ADB-based Adreno/OpenCL diagnostics — collects device info without root.
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

OUT="${1:-forensics/adreno}"
mkdir -p "$OUT"

echo "[adb-adreno-diagnose] Collecting device info..."

adb devices -l > "$OUT/adb-devices.txt" 2>/dev/null || echo "adb not connected" > "$OUT/adb-devices.txt"

adb shell getprop ro.board.platform > "$OUT/ro.board.platform.txt" 2>/dev/null || true
adb shell getprop ro.hardware > "$OUT/ro.hardware.txt" 2>/dev/null || true
adb shell getprop ro.soc.model > "$OUT/ro.soc.model.txt" 2>/dev/null || true
adb shell getprop ro.product.model > "$OUT/ro.product.model.txt" 2>/dev/null || true
adb shell getprop ro.build.version.release > "$OUT/android-release.txt" 2>/dev/null || true

adb shell 'ls -l /vendor/lib64/libOpenCL.so /system/vendor/lib64/libOpenCL.so /vendor/lib64/libOpenCL_adreno.so 2>/dev/null || echo NONE' \
  > "$OUT/opencl-libs.txt" 2>/dev/null || true

adb shell 'ls -l /dev/kgsl-3d0 /dev/dri/renderD128 2>/dev/null || echo NONE' \
  > "$OUT/gpu-devnodes.txt" 2>/dev/null || true

adb shell 'cat /proc/meminfo | head -5' > "$OUT/meminfo.txt" 2>/dev/null || true

echo "[adb-adreno-diagnose] Done. Output: $OUT/"
ls "$OUT/"
