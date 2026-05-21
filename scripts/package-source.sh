#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="$(basename "$ROOT")"
VERSION="$(node -p "require('$ROOT/package.json').version")"
OUT="$ROOT/../${NAME}-${VERSION}-source.tar.gz"

cd "$ROOT/.."

tar \
  --exclude="${NAME}/node_modules" \
  --exclude="${NAME}/dist" \
  --exclude="${NAME}/coverage" \
  --exclude="${NAME}/.git" \
  --exclude="${NAME}/.turbo" \
  --exclude="${NAME}/.cache" \
  --exclude="${NAME}/*.tgz" \
  --exclude="${NAME}/*.tar.gz" \
  --exclude="${NAME}/**/.claude/settings.local.json" \
  --exclude="${NAME}/**/.claude/*.local.json" \
  --exclude="${NAME}/forensics" \
  --exclude="${NAME}/*transcript*.jsonl" \
  --exclude="${NAME}/*transcript*.json" \
  --exclude="${NAME}/**/*transcript*.jsonl" \
  --exclude="${NAME}/**/*transcript*.json" \
  --exclude="${NAME}/native/opencl/build" \
  --exclude="${NAME}/native/opencl/build-tq-zero" \
  --exclude="${NAME}/native/opencl/runtime-pack" \
  --exclude="${NAME}/native/opencl/runtime-pack-fresh" \
  --exclude="${NAME}/native/opencl/driver-pack/out" \
  --exclude="${NAME}/native/opencl/driver-pack/out-fresh" \
  --exclude="${NAME}/native/opencl/driver-pack/patches" \
  --exclude="${NAME}/native/vulkan/build" \
  --exclude="${NAME}/driver" \
  --exclude="${NAME}/third_party/mesa-adreno" \
  --exclude="${NAME}/**/CMakeCache.txt" \
  --exclude="${NAME}/**/CMakeFiles" \
  --exclude="${NAME}/**/cmake_install.cmake" \
  -czf "$OUT" "$NAME"

if tar -tzf "$OUT" | grep -E '(^|/)(node_modules|dist|coverage|\.git)(/|$)'; then
  echo "ERROR: forbidden path found in archive"
  exit 1
fi

if tar -tzf "$OUT" | grep -E '\.claude/settings\.local\.json|\.claude/.*\.local\.json'; then
  echo "ERROR: local Claude settings found in archive"
  exit 1
fi

if tar -tzf "$OUT" | grep -E '(CMakeCache\.txt|CMakeFiles/|cmake_install\.cmake|/build/tq_opencl_cli|/build/test_)'; then
  echo "ERROR: CMake build artifacts found in source archive"
  exit 1
fi

ls -lh "$OUT"
echo "$OUT"
