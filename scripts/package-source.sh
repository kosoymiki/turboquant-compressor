#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR_NAME="$(basename "$ROOT")"
PACKAGE_NAME="$(node -p "require('$ROOT/package.json').name")"
VERSION="$(node -p "require('$ROOT/package.json').version")"
OUT="$ROOT/../${PACKAGE_NAME}-${VERSION}-source.tar.gz"

cd "$ROOT/.."

tar \
  --exclude="${SRC_DIR_NAME}/node_modules" \
  --exclude="${SRC_DIR_NAME}/dist" \
  --exclude="${SRC_DIR_NAME}/coverage" \
  --exclude="${SRC_DIR_NAME}/.git" \
  --exclude="${SRC_DIR_NAME}/**/.git" \
  --exclude="${SRC_DIR_NAME}/**/.git/**" \
  --exclude="${SRC_DIR_NAME}/.turbo" \
  --exclude="${SRC_DIR_NAME}/.cache" \
  --exclude="${SRC_DIR_NAME}/artifacts" \
  --exclude="${SRC_DIR_NAME}/bench/results" \
  --exclude="${SRC_DIR_NAME}/*.tgz" \
  --exclude="${SRC_DIR_NAME}/*.tar.gz" \
  --exclude="${SRC_DIR_NAME}/**/.claude/settings.local.json" \
  --exclude="${SRC_DIR_NAME}/**/.claude/*.local.json" \
  --exclude="${SRC_DIR_NAME}/docs/RELEASE_SUMMARY.md" \
  --exclude="${SRC_DIR_NAME}/docs/AGENT_DONOR_CORPUS_*.md" \
  --exclude="${SRC_DIR_NAME}/docs/FRONTIER_DONOR_MODEL_*.md" \
  --exclude="${SRC_DIR_NAME}/docs/MESA_STACK_FORENSICS_MODULE_*.md" \
  --exclude="${SRC_DIR_NAME}/docs/MESA_TQ_FORENSIC_AUDIT_*.md" \
  --exclude="${SRC_DIR_NAME}/docs/P1_*.md" \
  --exclude="${SRC_DIR_NAME}/docs/P2_*.md" \
  --exclude="${SRC_DIR_NAME}/docs/RENDERDOC_GPU_DEBUG_CORPUS_*.md" \
  --exclude="${SRC_DIR_NAME}/docs/REPO_HYGIENE_BASELINE.md" \
  --exclude="${SRC_DIR_NAME}/docs/REPO_TECHNICAL_AUDIT_*.md" \
  --exclude="${SRC_DIR_NAME}/docs/RUNTIME_EVIDENCE_CHAIN.md" \
  --exclude="${SRC_DIR_NAME}/docs/SYSTEM_SVM_DONOR_CORPUS_*.md" \
  --exclude="${SRC_DIR_NAME}/docs/frontier-tracing-contract-*.md" \
  --exclude="${SRC_DIR_NAME}/docs/mesa-rusticl-vm-audit-*.md" \
  --exclude="${SRC_DIR_NAME}/docs/mesa-single-process-build-forensics-*.md" \
  --exclude="${SRC_DIR_NAME}/docs/opencl-donor-file-audit-*.md" \
  --exclude="${SRC_DIR_NAME}/docs/opencl-donor-integration-*.md" \
  --exclude="${SRC_DIR_NAME}/docs/runtime-pack-drift-forensic-*.md" \
  --exclude="${SRC_DIR_NAME}/docs/tz-frontier-capability-matrix-*.md" \
  --exclude="${SRC_DIR_NAME}/docs/DRIVER_DEBUG_FORENSICS_*" \
  --exclude="${SRC_DIR_NAME}/docs/EXPORT_CHECKLIST.md" \
  --exclude="${SRC_DIR_NAME}/docs/EXPORT_VERDICT.md" \
  --exclude="${SRC_DIR_NAME}/forensics" \
  --exclude="${SRC_DIR_NAME}/*transcript*.jsonl" \
  --exclude="${SRC_DIR_NAME}/*transcript*.json" \
  --exclude="${SRC_DIR_NAME}/**/*transcript*.jsonl" \
  --exclude="${SRC_DIR_NAME}/**/*transcript*.json" \
  --exclude="${SRC_DIR_NAME}/native/opencl/build" \
  --exclude="${SRC_DIR_NAME}/native/opencl/build-tq-zero" \
  --exclude="${SRC_DIR_NAME}/native/opencl/runtime-pack" \
  --exclude="${SRC_DIR_NAME}/native/opencl/runtime-pack-fresh" \
  --exclude="${SRC_DIR_NAME}/native/opencl/driver-pack/out" \
  --exclude="${SRC_DIR_NAME}/native/opencl/driver-pack/out-fresh" \
  --exclude="${SRC_DIR_NAME}/native/opencl/driver-pack/patches" \
  --exclude="${SRC_DIR_NAME}/native/opencl/driver-root/**/*.bak*" \
  --exclude="${SRC_DIR_NAME}/native/vulkan/build" \
  --exclude="${SRC_DIR_NAME}/driver" \
  --exclude="${SRC_DIR_NAME}/third_party/mesa-adreno" \
  --exclude="${SRC_DIR_NAME}/third_party/OpenCL-Intercept-Layer/.git" \
  --exclude="${SRC_DIR_NAME}/third_party/OpenCL-Intercept-Layer/.git/**" \
  --exclude="${SRC_DIR_NAME}/**/CMakeCache.txt" \
  --exclude="${SRC_DIR_NAME}/**/CMakeFiles" \
  --exclude="${SRC_DIR_NAME}/**/cmake_install.cmake" \
  --transform "s,^${SRC_DIR_NAME},${PACKAGE_NAME}," \
  -czf "$OUT" "$SRC_DIR_NAME"

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

if tar -tzf "$OUT" | grep -E '(^|/)(artifacts|forensics|bench/results)(/|$)|docs/(AGENT_DONOR_CORPUS_|FRONTIER_DONOR_MODEL_|MESA_STACK_FORENSICS_MODULE_|MESA_TQ_FORENSIC_AUDIT_|P1_|P2_|RENDERDOC_GPU_DEBUG_CORPUS_|REPO_TECHNICAL_AUDIT_|SYSTEM_SVM_DONOR_CORPUS_|frontier-tracing-contract-|mesa-rusticl-vm-audit-|mesa-single-process-build-forensics-|opencl-donor-file-audit-|opencl-donor-integration-|runtime-pack-drift-forensic-|tz-frontier-capability-matrix-|DRIVER_DEBUG_FORENSICS_|EXPORT_CHECKLIST\.md|EXPORT_VERDICT\.md)|driver-root/.+\.bak'; then
  echo "ERROR: internal evidence/debug content found in source archive"
  exit 1
fi

ls -lh "$OUT"
echo "$OUT"
