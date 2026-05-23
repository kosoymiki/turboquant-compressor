#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST_ROOT="${1:-/data/local/tmp/tq-rusticl}"
CLI_SRC="${ROOT_DIR}/../.cache/turboquant/native-opencl-build/tq_opencl_cli"
DRIVER_ROOT="${ROOT_DIR}/native/opencl/driver-root"
USR_LIB="/data/data/com.termux/files/usr/lib"
USR_SHARE="/data/data/com.termux/files/usr/share"

log() {
  printf '[stage-adb-rusticl-runtime] %s\n' "$*"
}

push_file() {
  local src="$1"
  local dst="$2"
  adb push "$src" "$dst" >/dev/null
}

shell_cmd() {
  adb shell "$@"
}

require_file() {
  local path="$1"
  if [ ! -e "$path" ]; then
    printf 'missing required file: %s\n' "$path" >&2
    exit 1
  fi
}

require_file "$CLI_SRC"
require_file "${DRIVER_ROOT}/layer1-compute/libRusticlOpenCL.so.1.0.0"
require_file "${DRIVER_ROOT}/layer1-compute/rusticl.icd"

for dep in \
  libOpenCL.so \
  libc++_shared.so \
  libdrm.so \
  libz.so.1.3.2 \
  libzstd.so.1.5.7 \
  libLLVM.so \
  libclang-cpp.so \
  libffi.so \
  libxml2.so.16.1.3 \
  libiconv.so \
  libicuuc.so.78.3 \
  libicudata.so.78.3
do
  require_file "${USR_LIB}/${dep}"
done

for dep in \
  spirv-mesa3d-.spv \
  spirv64-mesa3d-.spv
do
  require_file "${USR_SHARE}/clc/${dep}"
done

log "preparing ${DEST_ROOT}"
shell_cmd rm -rf "$DEST_ROOT"
shell_cmd mkdir -p "$DEST_ROOT/icd" "$DEST_ROOT/clc"

log "pushing tq_opencl_cli"
push_file "$CLI_SRC" /data/local/tmp/tq_opencl_cli

log "pushing Rusticl loader + ICD"
push_file "${USR_LIB}/libOpenCL.so" "${DEST_ROOT}/libOpenCL.so"
push_file "${DRIVER_ROOT}/layer1-compute/libRusticlOpenCL.so.1.0.0" "${DEST_ROOT}/libRusticlOpenCL.so.1.0.0"
push_file "${DRIVER_ROOT}/layer1-compute/rusticl.icd" "${DEST_ROOT}/icd/rusticl.icd"

log "pushing staged dependencies"
for dep in \
  libc++_shared.so \
  libdrm.so \
  libz.so.1.3.2 \
  libzstd.so.1.5.7 \
  libLLVM.so \
  libclang-cpp.so \
  libffi.so \
  libxml2.so.16.1.3 \
  libiconv.so \
  libicuuc.so.78.3 \
  libicudata.so.78.3
do
  push_file "${USR_LIB}/${dep}" "${DEST_ROOT}/${dep}"
done

log "pushing libclc blobs"
for dep in \
  spirv-mesa3d-.spv \
  spirv64-mesa3d-.spv
do
  push_file "${USR_SHARE}/clc/${dep}" "${DEST_ROOT}/clc/${dep}"
done

log "creating symlink surface"
shell_cmd ln -sf libRusticlOpenCL.so.1.0.0 "${DEST_ROOT}/libRusticlOpenCL.so.1"
shell_cmd ln -sf libRusticlOpenCL.so.1 "${DEST_ROOT}/libRusticlOpenCL.so"
shell_cmd ln -sf libz.so.1.3.2 "${DEST_ROOT}/libz.so.1"
shell_cmd ln -sf libzstd.so.1.5.7 "${DEST_ROOT}/libzstd.so.1"
shell_cmd ln -sf libxml2.so.16.1.3 "${DEST_ROOT}/libxml2.so.16"
shell_cmd ln -sf libicuuc.so.78.3 "${DEST_ROOT}/libicuuc.so.78"
shell_cmd ln -sf libicudata.so.78.3 "${DEST_ROOT}/libicudata.so.78"

log "staged custom runtime to ${DEST_ROOT}"
log "launch probe:"
printf "adb shell 'LD_LIBRARY_PATH=%s OCL_ICD_VENDORS=%s/icd /data/local/tmp/tq_opencl_cli probe'\n" "$DEST_ROOT" "$DEST_ROOT"
log "launch frontier-smoke:"
printf "adb shell 'LD_LIBRARY_PATH=%s OCL_ICD_VENDORS=%s/icd TQ_OPENCL_TRACE=1 /data/local/tmp/tq_opencl_cli frontier-smoke'\n" "$DEST_ROOT" "$DEST_ROOT"
