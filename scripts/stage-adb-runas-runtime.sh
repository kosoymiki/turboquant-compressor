#!/bin/sh
set -eu

ADB_SERIAL="${1:-${TQ_ADB_SERIAL:-${ADB_SERIAL:-}}}"
RUN_AS_PKG="${TQ_ADB_RUN_AS_PACKAGE:-com.termux}"
ROOT="/data/data/com.termux/files/home/tmp_turboquant"
DRIVER_ROOT="${ROOT}/native/opencl/driver-root/layer1-compute"
CLI_BIN="/data/data/com.termux/files/home/.cache/turboquant/native-opencl-build/tq_opencl_cli"
SANDBOX_DIR="/data/user/0/${RUN_AS_PKG}/files/tq-rusticl"

need() {
  [ -f "$1" ] || { echo "missing: $1" >&2; exit 1; }
}

[ -n "$ADB_SERIAL" ] || {
  echo "missing adb serial: pass <host:port> or set TQ_ADB_SERIAL/ADB_SERIAL" >&2
  exit 1
}

need "$CLI_BIN"
need "${DRIVER_ROOT}/libRusticlOpenCL.so.1.0.0"
need "${DRIVER_ROOT}/rusticl.icd"

adb -s "$ADB_SERIAL" shell "run-as ${RUN_AS_PKG} sh -lc 'mkdir -p ${SANDBOX_DIR}/icd ${SANDBOX_DIR}/clc /data/user/0/${RUN_AS_PKG}/files && rm -f ${SANDBOX_DIR}/libRusticlOpenCL.so* ${SANDBOX_DIR}/libOpenCL.so ${SANDBOX_DIR}/rusticl.icd ${SANDBOX_DIR}/icd/rusticl.icd ${SANDBOX_DIR}/clc/*'"

cat "$CLI_BIN" | adb -s "$ADB_SERIAL" shell "run-as ${RUN_AS_PKG} sh -lc 'cat > /data/user/0/${RUN_AS_PKG}/files/tq_opencl_cli.tmp && chmod 700 /data/user/0/${RUN_AS_PKG}/files/tq_opencl_cli.tmp && mv -f /data/user/0/${RUN_AS_PKG}/files/tq_opencl_cli.tmp /data/user/0/${RUN_AS_PKG}/files/tq_opencl_cli'"
cat "${DRIVER_ROOT}/libRusticlOpenCL.so.1.0.0" | adb -s "$ADB_SERIAL" shell "run-as ${RUN_AS_PKG} sh -lc 'cat > ${SANDBOX_DIR}/libRusticlOpenCL.so.1.0.0.tmp && chmod 700 ${SANDBOX_DIR}/libRusticlOpenCL.so.1.0.0.tmp && mv -f ${SANDBOX_DIR}/libRusticlOpenCL.so.1.0.0.tmp ${SANDBOX_DIR}/libRusticlOpenCL.so.1.0.0'"
cat "/data/data/com.termux/files/usr/lib/libOpenCL.so" | adb -s "$ADB_SERIAL" shell "run-as ${RUN_AS_PKG} sh -lc 'cat > ${SANDBOX_DIR}/libOpenCL.so.tmp && chmod 700 ${SANDBOX_DIR}/libOpenCL.so.tmp && mv -f ${SANDBOX_DIR}/libOpenCL.so.tmp ${SANDBOX_DIR}/libOpenCL.so'"
cat "${DRIVER_ROOT}/rusticl.icd" | adb -s "$ADB_SERIAL" shell "run-as ${RUN_AS_PKG} sh -lc 'cat > ${SANDBOX_DIR}/rusticl.icd.tmp && mv -f ${SANDBOX_DIR}/rusticl.icd.tmp ${SANDBOX_DIR}/rusticl.icd && cat ${SANDBOX_DIR}/rusticl.icd > ${SANDBOX_DIR}/icd/rusticl.icd'"

for dep in \
  /data/data/com.termux/files/usr/lib/libc++_shared.so \
  /data/data/com.termux/files/usr/lib/libdrm.so \
  /data/data/com.termux/files/usr/lib/libz.so.1.3.2 \
  /data/data/com.termux/files/usr/lib/libzstd.so.1.5.7 \
  /data/data/com.termux/files/usr/lib/libLLVM.so \
  /data/data/com.termux/files/usr/lib/libclang-cpp.so \
  /data/data/com.termux/files/usr/lib/libffi.so \
  /data/data/com.termux/files/usr/lib/libxml2.so.16.1.3 \
  /data/data/com.termux/files/usr/lib/libiconv.so \
  /data/data/com.termux/files/usr/lib/libicuuc.so.78.3 \
  /data/data/com.termux/files/usr/lib/libicudata.so.78.3
do
  need "$dep"
  base="$(basename "$dep")"
  cat "$dep" | adb -s "$ADB_SERIAL" shell "run-as ${RUN_AS_PKG} sh -lc 'cat > ${SANDBOX_DIR}/${base}.tmp && chmod 700 ${SANDBOX_DIR}/${base}.tmp && mv -f ${SANDBOX_DIR}/${base}.tmp ${SANDBOX_DIR}/${base}'"
done

for clc in \
  /data/data/com.termux/files/usr/share/clc/spirv-mesa3d-.spv \
  /data/data/com.termux/files/usr/share/clc/spirv64-mesa3d-.spv
do
  [ -f "$clc" ] || continue
  base="$(basename "$clc")"
  cat "$clc" | adb -s "$ADB_SERIAL" shell "run-as ${RUN_AS_PKG} sh -lc 'cat > ${SANDBOX_DIR}/clc/${base}.tmp && mv -f ${SANDBOX_DIR}/clc/${base}.tmp ${SANDBOX_DIR}/clc/${base}'"
done

adb -s "$ADB_SERIAL" shell "run-as ${RUN_AS_PKG} sh -lc 'cd ${SANDBOX_DIR} && ln -sf libRusticlOpenCL.so.1.0.0 libRusticlOpenCL.so.1 && ln -sf libRusticlOpenCL.so.1 libRusticlOpenCL.so && sha256sum libRusticlOpenCL.so.1.0.0'"
