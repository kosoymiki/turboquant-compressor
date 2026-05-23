#!/data/data/com.termux/files/usr/bin/sh
set -eu

ROOT="/data/data/com.termux/files/home/tmp_turboquant"
SRC_DIR="$ROOT/third_party/OpenCL-Intercept-Layer"
BUILD_DIR="$ROOT/.cache/opencl-intercept-build"
INSTALL_DIR="$ROOT/.cache/opencl-intercept-install"
OUT_LIB="$SRC_DIR/libOpenCLIntercept.so"
OUT_OPENCL="$SRC_DIR/libOpenCL.so"
OUT_LOADER="$SRC_DIR/cliloader"
LOG_DIR="$ROOT/forensics/adreno"
CONFIGURE_LOG="$LOG_DIR/opencl-intercept-configure-j6-k0.log"
BUILD_LOG="$LOG_DIR/opencl-intercept-build-j6-k0.log"
INSTALL_LOG="$LOG_DIR/opencl-intercept-install-j6-k0.log"
MANIFEST_JSON="$LOG_DIR/opencl-intercept-build.json"

mkdir -p "$LOG_DIR"
rm -rf "$BUILD_DIR" "$INSTALL_DIR"

cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DENABLE_MDAPI=FALSE \
  -DENABLE_CLIPROF=FALSE \
  -DENABLE_ITT=FALSE \
  -DENABLE_KERNEL_OVERRIDES=FALSE \
  -DENABLE_SCRIPTS=FALSE \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  >"$CONFIGURE_LOG" 2>&1

ninja -C "$BUILD_DIR" -j6 -k0 >"$BUILD_LOG" 2>&1
ninja -C "$BUILD_DIR" install >"$INSTALL_LOG" 2>&1

cp -f "$INSTALL_DIR/lib/libOpenCL.so" "$OUT_OPENCL"
cp -f "$INSTALL_DIR/lib/libOpenCL.so" "$OUT_LIB"
if [ -f "$INSTALL_DIR/bin/cliloader" ]; then
  cp -f "$INSTALL_DIR/bin/cliloader" "$OUT_LOADER"
fi

HOST_SHA="$(sha256sum "$OUT_LIB" | awk '{print $1}')"
cat >"$MANIFEST_JSON" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "type": "opencl_intercept_build",
  "route": "mesa_rusticl_kgsl",
  "sourceDir": "$SRC_DIR",
  "buildDir": "$BUILD_DIR",
  "installDir": "$INSTALL_DIR",
  "outputLibrary": "$OUT_LIB",
  "outputOpenCL": "$OUT_OPENCL",
  "outputLoader": "$( [ -f "$OUT_LOADER" ] && printf '%s' "$OUT_LOADER" || printf 'null' )",
  "sha256": "$HOST_SHA",
  "upstreamHead": "$(cd "$SRC_DIR" && git rev-parse HEAD)",
  "buildOptions": {
    "ENABLE_MDAPI": false,
    "ENABLE_CLIPROF": false,
    "ENABLE_ITT": false,
    "ENABLE_KERNEL_OVERRIDES": false,
    "ENABLE_SCRIPTS": false
  },
  "logs": {
    "configure": "$CONFIGURE_LOG",
    "build": "$BUILD_LOG",
    "install": "$INSTALL_LOG"
  }
}
EOF

printf '%s\n' "$MANIFEST_JSON"
