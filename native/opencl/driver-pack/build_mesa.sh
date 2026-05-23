#!/bin/bash
# TurboQuant Mesa Two-Layer Build Script
# Builds the custom Rusticl/Freedreno/KGSL + Turnip/KGSL release stack.
#
# This script is source-of-truth for recovery on Mesa 26.2+:
# - no hardcoded single tree name
# - uses upstream-native Freedreno KGSL option
# - prefers custom stack over vendor OpenCL assumptions
#
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

die() { echo "[FATAL] $*" >&2; exit 1; }
info() { echo "[tq-build] $*"; }
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
TQ_STACK_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
TQ_REPO_ROOT="$(CDPATH= cd -- "${TQ_STACK_ROOT}/../.." && pwd)"
TQ_DATA_HOME="${XDG_DATA_HOME:-${HOME}/.local/share}"
TQ_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
detect_termux_prefix() {
    if [ -n "${PREFIX:-}" ] && [ -x "${PREFIX}/bin/clang" ]; then
        printf '%s\n' "${PREFIX}"
        return
    fi
    local clang_bin
    clang_bin="$(command -v clang 2>/dev/null || true)"
    if [ -n "${clang_bin}" ]; then
        printf '%s\n' "$(CDPATH= cd -- "$(dirname "${clang_bin}")/.." && pwd)"
        return
    fi
    return 1
}
TERMUX_PREFIX="${TQ_TERMUX_PREFIX:-$(detect_termux_prefix || true)}"
[ -n "${TERMUX_PREFIX}" ] || die "Unable to resolve Termux/LLVM prefix; set PREFIX or TQ_TERMUX_PREFIX"
TERMUX_BIN="${TERMUX_PREFIX}/bin"
TERMUX_CLANG="${TERMUX_BIN}/clang"
TERMUX_CLANGXX="${TERMUX_BIN}/clang++"
TERMUX_LLVM_CONFIG="${TERMUX_BIN}/llvm-config"
TERMUX_LLVM_AR="${TERMUX_BIN}/llvm-ar"
TERMUX_RANLIB="${TERMUX_BIN}/llvm-ranlib"
TERMUX_LLVM_SPIRV="${TERMUX_BIN}/llvm-spirv"
TERMUX_SPIRV_LINK="${TERMUX_BIN}/spirv-link"
TERMUX_SPIRV_VAL="${TERMUX_BIN}/spirv-val"

for tool in "$TERMUX_CLANG" "$TERMUX_CLANGXX" "$TERMUX_LLVM_CONFIG" "$TERMUX_LLVM_AR" "$TERMUX_RANLIB" "$TERMUX_LLVM_SPIRV" "$TERMUX_SPIRV_LINK" "$TERMUX_SPIRV_VAL"; do
    [ -x "$tool" ] || die "Required Termux tool not found: $tool"
done

unset CC CXX CPP LD AR AS NM STRIP OBJCOPY OBJDUMP RANLIB CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
export PATH="${TERMUX_BIN}:$PATH"
export AR="${TERMUX_LLVM_AR}"
export RANLIB="${TERMUX_RANLIB}"
export LLVM_CONFIG="${TERMUX_LLVM_CONFIG}"
export LLVM_SPIRV="${TERMUX_LLVM_SPIRV}"
export SPIRV_LINK="${TERMUX_SPIRV_LINK}"
export SPIRV_VAL="${TERMUX_SPIRV_VAL}"
export LIBCLC_PATH="${LIBCLC_PATH:-${TERMUX_PREFIX}/share/clc}"
STATIC_LIBCLC="${TQ_STATIC_LIBCLC:-all}"
DEVICE_LIBCLC_PATH="${TQ_DEVICE_LIBCLC_PATH:-/data/local/tmp/tq-rusticl/clc}"

resolve_upstream_mesa_base() {
    if [ -n "${TQ_MESA_UPSTREAM_BASE:-}" ] && [ -d "${TQ_MESA_UPSTREAM_BASE}/src/freedreno" ]; then
        printf '%s\n' "${TQ_MESA_UPSTREAM_BASE}"
        return
    fi
    for candidate in \
        "${HOME}/mesa-upstream-26.2-devel" \
        "${HOME}/mesa"; do
        if [ -d "${candidate}/src/freedreno" ]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done
    return 1
}

MESA_SRC="${1:-$(resolve_upstream_mesa_base || true)}"
[ -n "${MESA_SRC}" ] || die "Mesa source not found; pass path explicitly or set TQ_MESA_UPSTREAM_BASE"
[ -d "${MESA_SRC}/src/freedreno" ] || die "Mesa source not found at ${MESA_SRC}"

BUILD_DIR="${TQ_MESA_BUILD_DIR:-${TQ_CACHE_HOME}/turboquant/mesa-build-android${TQ_ANDROID_SDK_LEVEL:-34}}"
INSTALL_PREFIX="${TQ_DRIVER_INSTALL_PREFIX:-${TQ_STACK_ROOT}/driver-root}"
JOBS="${TQ_BUILD_JOBS:-4}"
SDK_LEVEL="${TQ_ANDROID_SDK_LEVEL:-34}"
LLVM_STATE="${TQ_LLVM_STATE:-enabled}"
RUSTICL_ENABLE_DRIVERS="${TQ_RUSTICL_ENABLE_DRIVERS:-freedreno}"
BUILD_META_JSON="${BUILD_DIR}/tq-build-metadata.json"
MESON_NATIVE_FILE="${BUILD_DIR}/tq-termux-android-sdk${SDK_LEVEL}.ini"
ANDROID_TARGET_TRIPLE="${TQ_ANDROID_TARGET_TRIPLE:-aarch64-linux-android}"
ANDROID_TARGET="${ANDROID_TARGET_TRIPLE}${SDK_LEVEL}"
export CC="${TERMUX_CLANG}"
export CXX="${TERMUX_CLANGXX}"
export CFLAGS="${CFLAGS:-} --target=${ANDROID_TARGET}"
export CXXFLAGS="${CXXFLAGS:-} --target=${ANDROID_TARGET}"
export LDFLAGS="${LDFLAGS:-} --target=${ANDROID_TARGET}"
export RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=--target=${ANDROID_TARGET}"
export BINDGEN_EXTRA_CLANG_ARGS="--target=${ANDROID_TARGET} -D_LIBCPP_HAS_COND_CLOCKWAIT=0"
export BINDGEN_EXTRA_CLANG_ARGS_CXX="${BINDGEN_EXTRA_CLANG_ARGS}"

mkdir -p "${BUILD_DIR}"
cat > "${MESON_NATIVE_FILE}" <<EOF
[binaries]
c = '${TERMUX_CLANG}'
cpp = '${TERMUX_CLANGXX}'
ar = '${TERMUX_LLVM_AR}'
ranlib = '${TERMUX_RANLIB}'
llvm-config = '${TERMUX_LLVM_CONFIG}'

[built-in options]
c_args = ['--target=${ANDROID_TARGET}', '-O2', '-march=armv8.2-a', '-mno-outline-atomics']
cpp_args = ['--target=${ANDROID_TARGET}', '-O2', '-march=armv8.2-a', '-mno-outline-atomics']
c_link_args = ['--target=${ANDROID_TARGET}']
cpp_link_args = ['--target=${ANDROID_TARGET}']
EOF

configure_args=(
    -Dplatforms=android
    -Dplatform-sdk-version="${SDK_LEVEL}"
    -Dandroid-stub=true
    -Dandroid-libbacktrace=disabled
    -Dgallium-rusticl=true
    -Dgallium-rusticl-enable-drivers="${RUSTICL_ENABLE_DRIVERS}"
    -Dvulkan-drivers=freedreno
    -Dgallium-drivers=freedreno
    -Dfreedreno-kmds=kgsl
    -Dllvm="${LLVM_STATE}"
    -Dshared-llvm=enabled
    -Dstatic-libclc="${STATIC_LIBCLC}"
    -Dbuildtype=debugoptimized
    -Dstrip=false
    -Dglx=disabled
    -Degl=disabled
    -Dgbm=disabled
    -Dgles1=disabled
    -Dgles2=disabled
    -Dopengl=false
    --prefix="${INSTALL_PREFIX}"
)

if [ ! -f "${BUILD_DIR}/build.ninja" ]; then
    info "Configuring Mesa at ${MESA_SRC}"
    meson setup --native-file "${MESON_NATIVE_FILE}" "${BUILD_DIR}" "${MESA_SRC}" "${configure_args[@]}" || die "Meson configure failed"
else
    info "Reconfiguring existing build dir ${BUILD_DIR}"
    meson setup --reconfigure --native-file "${MESON_NATIVE_FILE}" "${BUILD_DIR}" "${MESA_SRC}" "${configure_args[@]}" || die "Meson reconfigure failed"
fi

info "Building with ninja -j${JOBS} -k0"
ninja -C "${BUILD_DIR}" -j"${JOBS}" -k0 \
    src/gallium/targets/rusticl/libRusticlOpenCL.so.1.0.0 \
    src/freedreno/vulkan/libvulkan_freedreno.so

RUSTICL="${BUILD_DIR}/src/gallium/targets/rusticl/libRusticlOpenCL.so.1.0.0"
TURNIP="${BUILD_DIR}/src/freedreno/vulkan/libvulkan_freedreno.so"

if [ -f "${RUSTICL}" ] && [ -f "${TURNIP}" ]; then
python3 - <<EOF
import json
data = {
  "meson_options": "-Dplatforms=android -Dplatform-sdk-version=${SDK_LEVEL} -Dandroid-stub=true -Dandroid-libbacktrace=disabled -Dgallium-rusticl=true -Dgallium-rusticl-enable-drivers=${RUSTICL_ENABLE_DRIVERS} -Dvulkan-drivers=freedreno -Dgallium-drivers=freedreno -Dfreedreno-kmds=kgsl -Dllvm=${LLVM_STATE} -Dshared-llvm=enabled -Dstatic-libclc=${STATIC_LIBCLC} -Dbuildtype=debugoptimized -Dstrip=false",
  "compiler": "llvm/clang (Termux canonical toolchain)",
  "linker": "lld",
  "static_libcxx": False,
  "strip": False,
  "mesa_source": "${MESA_SRC}",
  "build_dir": "${BUILD_DIR}",
  "install_prefix": "${INSTALL_PREFIX}",
  "sdk_level": "${SDK_LEVEL}",
  "android_target": "${ANDROID_TARGET}",
  "android_target_triple": "${ANDROID_TARGET_TRIPLE}",
  "llvm_state": "${LLVM_STATE}",
  "rusticl_enable_drivers": "${RUSTICL_ENABLE_DRIVERS}",
  "static_libclc": "${STATIC_LIBCLC}",
  "toolchain_contract": {
    "clang": "${TERMUX_CLANG}",
    "clangxx": "${TERMUX_CLANGXX}",
    "llvm_config": "${TERMUX_LLVM_CONFIG}",
    "llvm_spirv": "${TERMUX_LLVM_SPIRV}",
    "spirv_link": "${TERMUX_SPIRV_LINK}",
    "spirv_val": "${TERMUX_SPIRV_VAL}",
    "libclc_path": "${LIBCLC_PATH}",
    "device_libclc_path": "${DEVICE_LIBCLC_PATH}"
  },
  "rusticl_runtime_env": {
    "RUSTICL_ENABLE": "${RUSTICL_ENABLE_DRIVERS}",
    "MESA_SHADER_CACHE_DIR": "${TQ_CACHE_HOME}/turboquant-rusticl"
  }
}
with open("${BUILD_META_JSON}", "w") as f:
    json.dump(data, f, indent=2)
EOF
  info "BUILD SUCCESS"
  info "  Mesa source: ${MESA_SRC}"
  info "  Build dir:    ${BUILD_DIR}"
  info "  Build meta:   ${BUILD_META_JSON}"
  info "  Layer1:       ${RUSTICL} ($(ls -lh "${RUSTICL}" | awk '{print $5}'))"
  info "  Layer2:       ${TURNIP} ($(ls -lh "${TURNIP}" | awk '{print $5}'))"
  info "Next: ./pack_driver.sh ${BUILD_DIR} ${INSTALL_PREFIX}"
else
  info "BUILD INCOMPLETE"
  [ ! -f "${RUSTICL}" ] && info "  MISSING: ${RUSTICL}"
  [ ! -f "${TURNIP}" ] && info "  MISSING: ${TURNIP}"
fi
