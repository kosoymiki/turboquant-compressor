#!/bin/bash
# TurboQuant Mesa Two-Layer Build Script
# Builds Rusticl (Layer 1) + Turnip (Layer 2) from mesa-upstream source
# Uses -j4 -k0 (caveman mode), safe for Termux
#
# SPDX-License-Identifier: GPL-3.0-or-later

set -uo pipefail

MESA_SRC="$HOME/mesa-upstream"
BUILD_DIR="$MESA_SRC/build"

die() { echo "[FATAL] $*" >&2; exit 1; }
info() { echo "[tq-build] $*"; }

[ -d "$MESA_SRC/src/freedreno" ] || die "Mesa source not found at $MESA_SRC"

# Meson configure (skip if already configured)
if [ ! -f "$BUILD_DIR/build.ninja" ]; then
    info "Configuring Mesa (Rusticl + Turnip, kgsl=true)..."
    meson setup "$BUILD_DIR" "$MESA_SRC" \
        -Dgallium-rusticl=true \
        -Dvulkan-drivers=freedreno \
        -Dgallium-drivers=freedreno \
        -Dfreedreno-kgsl=true \
        -Dllvm=enabled \
        -Dshared-llvm=disabled \
        -Dbuildtype=release \
        -Dstrip=true \
        -Dcpp_args="-O2 -march=armv8.2-a" \
        -Dc_args="-O2 -march=armv8.2-a" \
        -Dplatforms="" \
        -Dglx=disabled \
        -Degl=disabled \
        -Dgbm=disabled \
        -Dgles1=disabled \
        -Dgles2=disabled \
        -Dopengl=false \
        --prefix="$HOME/turboquant/drivers" \
        || die "Meson configure failed"
else
    info "Build dir exists, reconfiguring skipped. Use 'meson setup --reconfigure' if needed."
fi

# Build — -j4 -k0 (don't crash Termux, don't stop on errors)
info "Building... (-j4 -k0)"
ninja -C "$BUILD_DIR" -j4 -k0 \
    src/gallium/targets/rusticl/libRusticlOpenCL.so.1.0.0 \
    src/freedreno/vulkan/libvulkan_freedreno.so \
    2>&1 | tail -20

RUSTICL="$BUILD_DIR/src/gallium/targets/rusticl/libRusticlOpenCL.so.1.0.0"
TURNIP="$BUILD_DIR/src/freedreno/vulkan/libvulkan_freedreno.so"

if [ -f "$RUSTICL" ] && [ -f "$TURNIP" ]; then
    info "BUILD SUCCESS"
    info "  Layer1: $RUSTICL ($(ls -lh "$RUSTICL" | awk '{print $5}'))"
    info "  Layer2: $TURNIP ($(ls -lh "$TURNIP" | awk '{print $5}'))"
    info ""
    info "Next: ./pack_driver.sh $BUILD_DIR ~/turboquant/driver-pack-out"
else
    info "BUILD INCOMPLETE — check errors above"
    [ ! -f "$RUSTICL" ] && info "  MISSING: $RUSTICL"
    [ ! -f "$TURNIP" ] && info "  MISSING: $TURNIP"
fi
