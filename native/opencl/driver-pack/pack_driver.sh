#!/bin/bash
# TurboQuant Driver Pack Builder — canonical binary release layout.
# Layer 1 (compute): Rusticl/Freedreno/KGSL
# Layer 2 (backend): Turnip/KGSL
#
# Output: tq-driver-pack-adreno-a7xx-a8xx.tar.zst in native/opencl/driver-pack plus unpacked canonical native/opencl/driver-root
# Usage: ./pack_driver.sh [mesa_build_dir] [driver_root_dir]
# The unpacked canonical artifact lives at tmp_turboquant/native/opencl/driver-root by default.
#
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
TQ_STACK_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
TQ_REPO_ROOT="$(CDPATH= cd -- "${TQ_STACK_ROOT}/../.." && pwd)"
TQ_DATA_HOME="${XDG_DATA_HOME:-${HOME}/.local/share}"
TQ_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
detect_termux_prefix() {
  if [ -n "${PREFIX:-}" ] && [ -d "${PREFIX}/lib" ]; then
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
[ -n "${TERMUX_PREFIX}" ] || { echo "[FATAL] Unable to resolve Termux prefix; set PREFIX or TQ_TERMUX_PREFIX" >&2; exit 1; }
TERMUX_LIB_DIR="${TERMUX_PREFIX}/lib"
TERMUX_SHARE_DIR="${TERMUX_PREFIX}/share"

resolve_mesa_build() {
  local sdk_level="${TQ_ANDROID_SDK_LEVEL:-34}"
  if [ -n "${TQ_MESA_BUILD_DIR:-}" ] && [ -f "${TQ_MESA_BUILD_DIR}/src/gallium/targets/rusticl/libRusticlOpenCL.so.1.0.0" ]; then
    printf '%s\n' "${TQ_MESA_BUILD_DIR}"
    return
  fi
  for candidate in \
    "${1:-}" \
    "${TQ_CACHE_HOME}/turboquant/mesa-build-android${sdk_level}" \
    "${TQ_CACHE_HOME}/turboquant/mesa-build" \
    "${TQ_STACK_ROOT}/mesa-build"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}/src/gallium/targets/rusticl/libRusticlOpenCL.so.1.0.0" ]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  return 1
}

MESA_BUILD="${1:-$(resolve_mesa_build || true)}"
OUT_DIR="${2:-${TQ_DRIVER_ROOT_DIR:-${TQ_STACK_ROOT}/driver-root}}"
ARCHIVE_DIR="${TQ_DRIVER_ARCHIVE_DIR:-${TQ_STACK_ROOT}/driver-pack}"
MANIFEST="$(dirname "$0")/manifest.json"
BUILD_META_JSON="${MESA_BUILD}/tq-build-metadata.json"

PACK_NAME="${TQ_DRIVER_PACK_NAME:-tq-driver-pack-adreno-a7xx-a8xx}"
STAGING="$OUT_DIR/.staging-$$"

die() { echo "[FATAL] $*" >&2; exit 1; }
info() { echo "[tq-pack] $*"; }

append_needed_entries() {
  local so_path="$1"
  local out_path="$2"
  local rel_path="$3"
  if ! command -v readelf >/dev/null 2>&1; then
    echo "readelf unavailable" >> "$out_path"
    return 0
  fi
  {
    echo "FILE $rel_path"
    readelf -d "$so_path" | awk '/NEEDED/ {gsub(/\[|\]/, "", $5); print "NEEDED " $5}'
    echo
  } >> "$out_path"
}

validate_needed_entries() {
  local out_path="$1"
  local missing=0
  while read -r tag lib; do
    [ "$tag" = "NEEDED" ] || continue
    if [ -f "$STAGING/layer1-compute/$lib" ] || [ -f "$STAGING/layer2-vulkan/$lib" ] || [ -f "$TERMUX_LIB_DIR/$lib" ] || [ -f "/system/lib64/$lib" ] || [ -f "/vendor/lib64/$lib" ] || [ -f "/apex/com.android.runtime/lib64/$lib" ]; then
      continue
    fi
    echo "MISSING $lib" >> "$out_path"
    missing=1
  done < "$out_path"
  return "$missing"
}

# Validate Mesa build artifacts
RUSTICL_SO="$MESA_BUILD/src/gallium/targets/rusticl/libRusticlOpenCL.so.1.0.0"
TURNIP_SO="$MESA_BUILD/src/freedreno/vulkan/libvulkan_freedreno.so"

[ -n "$MESA_BUILD" ] || die "Mesa build dir not resolved"
[ -f "$RUSTICL_SO" ] || die "Rusticl not found: $RUSTICL_SO"
[ -f "$TURNIP_SO" ] || die "Turnip not found: $TURNIP_SO"
[ -f "$MANIFEST" ] || die "Manifest not found: $MANIFEST"

info "Mesa build: $MESA_BUILD"
info "Rusticl: $(ls -lh "$RUSTICL_SO" | awk '{print $5}')"
info "Turnip:  $(ls -lh "$TURNIP_SO" | awk '{print $5}')"
info "Driver root dir: $OUT_DIR"

# Create staging
rm -rf "$STAGING"
find "$OUT_DIR" -mindepth 1 -maxdepth 1 -name '.staging-*' -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$STAGING/layer1-compute" "$STAGING/layer2-vulkan" "$STAGING/meta" "$STAGING/kernels"
mkdir -p "$STAGING/layer1-compute/clc"

# --- Layer 1: Compute (Rusticl OpenCL via KGSL) ---
info "Packing Layer 1: Rusticl/KGSL compute..."
cp "$RUSTICL_SO" "$STAGING/layer1-compute/libRusticlOpenCL.so.1"
cp "$RUSTICL_SO" "$STAGING/layer1-compute/libRusticlOpenCL.so.1.0.0"
# Symlinks for ICD compatibility
ln -sf libRusticlOpenCL.so.1.0.0 "$STAGING/layer1-compute/libRusticlOpenCL.so.1"
ln -sf libRusticlOpenCL.so.1 "$STAGING/layer1-compute/libRusticlOpenCL.so"
ln -sf libRusticlOpenCL.so.1 "$STAGING/layer1-compute/libOpenCL.so.1"

# ICD file
cat > "$STAGING/layer1-compute/rusticl.icd" << 'EOF'
libRusticlOpenCL.so.1
EOF

cp "${TQ_STACK_ROOT}/kernels/"*.cl "$STAGING/kernels/"

# --- Layer 2: Vulkan Backend (Turnip for VkCompute fallback + shader compile) ---
info "Packing Layer 2: Turnip Vulkan backend..."
cp "$TURNIP_SO" "$STAGING/layer2-vulkan/libvulkan_freedreno.so"

for clc_blob in \
  "${TERMUX_SHARE_DIR}/clc/spirv-mesa3d-.spv" \
  "${TERMUX_SHARE_DIR}/clc/spirv64-mesa3d-.spv"
do
  if [ -f "$clc_blob" ]; then
    cp "$clc_blob" "$STAGING/layer1-compute/clc/"
  fi
done

# Vulkan ICD manifest
cat > "$STAGING/layer2-vulkan/freedreno_icd.aarch64.json" << 'EOF'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "./libvulkan_freedreno.so",
    "api_version": "1.4.335"
  }
}
EOF

# --- Meta layer ---
info "Packing metadata..."
cp "$MANIFEST" "$STAGING/meta/manifest.json"

# Compute SHA256
RUSTICL_SHA=$(sha256sum "$STAGING/layer1-compute/libRusticlOpenCL.so.1" | cut -d' ' -f1)
TURNIP_SHA=$(sha256sum "$STAGING/layer2-vulkan/libvulkan_freedreno.so" | cut -d' ' -f1)

python3 - <<EOF || info "WARN: python3 manifest update skipped"
import json, os
with open("$STAGING/meta/manifest.json", "r") as f:
    m = json.load(f)
build = {
  "meson_options": "-Dplatforms=android -Dplatform-sdk-version=${TQ_ANDROID_SDK_LEVEL:-34} -Dandroid-stub=true -Dandroid-libbacktrace=disabled -Dgallium-rusticl=true -Dgallium-rusticl-enable-drivers=${TQ_RUSTICL_ENABLE_DRIVERS:-freedreno} -Dvulkan-drivers=freedreno -Dgallium-drivers=freedreno -Dfreedreno-kmds=kgsl -Dllvm=${TQ_LLVM_STATE:-enabled} -Dshared-llvm=enabled -Dbuildtype=debugoptimized -Dstrip=false",
  "compiler": "llvm/clang (Termux canonical toolchain)",
  "linker": "lld",
  "static_libcxx": False,
  "strip": False,
  "sdk_level": "${TQ_ANDROID_SDK_LEVEL:-34}",
  "android_target_triple": "${TQ_ANDROID_TARGET_TRIPLE:-aarch64-linux-android}",
  "android_target": "${TQ_ANDROID_TARGET_TRIPLE:-aarch64-linux-android}${TQ_ANDROID_SDK_LEVEL:-34}"
}
if os.path.isfile("$BUILD_META_JSON"):
    with open("$BUILD_META_JSON", "r") as bf:
        build = json.load(bf)
build["install_prefix"] = "$OUT_DIR"
m["build"] = build
m["artifact_layout"] = {
  "root_contract": "unpacked archive root is the canonical TQ_DRIVER_ROOT",
  "entrypoints": {
    "env": "env.sh",
    "opencl_icd_dir": "layer1-compute",
    "vulkan_icd_manifest": "layer2-vulkan/freedreno_icd.aarch64.json",
    "kernel_dir": "kernels",
    "metadata_dir": "meta"
  },
  "files": [
    "env.sh",
    "layer1-compute/libRusticlOpenCL.so.1",
    "layer1-compute/libRusticlOpenCL.so.1.0.0",
    "layer1-compute/libRusticlOpenCL.so",
    "layer1-compute/libOpenCL.so.1",
    "layer1-compute/rusticl.icd",
    "layer1-compute/clc/spirv-mesa3d-.spv",
    "layer1-compute/clc/spirv64-mesa3d-.spv",
    "layer2-vulkan/libvulkan_freedreno.so",
    "layer2-vulkan/freedreno_icd.aarch64.json",
    "meta/manifest.json",
    "meta/dependencies.txt"
  ]
}
m['sha256'] = {
    'libRusticlOpenCL.so.1': '$RUSTICL_SHA',
    'libRusticlOpenCL.so.1.0.0': '$RUSTICL_SHA',
    'libvulkan_freedreno.so': '$TURNIP_SHA'
}
m['validated'] = True
m['pack_date'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
m.setdefault("policy", {})
m["policy"]["notes"] = [
  "This manifest describes the assembled custom driver runtime.",
  "Mesa 26.2+ already exposes freedreno-kmds=kgsl upstream; old freedreno-kgsl=true guidance is stale.",
  "Binary release contract is TQ_DRIVER_ROOT-first with native/opencl/driver-root as the canonical assembled stack.",
  "Per-device subgroup and workgroup tuning must be revalidated on real A7xx/A8xx hardware.",
  "Treat A7xx as wave128-capable when the stack explicitly exposes it; treat A8xx as dual wave64."
]
with open("$STAGING/meta/manifest.json", 'w') as f:
    json.dump(m, f, indent=2)
EOF

DEPS_FILE="$STAGING/meta/dependencies.txt"
: > "$DEPS_FILE"
append_needed_entries "$STAGING/layer1-compute/libRusticlOpenCL.so.1" "$DEPS_FILE" "layer1-compute/libRusticlOpenCL.so.1"
append_needed_entries "$STAGING/layer2-vulkan/libvulkan_freedreno.so" "$DEPS_FILE" "layer2-vulkan/libvulkan_freedreno.so"
if ! validate_needed_entries "$DEPS_FILE"; then
  die "Unresolved DT_NEEDED entries detected; see $DEPS_FILE"
fi

# Environment loader script
cat > "$STAGING/env.sh" << 'ENVEOF'
#!/bin/bash
# Source this to configure TurboQuant driver environment
# Usage: source /path/to/driver-root/env.sh

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TQ_DRIVER_ROOT="$PACK_DIR"

# Layer 1: OpenCL via Rusticl/KGSL
export OCL_ICD_VENDORS="$PACK_DIR/layer1-compute"
export LD_LIBRARY_PATH="$PACK_DIR/layer1-compute${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TQ_OPENCL_KERNEL_DIR="$PACK_DIR/kernels"
export TQ_LIBCLC_DIR="$PACK_DIR/layer1-compute/clc"
export TQ_OPENCL_CUSTOM_STACK_ONLY="${TQ_OPENCL_CUSTOM_STACK_ONLY:-1}"
export TQ_OPENCL_SILENCE_STDERR="${TQ_OPENCL_SILENCE_STDERR:-1}"
export TQ_OPENCL_VERBOSE_INIT="${TQ_OPENCL_VERBOSE_INIT:-0}"
export RUSTICL_CACHE_DIR="$HOME/.cache/rusticl"
export MESA_SHADER_CACHE_DIR="$HOME/.cache/rusticl"

# Layer 2: Vulkan backend (Turnip/kgsl)
export VK_ICD_FILENAMES="$PACK_DIR/layer2-vulkan/freedreno_icd.aarch64.json"
export VK_DRIVER_FILES="$PACK_DIR/layer2-vulkan/freedreno_icd.aarch64.json"

# Performance tuning
export MESA_NO_ERROR=1
export IR3_SHADER_DEBUG=""
export FD_MESA_DEBUG=""

# Ensure cache dir
mkdir -p "$RUSTICL_CACHE_DIR" 2>/dev/null

echo "[tq-drv] Driver root active: $PACK_DIR"
echo "[tq-drv] Layer1: Rusticl/KGSL (OpenCL 3.0)"
echo "[tq-drv] Layer2: Turnip (Vulkan 1.4.335)"
ENVEOF
chmod +x "$STAGING/env.sh"

mkdir -p "$OUT_DIR"
mkdir -p "$ARCHIVE_DIR"
ARCHIVE="$ARCHIVE_DIR/${PACK_NAME}.tar.zst"

info "Creating archive: $ARCHIVE"
rm -f "$ARCHIVE"
tar -C "$STAGING" -cf - . | zstd -T2 -9 -o "$ARCHIVE"

info "Refreshing unpacked canonical driver-root at: $OUT_DIR"
find "$OUT_DIR" -mindepth 1 -maxdepth 1 ! -name "${PACK_NAME}.tar.zst" -exec rm -rf {} +
tar --zstd -xf "$ARCHIVE" -C "$OUT_DIR"

# Cleanup staging
rm -rf "$STAGING"

ARCHIVE_SIZE=$(ls -lh "$ARCHIVE" | awk '{print $5}')
info "Done: $ARCHIVE ($ARCHIVE_SIZE)"
info "SHA256: $(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
info ""
info "Canonical root:"
info "  source \"$OUT_DIR/env.sh\""
info "  # TurboQuant now prefers native/opencl/driver-root as primary contract"
