#!/bin/bash
# TurboQuant Driver Pack Builder — Two-Layer Archive
# Layer 1 (compute): libRusticlOpenCL.so.1 + freedreno/kgsl pipe loader
# Layer 2 (backend): libvulkan_freedreno.so (Turnip) for VkCompute fallback
#
# Output: tq-driver-pack-adreno730.tar.zst (flat two-layer archive)
# Usage: ./pack_driver.sh [mesa_build_dir] [output_dir]
#
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

MESA_BUILD="${1:-$HOME/mesa-upstream/build}"
OUT_DIR="${2:-$HOME/turboquant/driver-pack-out}"
MANIFEST="$(dirname "$0")/manifest.json"

PACK_NAME="tq-driver-pack-adreno730"
STAGING="$OUT_DIR/.staging-$$"

die() { echo "[FATAL] $*" >&2; exit 1; }
info() { echo "[tq-pack] $*"; }

# Validate Mesa build artifacts
RUSTICL_SO="$MESA_BUILD/src/gallium/targets/rusticl/libRusticlOpenCL.so.1.0.0"
TURNIP_SO="$MESA_BUILD/src/freedreno/vulkan/libvulkan_freedreno.so"

[ -f "$RUSTICL_SO" ] || die "Rusticl not found: $RUSTICL_SO"
[ -f "$TURNIP_SO" ] || die "Turnip not found: $TURNIP_SO"
[ -f "$MANIFEST" ] || die "Manifest not found: $MANIFEST"

info "Mesa build: $MESA_BUILD"
info "Rusticl: $(ls -lh "$RUSTICL_SO" | awk '{print $5}')"
info "Turnip:  $(ls -lh "$TURNIP_SO" | awk '{print $5}')"

# Create staging
rm -rf "$STAGING"
mkdir -p "$STAGING/layer1-compute" "$STAGING/layer2-vulkan" "$STAGING/meta"

# --- Layer 1: Compute (Rusticl OpenCL via KGSL) ---
info "Packing Layer 1: Rusticl/KGSL compute..."
cp "$RUSTICL_SO" "$STAGING/layer1-compute/libRusticlOpenCL.so.1"
# Symlinks for ICD compatibility
ln -sf libRusticlOpenCL.so.1 "$STAGING/layer1-compute/libRusticlOpenCL.so"
ln -sf libRusticlOpenCL.so.1 "$STAGING/layer1-compute/libOpenCL.so.1"

# ICD file
cat > "$STAGING/layer1-compute/rusticl.icd" << 'EOF'
libRusticlOpenCL.so.1
EOF

# --- Layer 2: Vulkan Backend (Turnip for VkCompute fallback + shader compile) ---
info "Packing Layer 2: Turnip Vulkan backend..."
cp "$TURNIP_SO" "$STAGING/layer2-vulkan/libvulkan_freedreno.so"

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

# Update manifest with hashes
python3 -c "
import json, sys
with open('$STAGING/meta/manifest.json', 'r') as f:
    m = json.load(f)
m['sha256'] = {
    'libRusticlOpenCL.so.1': '$RUSTICL_SHA',
    'libvulkan_freedreno.so': '$TURNIP_SHA'
}
m['validated'] = True
m['pack_date'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
with open('$STAGING/meta/manifest.json', 'w') as f:
    json.dump(m, f, indent=2)
" 2>/dev/null || info "WARN: python3 manifest update skipped"

# Environment loader script
cat > "$STAGING/env.sh" << 'ENVEOF'
#!/bin/bash
# Source this to configure TurboQuant driver environment
# Usage: source /path/to/tq-driver-pack/env.sh

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Layer 1: OpenCL via Rusticl/KGSL
export OCL_ICD_VENDORS="$PACK_DIR/layer1-compute"
export LD_LIBRARY_PATH="$PACK_DIR/layer1-compute${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RUSTICL_CACHE_DIR="$HOME/.cache/rusticl"
export MESA_SHADER_CACHE_DIR="$HOME/.cache/rusticl"

# Layer 2: Vulkan backend (Turnip/kgsl)
export VK_ICD_FILENAMES="$PACK_DIR/layer2-vulkan/freedreno_icd.aarch64.json"
export VK_DRIVER_FILES="$PACK_DIR/layer2-vulkan/freedreno_icd.aarch64.json"

# Performance tuning
export MESA_NO_ERROR=1
export AMD_DEBUG=nodma
export IR3_SHADER_DEBUG=""
export FD_MESA_DEBUG=""

# Ensure cache dir
mkdir -p "$RUSTICL_CACHE_DIR" 2>/dev/null

echo "[tq-drv] Driver pack active: $PACK_DIR"
echo "[tq-drv] Layer1: Rusticl/KGSL (OpenCL 3.0)"
echo "[tq-drv] Layer2: Turnip (Vulkan 1.4.335)"
ENVEOF
chmod +x "$STAGING/env.sh"

# --- Archive ---
mkdir -p "$OUT_DIR"
ARCHIVE="$OUT_DIR/${PACK_NAME}.tar.zst"

info "Creating archive: $ARCHIVE"
tar -C "$STAGING" -cf - . | zstd -T2 -9 -o "$ARCHIVE"

# Cleanup staging
rm -rf "$STAGING"

ARCHIVE_SIZE=$(ls -lh "$ARCHIVE" | awk '{print $5}')
info "Done: $ARCHIVE ($ARCHIVE_SIZE)"
info "SHA256: $(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
info ""
info "Deploy:"
info "  tar --zstd -xf $ARCHIVE -C ~/turboquant/drivers/"
info "  source ~/turboquant/drivers/env.sh"
info "  # Then run TurboQuant — driver auto-detected via tq_driver_loader"
