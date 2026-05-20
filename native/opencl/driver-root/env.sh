#!/bin/bash
# Source this to configure TurboQuant driver environment
# Usage: source /path/to/driver-root/env.sh

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TQ_DRIVER_ROOT="$PACK_DIR"

# Layer 1: OpenCL via Rusticl/KGSL
export OCL_ICD_VENDORS="$PACK_DIR/layer1-compute"
export LD_LIBRARY_PATH="$PACK_DIR/layer1-compute${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TQ_OPENCL_KERNEL_DIR="$PACK_DIR/kernels"
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
