/**
 * TurboQuant GPU Auto-Profile — per-GPU optimal compute parameters.
 *
 * Derived from GameNative ContainerUtils.setContainerDefaults() pattern,
 * adapted for our stack: MCP → OpenCL (Rusticl/KGSL) → Turnip → nvbuildapi.
 *
 * Each Adreno generation has different optimal:
 *   - Wave size (subgroup): 64 vs 128
 *   - Preferred workgroup size for attention kernels
 *   - Local memory tiling strategy (GMEM vs sysmem)
 *   - Register pressure limits
 *   - PC_MODE_CNTL value (perf register)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cstdint>
#include <string>

namespace tq {

enum class GpuGen : uint8_t {
    ADRENO_6XX = 6,   // A610-A695
    ADRENO_7XX = 7,   // A710-A750
    ADRENO_8XX = 8,   // A810-A850
    UNKNOWN    = 0,
};

enum class WaveMode : uint8_t {
    WAVE64  = 64,
    WAVE128 = 128,
};

enum class TileStrategy : uint8_t {
    GMEM_TILED = 0,   // Use on-chip GMEM for tiled rendering/compute
    SYSMEM     = 1,   // Direct system memory (large buffers)
    AUTO       = 2,   // Runtime decision based on buffer size vs GMEM
};

struct GpuProfile {
    GpuGen gen;
    uint32_t chip_id;
    std::string name;              // "Adreno 730", "Adreno 750", etc.

    // Compute parameters
    WaveMode compute_wave;         // Optimal wave size for compute kernels
    uint32_t preferred_wg_size;    // Optimal workgroup size for attention
    uint32_t max_waves_per_cu;     // Max concurrent waves per SP
    uint32_t reg_size_vec4;        // Register file size in vec4 units
    uint32_t gmem_size_kb;         // On-chip GMEM in KB

    // Tuning knobs
    uint32_t pc_mode_cntl;         // Performance counter mode (0x1f1f for A730)
    TileStrategy tile_strategy;    // Memory access pattern
    bool supports_fp16_fma;        // Native fp16 fused multiply-add
    bool supports_dot_product;     // Integer dot product (dp4a)
    bool dual_wave_dispatch;       // A8XX dual wave64 mode

    // Kernel-specific overrides
    uint32_t attention_wg_size;    // Workgroup for fused attention
    uint32_t dequant_wg_size;     // Workgroup for value dequant
    uint32_t mse_wg_size;         // Workgroup for MSE scoring
    uint32_t qjl_wg_size;         // Workgroup for QJL scoring
};

/**
 * Detect GPU and return optimal profile.
 * Uses KGSL chip_id + GL_RENDERER string matching.
 */
GpuProfile detect_gpu_profile();

/**
 * Get profile by chip_id directly (for testing/override).
 */
GpuProfile profile_from_chip_id(uint32_t chip_id);

/**
 * Get profile by renderer string (fallback when KGSL unavailable).
 */
GpuProfile profile_from_renderer(const char* renderer);

/**
 * Get optimal build options string for clBuildProgram based on profile.
 */
std::string get_build_opts(const GpuProfile& p);

/**
 * Get the active profile (cached after first detect).
 */
const GpuProfile& active_profile();

} // namespace tq
