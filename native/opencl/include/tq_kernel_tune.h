/**
 * TurboQuant Per-Kernel Tuning Profiles.
 *
 * Composable optimization system inspired by GameNative GameFix pattern.
 * Each OpenCL kernel gets optimal parameters based on GPU profile:
 *   - Workgroup size (local_work_size)
 *   - Tile dimensions for tiled kernels
 *   - Memory access pattern (GMEM tiled vs sysmem linear)
 *   - Subgroup utilization strategy
 *   - Register pressure hints
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include "tq_gpu_profile.h"
#include <cstdint>
#include <string>

namespace tq {

enum class MemoryLayout : uint8_t {
    LINEAR = 0,       // Standard row-major
    TILED_4x4 = 1,   // 4x4 tile for GMEM locality
    TILED_8x8 = 2,   // 8x8 tile for large matrices
    TILED_16x4 = 3,  // 16x4 for attention logits (wide rows)
    INTERLEAVED = 4,  // Subgroup-interleaved for coalescing
};

enum class SubgroupUse : uint8_t {
    NONE = 0,         // No subgroup ops
    REDUCE = 1,       // sub_group_reduce_add/max
    SHUFFLE = 2,      // sub_group_shuffle for data sharing
    BROADCAST = 3,    // sub_group_broadcast for leader pattern
};

struct KernelTuneParams {
    std::string kernel_name;

    // Dispatch geometry
    uint32_t wg_size_x = 256;
    uint32_t wg_size_y = 1;
    uint32_t wg_size_z = 1;
    uint32_t preferred_multiple = 64;  // Should be multiple of wave size

    // Tiling
    MemoryLayout input_layout = MemoryLayout::LINEAR;
    MemoryLayout output_layout = MemoryLayout::LINEAR;
    uint32_t tile_m = 1;  // Tile rows
    uint32_t tile_n = 1;  // Tile cols
    uint32_t tile_k = 1;  // Tile depth (for matmul-like)

    // Subgroup
    SubgroupUse subgroup_use = SubgroupUse::NONE;
    uint32_t items_per_thread = 1;  // Work items processed per thread

    // Memory
    uint32_t local_mem_bytes = 0;   // Required __local memory
    bool use_gmem_tiling = false;   // Hint: prefer GMEM path
    bool prefetch_global = false;   // Use async_work_group_copy

    // Vectorization
    uint32_t vec_width = 4;         // float4/half8 etc.
};

/**
 * Get tuned parameters for a kernel on the active GPU profile.
 */
KernelTuneParams get_kernel_tune(const char* kernel_name);

/**
 * Get tuned parameters for a kernel on a specific GPU profile.
 */
KernelTuneParams get_kernel_tune_for(const char* kernel_name, const GpuProfile& profile);

/**
 * Get the optimal local_work_size array for clEnqueueNDRangeKernel.
 * Returns {wg_size_x, wg_size_y, wg_size_z}.
 */
void get_local_work_size(const KernelTuneParams& t, size_t out[3]);

} // namespace tq
