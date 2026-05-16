/**
 * TurboQuant Adreno Quirks Registry — device-specific dispatch policies.
 * Vortek-inspired driver workaround layer for Adreno/OpenCL.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cstdint>
#include <string>

namespace tq {

enum class BackendRoute {
    OPENCL_ADRENO,
    OPENCL_TILED,
    NEON,
    CPU_SCALAR,
    VULKAN_COMPUTE,
    MESA_RUSTICL,
    AUTO,
};

struct DeviceProfile {
    std::string gpu_name;
    std::string opencl_version;
    bool fp16 = false;
    bool subgroups = false;
    uint32_t max_work_group_size = 0;
    uint32_t compute_units = 0;
    uint64_t global_mem_bytes = 0;
};

struct KernelPolicy {
    BackendRoute mse_score = BackendRoute::AUTO;
    BackendRoute qjl_score = BackendRoute::AUTO;
    BackendRoute value_dequant = BackendRoute::AUTO;
    BackendRoute fused_attention = BackendRoute::AUTO;
    int fused_attention_neon_threshold_tokens = 64;  // below this, NEON may be faster
    bool enable_subgroup_path = false;
    bool enable_fp16_values = false;
    bool three_bit_cross_byte_checked = true;
};

struct QuirksProfile {
    DeviceProfile device;
    KernelPolicy policy;
};

// Build quirks profile from device probe results
QuirksProfile build_quirks_profile(const DeviceProfile& device);

// Get recommended backend for a specific kernel + shape
BackendRoute route_kernel(const QuirksProfile& profile, const std::string& kernel, int tokens, int bits);

// Serialize profile to JSON string
std::string quirks_to_json(const QuirksProfile& profile);

} // namespace tq
