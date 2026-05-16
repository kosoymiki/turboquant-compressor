/**
 * TurboQuant Adreno Quirks Registry — device-specific dispatch policies.
 * Builds kernel routing policy from device probe facts + benchmark evidence.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_quirks.h"
#include <cstdio>

namespace tq {

QuirksProfile build_quirks_profile(const DeviceProfile& device) {
    QuirksProfile profile;
    profile.device = device;

    // Default: route individual kernels to OpenCL when available
    if (device.opencl_version.find("OpenCL") != std::string::npos) {
        profile.policy.mse_score = BackendRoute::OPENCL_ADRENO;
        profile.policy.qjl_score = BackendRoute::OPENCL_ADRENO;
        profile.policy.value_dequant = BackendRoute::OPENCL_ADRENO;
        profile.policy.fused_attention = BackendRoute::OPENCL_TILED;
    } else {
        profile.policy.mse_score = BackendRoute::NEON;
        profile.policy.qjl_score = BackendRoute::NEON;
        profile.policy.value_dequant = BackendRoute::NEON;
        profile.policy.fused_attention = BackendRoute::NEON;
    }

    // Subgroup path: enable only if device reports support
    profile.policy.enable_subgroup_path = device.subgroups;

    // fp16 values: experimental, disabled by default even if device supports it
    profile.policy.enable_fp16_values = false;

    // 3-bit packing always needs cross-byte check
    profile.policy.three_bit_cross_byte_checked = true;

    // Adreno-specific tuning by generation
    if (device.gpu_name.find("Adreno") != std::string::npos) {
        // Adreno 7xx (Snapdragon 8 Gen 1/2/3): 4 CUs, 1024 max WG
        if (device.compute_units >= 4 && device.max_work_group_size >= 1024) {
            profile.policy.fused_attention_neon_threshold_tokens = 32;
        }
        // Adreno 6xx (Snapdragon 8xx older): smaller, higher threshold
        else if (device.compute_units >= 2) {
            profile.policy.fused_attention_neon_threshold_tokens = 128;
        }
    }

    return profile;
}

BackendRoute route_kernel(const QuirksProfile& profile, const std::string& kernel, int tokens, int bits) {
    (void)bits; // reserved for future bit-width-specific routing

    if (kernel == "mse_score") return profile.policy.mse_score;
    if (kernel == "qjl_score") return profile.policy.qjl_score;
    if (kernel == "value_dequant") return profile.policy.value_dequant;

    if (kernel == "fused_attention") {
        // Small token counts: NEON may be faster due to GPU dispatch overhead
        if (tokens <= profile.policy.fused_attention_neon_threshold_tokens) {
            return BackendRoute::NEON;
        }
        return profile.policy.fused_attention;
    }

    return BackendRoute::AUTO;
}

static const char* route_name(BackendRoute r) {
    switch (r) {
        case BackendRoute::OPENCL_ADRENO: return "opencl_adreno";
        case BackendRoute::OPENCL_TILED: return "opencl_tiled";
        case BackendRoute::NEON: return "neon";
        case BackendRoute::CPU_SCALAR: return "cpu_scalar";
        case BackendRoute::VULKAN_COMPUTE: return "vulkan_compute";
        case BackendRoute::MESA_RUSTICL: return "mesa_rusticl";
        case BackendRoute::AUTO: return "auto";
    }
    return "unknown";
}

std::string quirks_to_json(const QuirksProfile& profile) {
    char buf[2048];
    snprintf(buf, sizeof(buf),
        "{\n"
        "  \"device\": {\n"
        "    \"gpu\": \"%s\",\n"
        "    \"opencl\": \"%s\",\n"
        "    \"fp16\": %s,\n"
        "    \"subgroups\": %s,\n"
        "    \"maxWorkGroupSize\": %u,\n"
        "    \"computeUnits\": %u,\n"
        "    \"globalMemBytes\": %llu\n"
        "  },\n"
        "  \"policies\": {\n"
        "    \"mse_score\": \"%s\",\n"
        "    \"qjl_score\": \"%s\",\n"
        "    \"value_dequant\": \"%s\",\n"
        "    \"fused_attention\": \"%s\",\n"
        "    \"fused_attention_neon_threshold_tokens\": %d,\n"
        "    \"enable_subgroup_path\": %s,\n"
        "    \"enable_fp16_values\": %s,\n"
        "    \"three_bit_cross_byte_checked\": %s\n"
        "  }\n"
        "}",
        profile.device.gpu_name.c_str(),
        profile.device.opencl_version.c_str(),
        profile.device.fp16 ? "true" : "false",
        profile.device.subgroups ? "true" : "false",
        profile.device.max_work_group_size,
        profile.device.compute_units,
        (unsigned long long)profile.device.global_mem_bytes,
        route_name(profile.policy.mse_score),
        route_name(profile.policy.qjl_score),
        route_name(profile.policy.value_dequant),
        route_name(profile.policy.fused_attention),
        profile.policy.fused_attention_neon_threshold_tokens,
        profile.policy.enable_subgroup_path ? "true" : "false",
        profile.policy.enable_fp16_values ? "true" : "false",
        profile.policy.three_bit_cross_byte_checked ? "true" : "false"
    );
    return std::string(buf);
}

} // namespace tq
