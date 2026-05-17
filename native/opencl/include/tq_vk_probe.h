/**
 * TurboQuant Vulkan Runtime Probe — detect Turnip/driver capabilities.
 *
 * From GameNative gpu_helper.c pattern: dlopen("libvulkan.so") → probe.
 * Used by layer2-vulkan to determine available extensions and API version
 * before dispatching VkCompute workloads.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tq {

struct VkProbeResult {
    bool available = false;
    uint32_t api_version = 0;       // VK_MAKE_VERSION packed
    uint8_t api_major = 0;
    uint8_t api_minor = 0;
    uint16_t api_patch = 0;
    std::string device_name;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t driver_version = 0;
    uint32_t subgroup_size = 0;
    std::vector<std::string> extensions;
    bool has_compute_queue = false;
    bool has_16bit_storage = false;
    bool has_8bit_storage = false;
    bool has_shader_float16 = false;
    bool has_shader_int8 = false;
    bool has_subgroup_basic = false;
    bool has_subgroup_vote = false;
    bool has_subgroup_arithmetic = false;
    bool has_subgroup_shuffle = false;
    bool has_descriptor_indexing = false;
    bool has_buffer_device_address = false;
    bool has_spirv_1_4 = false;
    bool is_turnip = false;
    double probe_time_ms = 0.0;
};

/**
 * Probe Vulkan via dlopen. Does NOT require linked libvulkan.
 * Loads from driver_path if non-null, else system libvulkan.so.
 */
VkProbeResult probe_vulkan(const char* driver_path = nullptr);

/**
 * Check if a specific extension is present.
 */
bool vk_has_extension(const VkProbeResult& r, const char* ext_name);

} // namespace tq
