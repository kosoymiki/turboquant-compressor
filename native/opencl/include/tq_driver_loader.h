/**
 * TurboQuant Custom Driver Loader — multi-backend GPU dispatch.
 *
 * Priority tiers (forensic-validated):
 *   T0: Custom Rusticl/kgsl (our fork, max perf, full control, production path)
 *   T1: Vendor Qualcomm OpenCL (legacy opt-in fallback only)
 *   T2: System Rusticl via ICD (legacy opt-in fallback only)
 *   T3: Turnip Vulkan compute (stack-local fallback via kgsl)
 *   T4: CPU fallback (WASM SIMD128 or scalar)
 *
 * Forensic backing:
 *   - kgsl_forensic.h tracing model
 *   - current Mesa KGSL probe route through pipe_loader_kgsl.c / freedreno kgsl backend
 *   - Turnip kgsl direct ioctl path
 *
 * Runtime contract:
 *   - Production execution is env.sh + OCL ICD + packaged Rusticl layer.
 *   - This loader is a stack-aware selector and forensic validator.
 *   - Vendor/system OpenCL fallbacks are disabled unless
 *     TQ_ENABLE_LEGACY_OPENCL_FALLBACKS=1 is set explicitly.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include "tq_opencl_types.h"
#include <cstdint>
#include <string>

namespace tq {

enum class DriverTier : uint8_t {
    RUSTICL_KGSL = 0,   // T0: our fork, /dev/kgsl-3d0 direct
    VENDOR_CL    = 1,   // T1: /vendor/lib64/libOpenCL.so
    SYSTEM_ICD   = 2,   // T2: ICD loader → any platform
    TURNIP_VK    = 3,   // T3: Vulkan compute via Turnip
    CPU_FALLBACK = 4,   // T4: no GPU
};

struct DriverCaps {
    DriverTier tier;
    std::string lib_path;        // actual .so loaded
    std::string device_name;
    uint32_t compute_units;
    size_t max_wg_size;
    size_t local_mem_size;
    uint32_t subgroup_size;      // 0 = unknown
    bool has_fp16;
    bool has_subgroups;
    bool has_kgsl;               // direct /dev/kgsl-3d0 access
    bool has_fma;
    double probe_latency_ms;
};

struct DriverHandle {
    void* lib;                   // dlopen handle
    void* cl_context;            // cl_context or VkDevice
    void* cl_queue;              // cl_command_queue or VkQueue
    void* cl_device;             // cl_device_id or VkPhysicalDevice
    DriverTier tier;
    bool valid;
};

/**
 * Probe all available backends, select highest priority.
 * Returns caps of selected driver. Non-blocking, <50ms target.
 */
DriverCaps probe_best_driver();

/**
 * Load and initialize the selected driver tier.
 * If tier fails, falls through to next available.
 */
DriverHandle load_driver(DriverTier preferred = DriverTier::RUSTICL_KGSL);

/**
 * Release driver resources.
 */
void unload_driver(DriverHandle& h);

/**
 * Get active driver info string for diagnostics.
 */
std::string driver_info_string(const DriverCaps& caps);

/**
 * Driver manifest for deployment packaging (aesolator-inspired).
 */
struct DriverManifest {
    std::string id;
    std::string version;
    std::string provider;
    std::string library_name;
    std::string sha256;
    DriverTier tier = DriverTier::CPU_FALLBACK;
    bool validated = false;
};

/**
 * Load driver manifest from JSON file.
 */
bool load_manifest(const char* json_path, DriverManifest* out);

} // namespace tq
