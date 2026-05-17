/**
 * TurboQuant Driver Manifest + Firmware Patching.
 *
 * Two-layer driver pack management:
 *   layer1-compute: libRusticlOpenCL.so (OpenCL 3.0 via KGSL)
 *   layer2-vulkan:  libvulkan_freedreno.so (Turnip VK 1.4)
 *
 * Firmware patching from GameNative AdrenotoolsManager pattern:
 *   - Detect stock GPU firmware version via KGSL/GL_VERSION
 *   - Apply binary patches at known offsets for compatibility
 *   - Validate SHA256 before/after patching
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tq {

enum class DriverLayer : uint8_t {
    LAYER1_COMPUTE = 1,  // Rusticl OpenCL
    LAYER2_VULKAN  = 2,  // Turnip Vulkan
};

struct FirmwarePatch {
    std::string firmware_version;  // e.g. "512.530", "512.502", "615.x"
    std::string target_file;       // relative to driver dir, e.g. "notadreno_utils.so"
    uint64_t offset;               // byte offset in file
    uint8_t value;                 // byte to write
    std::string description;       // human-readable
};

struct DriverPackManifest {
    // Identity
    std::string id;                // e.g. "tq-rusticl-a730-v26.1"
    std::string version;           // e.g. "26.1.0-tq3"
    std::string provider;          // "turboquantum"
    DriverLayer layer;

    // Files
    std::string library_name;      // e.g. "libRusticlOpenCL.so.1"
    std::string library_path;      // full path after deployment
    std::string sha256_pristine;   // hash before patching
    std::string sha256_patched;    // hash after patching (if applicable)

    // Compatibility
    std::string min_firmware;      // minimum firmware version
    std::vector<std::string> supported_chips;  // chip_id hex strings
    std::vector<FirmwarePatch> patches;        // firmware-specific patches

    // Capabilities declared
    bool has_opencl_3_0 = false;
    bool has_vulkan_1_4 = false;
    bool has_fp16 = false;
    bool has_subgroups = false;
    bool has_command_buffer = false;

    // Validation
    bool validated = false;
    double load_time_ms = 0.0;
};

/**
 * Load manifest from JSON file (meta.json in driver directory).
 * Compatible with GameNative adrenotools meta.json format.
 */
bool load_pack_manifest(const char* json_path, DriverPackManifest* out);

/**
 * Write manifest to JSON file.
 */
bool save_pack_manifest(const char* json_path, const DriverPackManifest& m);

/**
 * Detect current GPU firmware version string.
 * Reads from /sys/class/kgsl/kgsl-3d0/gpu_model or EGL GL_VERSION.
 */
std::string detect_firmware_version();

/**
 * Apply firmware-specific patches to driver .so file.
 * Returns number of patches applied, -1 on error.
 */
int apply_firmware_patches(const char* driver_dir, const DriverPackManifest& m);

/**
 * Verify SHA256 of a file matches expected hash.
 */
bool verify_sha256(const char* file_path, const std::string& expected_hash);

/**
 * Scan driver pack directory, load both layer manifests.
 * Returns true if at least layer1 is valid.
 */
bool scan_driver_pack(const char* pack_dir,
                      DriverPackManifest* layer1_out,
                      DriverPackManifest* layer2_out);

} // namespace tq
