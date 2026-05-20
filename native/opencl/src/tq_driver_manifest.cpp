/**
 * TurboQuant Driver Manifest + Firmware Patching — implementation.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_driver_manifest.h"
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>

namespace tq {

// --- Minimal JSON parser (no external deps) ---
static std::string json_extract_string(const char* buf, const char* key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* pos = strstr(buf, pattern);
    if (!pos) return {};
    pos = strchr(pos + strlen(pattern), '"');
    if (!pos) return {};
    pos++;
    const char* end = strchr(pos, '"');
    if (!end) return {};
    return std::string(pos, end - pos);
}

static bool json_extract_bool(const char* buf, const char* key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* pos = strstr(buf, pattern);
    if (!pos) return false;
    pos += strlen(pattern);
    return strstr(pos, "true") != nullptr && (strstr(pos, "true") - pos) < 20;
}

// --- Firmware version detection ---
std::string detect_firmware_version() {
    // Try sysfs first
    FILE* f = fopen("/sys/class/kgsl/kgsl-3d0/gpu_model", "r");
    if (f) {
        char buf[128] = {};
        if (fgets(buf, sizeof(buf), f)) {
            fclose(f);
            // Strip newline
            char* nl = strchr(buf, '\n');
            if (nl) *nl = '\0';
            return buf;
        }
        fclose(f);
    }

    // Try firmware_version
    f = fopen("/sys/class/kgsl/kgsl-3d0/fw_version", "r");
    if (f) {
        char buf[128] = {};
        if (fgets(buf, sizeof(buf), f)) {
            fclose(f);
            char* nl = strchr(buf, '\n');
            if (nl) *nl = '\0';
            return buf;
        }
        fclose(f);
    }

    // Try gpu_busy_percentage parent for driver version
    f = fopen("/vendor/firmware/a730_sqe.fw", "rb");
    if (f) {
        fclose(f);
        return "a730_fw_present";
    }

    return "unknown";
}

// --- Binary patching ---
int apply_firmware_patches(const char* driver_dir, const DriverPackManifest& m) {
    if (!driver_dir || m.patches.empty()) return 0;

    std::string fw_ver = detect_firmware_version();
    int applied = 0;

    for (auto& patch : m.patches) {
        // Match firmware version (prefix match)
        if (!patch.firmware_version.empty() &&
            fw_ver.find(patch.firmware_version) == std::string::npos) {
            continue;
        }

        // Build full path
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", driver_dir, patch.target_file.c_str());

        // Open file for read+write
        int fd = open(path, O_RDWR);
        if (fd < 0) continue;

        // Seek to offset and write byte
        if (lseek(fd, patch.offset, SEEK_SET) == (off_t)patch.offset) {
            uint8_t val = patch.value;
            if (write(fd, &val, 1) == 1) {
                applied++;
            }
        }
        close(fd);
    }

    return applied;
}

// --- Manifest loading ---
bool load_pack_manifest(const char* json_path, DriverPackManifest* out) {
    if (!json_path || !out) return false;

    FILE* f = fopen(json_path, "r");
    if (!f) return false;

    char buf[8192] = {};
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) return false;

    out->id = json_extract_string(buf, "id");
    out->version = json_extract_string(buf, "version");
    out->provider = json_extract_string(buf, "provider");
    out->library_name = json_extract_string(buf, "library_name");
    // GameNative compat: also try "libraryName"
    if (out->library_name.empty())
        out->library_name = json_extract_string(buf, "libraryName");
    out->sha256_pristine = json_extract_string(buf, "sha256");
    out->min_firmware = json_extract_string(buf, "min_firmware");

    std::string layer_str = json_extract_string(buf, "layer");
    if (layer_str == "compute" || layer_str == "1")
        out->layer = DriverLayer::LAYER1_COMPUTE;
    else
        out->layer = DriverLayer::LAYER2_VULKAN;

    out->has_opencl_3_0 = json_extract_bool(buf, "has_opencl_3_0");
    out->has_vulkan_1_4 = json_extract_bool(buf, "has_vulkan_1_4");
    out->has_fp16 = json_extract_bool(buf, "has_fp16");
    out->has_subgroups = json_extract_bool(buf, "has_subgroups");
    out->has_command_buffer = json_extract_bool(buf, "has_command_buffer");
    out->validated = json_extract_bool(buf, "validated");

    // GameNative compat: "name" field
    if (out->id.empty())
        out->id = json_extract_string(buf, "name");

    // GameNative compat: "driverVersion"
    if (out->version.empty())
        out->version = json_extract_string(buf, "driverVersion");

    return !out->library_name.empty();
}

// --- Manifest saving ---
bool save_pack_manifest(const char* json_path, const DriverPackManifest& m) {
    FILE* f = fopen(json_path, "w");
    if (!f) return false;

    fprintf(f, "{\n");
    fprintf(f, "  \"id\": \"%s\",\n", m.id.c_str());
    fprintf(f, "  \"version\": \"%s\",\n", m.version.c_str());
    fprintf(f, "  \"provider\": \"%s\",\n", m.provider.c_str());
    fprintf(f, "  \"layer\": \"%s\",\n", m.layer == DriverLayer::LAYER1_COMPUTE ? "compute" : "vulkan");
    fprintf(f, "  \"library_name\": \"%s\",\n", m.library_name.c_str());
    fprintf(f, "  \"sha256\": \"%s\",\n", m.sha256_pristine.c_str());
    fprintf(f, "  \"min_firmware\": \"%s\",\n", m.min_firmware.c_str());
    fprintf(f, "  \"has_opencl_3_0\": %s,\n", m.has_opencl_3_0 ? "true" : "false");
    fprintf(f, "  \"has_vulkan_1_4\": %s,\n", m.has_vulkan_1_4 ? "true" : "false");
    fprintf(f, "  \"has_fp16\": %s,\n", m.has_fp16 ? "true" : "false");
    fprintf(f, "  \"has_subgroups\": %s,\n", m.has_subgroups ? "true" : "false");
    fprintf(f, "  \"has_command_buffer\": %s,\n", m.has_command_buffer ? "true" : "false");
    fprintf(f, "  \"validated\": %s\n", m.validated ? "true" : "false");
    fprintf(f, "}\n");
    fclose(f);
    return true;
}

// --- SHA256 stub (link with -lcrypto or implement) ---
bool verify_sha256(const char* file_path, const std::string& expected_hash) {
    if (expected_hash.empty()) return true; // no hash = skip check
    // Future cryptographic verification can use OpenSSL or mbedtls.
    // Current release gate only verifies that the file exists and is non-empty.
    struct stat st;
    if (stat(file_path, &st) != 0 || st.st_size == 0) return false;
    return true;
}

// --- Driver pack scanner ---
bool scan_driver_pack(const char* pack_dir,
                      DriverPackManifest* layer1_out,
                      DriverPackManifest* layer2_out) {
    if (!pack_dir) return false;

    char path[512];
    bool found_l1 = false, found_l2 = false;

    // Try layer1-compute/meta.json
    snprintf(path, sizeof(path), "%s/layer1-compute/meta.json", pack_dir);
    if (layer1_out && load_pack_manifest(path, layer1_out)) {
        layer1_out->layer = DriverLayer::LAYER1_COMPUTE;
        // Set full library path
        char lib_path[512];
        snprintf(lib_path, sizeof(lib_path), "%s/layer1-compute/%s",
                 pack_dir, layer1_out->library_name.c_str());
        layer1_out->library_path = lib_path;
        found_l1 = true;
    }

    // Try layer2-vulkan/meta.json
    snprintf(path, sizeof(path), "%s/layer2-vulkan/meta.json", pack_dir);
    if (layer2_out && load_pack_manifest(path, layer2_out)) {
        layer2_out->layer = DriverLayer::LAYER2_VULKAN;
        char lib_path[512];
        snprintf(lib_path, sizeof(lib_path), "%s/layer2-vulkan/%s",
                 pack_dir, layer2_out->library_name.c_str());
        layer2_out->library_path = lib_path;
        found_l2 = true;
    }

    // Apply firmware patches if needed
    if (found_l1 && !layer1_out->patches.empty()) {
        snprintf(path, sizeof(path), "%s/layer1-compute", pack_dir);
        apply_firmware_patches(path, *layer1_out);
    }
    if (found_l2 && !layer2_out->patches.empty()) {
        snprintf(path, sizeof(path), "%s/layer2-vulkan", pack_dir);
        apply_firmware_patches(path, *layer2_out);
    }

    return found_l1;
}

} // namespace tq
