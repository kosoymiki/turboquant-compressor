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

namespace {

static std::string sha256_bytes(const uint8_t* data, size_t size) {
    static const uint32_t k[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };

    auto rotr = [](uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); };

    std::vector<uint8_t> msg(data, data + size);
    uint64_t bit_len = static_cast<uint64_t>(msg.size()) * 8u;
    msg.push_back(0x80u);
    while ((msg.size() % 64u) != 56u) msg.push_back(0u);
    for (int i = 7; i >= 0; --i) msg.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xFFu));

    uint32_t h[8] = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
    };

    for (size_t chunk = 0; chunk < msg.size(); chunk += 64u) {
        uint32_t w[64] = {};
        for (size_t i = 0; i < 16; ++i) {
            size_t off = chunk + i * 4u;
            w[i] = (static_cast<uint32_t>(msg[off]) << 24) |
                   (static_cast<uint32_t>(msg[off + 1]) << 16) |
                   (static_cast<uint32_t>(msg[off + 2]) << 8) |
                   static_cast<uint32_t>(msg[off + 3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

        for (size_t i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;

            hh = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    char out[65];
    std::snprintf(out, sizeof(out),
                  "%08x%08x%08x%08x%08x%08x%08x%08x",
                  h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    return out;
}

}

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

bool verify_sha256(const char* file_path, const std::string& expected_hash) {
    if (expected_hash.empty()) return true; // no hash = skip check
    if (!file_path || expected_hash.empty()) return false;

    int fd = open(file_path, O_RDONLY);
    if (fd < 0) return false;

    std::vector<uint8_t> data;
    uint8_t buffer[4096];
    for (;;) {
        ssize_t n = read(fd, buffer, sizeof(buffer));
        if (n < 0) {
            close(fd);
            return false;
        }
        if (n == 0) break;
        data.insert(data.end(), buffer, buffer + n);
    }
    close(fd);
    if (data.empty()) return false;

    return sha256_bytes(data.data(), data.size()) == expected_hash;
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
