/**
 * Kernel binary cache test — stable hash + store/load roundtrip.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_kernel_cache.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

int main() {
    namespace fs = std::filesystem;

    fs::path root = fs::temp_directory_path() / "tq-kernel-cache-test";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    if (ec) {
        std::fprintf(stderr, "FAIL: cannot create temp cache dir: %s\n", root.c_str());
        return 1;
    }

    tq::KernelBinaryCache cache(root.string());
    const std::string digest_a = cache.device_hash("Adreno (TM) 750", "Mesa 26.2", 12, 256);
    const std::string digest_b = cache.device_hash("Adreno (TM) 750", "Mesa 26.2", 12, 256);
    const std::string digest_c = cache.device_hash("Adreno (TM) 750", "Mesa 26.3", 12, 256);

    if (digest_a != digest_b) {
        std::fprintf(stderr, "FAIL: identical device tuple produced different hash\n");
        return 1;
    }
    if (digest_a == digest_c) {
        std::fprintf(stderr, "FAIL: changed driver version did not change hash\n");
        return 1;
    }

    const std::vector<uint8_t> expected = {0x03u, 0x14u, 0x15u, 0x92u, 0x65u};
    cache.store("tq_mse_score", digest_a, expected);

    std::vector<uint8_t> loaded;
    if (!cache.load("tq_mse_score", digest_a, loaded)) {
        std::fprintf(stderr, "FAIL: stored binary could not be loaded back\n");
        return 1;
    }
    if (loaded != expected) {
        std::fprintf(stderr, "FAIL: loaded binary payload mismatch\n");
        return 1;
    }

    std::vector<uint8_t> missing;
    if (cache.load("tq_qjl_score", digest_a, missing)) {
        std::fprintf(stderr, "FAIL: unexpected load success for missing cache entry\n");
        return 1;
    }

    std::printf("Kernel cache test: PASS\n");
    return 0;
}
