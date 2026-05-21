/**
 * TurboQuant OpenCL — compiled program binary cache.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_kernel_cache.h"
#include "../include/tq_repo_paths.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace tq {

namespace {

std::string sha256_hex(const std::string& input) {
    // Compact public-domain style SHA-256 implementation for stable cache keys.
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

    std::vector<uint8_t> msg(input.begin(), input.end());
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

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint32_t v : h) out << std::setw(8) << v;
    return out.str();
}

std::string join_path(const std::string& base, const std::string& leaf) {
    if (base.empty()) return leaf;
    if (!base.empty() && base.back() == '/') return base + leaf;
    return base + "/" + leaf;
}

} // namespace

KernelBinaryCache::KernelBinaryCache(const std::string& cache_root) {
    if (!cache_root.empty()) {
        cache_dir_ = cache_root;
    } else {
        std::string forensics_dir = resolve_forensics_dir();
        cache_dir_ = forensics_dir.empty()
            ? std::string("forensics/kernel-cache")
            : join_path(forensics_dir, "kernel-cache");
    }
    std::error_code ec;
    std::filesystem::create_directories(cache_dir_, ec);
}

bool KernelBinaryCache::load(
    const std::string& kernel_name,
    const std::string& device_hash,
    std::vector<uint8_t>& binary) const {
    std::ifstream in(join_path(cache_dir_, kernel_name + "_" + device_hash + ".bin"), std::ios::binary);
    if (!in.is_open()) return false;
    binary.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return !binary.empty();
}

void KernelBinaryCache::store(
    const std::string& kernel_name,
    const std::string& device_hash,
    const std::vector<uint8_t>& binary) const {
    if (binary.empty()) return;
    std::ofstream out(join_path(cache_dir_, kernel_name + "_" + device_hash + ".bin"), std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return;
    out.write(reinterpret_cast<const char*>(binary.data()), static_cast<std::streamsize>(binary.size()));
}

std::string KernelBinaryCache::device_hash(
    const std::string& device_name,
    const std::string& driver_version,
    uint32_t compute_units,
    size_t max_wg_size) const {
    std::ostringstream id;
    id << device_name << '\n'
       << driver_version << '\n'
       << compute_units << '\n'
       << max_wg_size;
    return sha256_hex(id.str());
}

} // namespace tq
