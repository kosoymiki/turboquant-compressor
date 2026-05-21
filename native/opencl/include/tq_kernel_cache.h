/**
 * TurboQuant OpenCL — compiled program binary cache.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tq {

class KernelBinaryCache {
public:
    explicit KernelBinaryCache(const std::string& cache_root = "");

    bool load(
        const std::string& kernel_name,
        const std::string& device_hash,
        std::vector<uint8_t>& binary) const;

    void store(
        const std::string& kernel_name,
        const std::string& device_hash,
        const std::vector<uint8_t>& binary) const;

    std::string device_hash(
        const std::string& device_name,
        const std::string& driver_version,
        uint32_t compute_units,
        size_t max_wg_size) const;

    const std::string& cache_dir() const { return cache_dir_; }

private:
    std::string cache_dir_;
};

} // namespace tq
