/**
 * TurboQuant OpenCL — type definitions.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tq {

struct DeviceInfo {
    std::string name;
    std::string vendor;
    std::string version;
    std::string device_uuid;
    bool has_fp16 = false;
    bool has_subgroups = false;
    bool has_subgroup_shuffle = false;
    bool has_subgroup_shuffle_relative = false;
    bool has_subgroup_ballot = false;
    bool has_subgroup_clustered_reduce = false;
    bool has_subgroup_non_uniform_arithmetic = false;
    bool has_subgroup_forward_progress = false;
    bool has_il_program = false;
    bool has_async_program_compilation = false;
    bool has_integer_dot_product = false;
    bool has_suggested_local_work_size = false;
    bool has_create_command_queue = false;
    bool has_initialize_memory = false;
    bool has_device_uuid = false;
    bool has_priority_hints = false;
    bool has_throttle_hints = false;
    bool has_command_buffer = false;
    bool has_command_buffer_mutable_dispatch = false;
    bool has_external_memory = false;
    bool has_external_memory_ahb = false;
    bool has_external_semaphore = false;
    bool has_svm = false;
    bool has_svm_coarse = false;
    bool has_svm_fine = false;
    bool has_svm_atomics = false;
    bool is_adreno = false;
    uint64_t global_mem_bytes = 0;
    uint32_t max_work_group_size = 0;
    uint32_t compute_units = 0;
    uint32_t max_subgroups = 0;
    uint32_t integer_dot_product_capabilities = 0;
    uint64_t command_buffer_capabilities = 0;
};

struct ProbeResult {
    bool available = false;
    int platform_count = 0;
    int device_count = 0;
    std::vector<DeviceInfo> devices;
    std::string recommended_backend; // "opencl_adreno" | "opencl_generic" | "unavailable"
    std::vector<std::string> warnings;
    double probe_time_ms = 0.0;
};

enum class TqStatus {
    OK = 0,
    ERR_NO_PLATFORM,
    ERR_NO_DEVICE,
    ERR_BUILD_FAILED,
    ERR_KERNEL_LAUNCH,
    ERR_OUT_OF_MEMORY,
    ERR_INVALID_ARG,
    ERR_TIMEOUT,
};

struct MseScoreInput {
    const uint8_t* packed_indices;  // [tokens * packed_stride]
    const float* norms;             // [tokens]
    const float* query_rotated;     // [dim]
    const float* centroids;         // [2^bits]
    int tokens;
    int dim;
    int bits;
    int packed_stride;
};

struct QjlScoreInput {
    const uint32_t* query_signs;     // [proj_words]
    const uint32_t* residual_signs;  // [tokens * proj_words]
    const float* residual_norms;     // [tokens]
    float* base_scores;              // [tokens] — in/out
    int tokens;
    int projections;
    float qjl_scale;
};

struct ValueDequantInput {
    const uint8_t* packed_values;  // [tokens * packed_stride]
    const float* scales;           // [tokens * n_groups]
    const float* zeros;            // [tokens * n_groups]
    int tokens;
    int dim;
    int bits;
    int group_size;
};

struct FusedAttentionInput {
    MseScoreInput mse;
    QjlScoreInput qjl;
    ValueDequantInput value;
    float sm_scale;
    float* output;  // [dim]
};

} // namespace tq
