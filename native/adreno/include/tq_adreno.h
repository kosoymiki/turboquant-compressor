/**
 * TurboQuant Adreno runtime — types and backend interface.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cstdint>
#include <cstddef>

namespace tq {

enum BackendKind {
    BACKEND_CPU_SCALAR = 0,
    BACKEND_CPU_NEON = 1,
    BACKEND_VENDOR_OPENCL = 2,
    BACKEND_MESA_RUSTICL = 3,
    BACKEND_VULKAN_COMPUTE = 4,
    BACKEND_ANDROID_SERVICE = 5,
};

struct MseInput {
    const uint8_t* packed_indices;
    const float* norms;
    const float* query_rotated;
    const float* centroids;
    int tokens;
    int dim;
    int bits;
    int packed_stride;
};

struct ValueDequantInput {
    const uint8_t* packed_values;
    const float* scales;
    const float* zeros;
    int tokens;
    int dim;
    int bits;
    int group_size;
};

struct FusedAttentionInput {
    MseInput mse;
    const uint32_t* query_signs;
    const uint32_t* residual_signs;
    const float* residual_norms;
    int projections;
    float qjl_scale;
    const uint8_t* val_packed;
    const float* val_scales;
    const float* val_zeros;
    int val_bits;
    int val_group_size;
    float sm_scale;
    float* output;
};

// Backend function pointers
typedef void (*MseScoreFn)(const MseInput& input, float* scores);
typedef void (*ValueDequantFn)(const ValueDequantInput& input, float* output);
typedef void (*FusedAttentionFn)(const FusedAttentionInput& input);

struct Backend {
    BackendKind kind;
    const char* name;
    MseScoreFn mse_score;
    ValueDequantFn value_dequant;
    FusedAttentionFn fused_attention;
};

const Backend& get_backend();
const Backend& get_backend_by_kind(BackendKind kind);
bool has_neon();

// CPU scalar implementations
void cpu_scalar_mse_score(const MseInput& input, float* scores);
void cpu_scalar_value_dequant(const ValueDequantInput& input, float* output);
void cpu_scalar_fused_attention(const FusedAttentionInput& input);

// NEON implementations
void neon_mse_score(const MseInput& input, float* scores);
void neon_value_dequant(const ValueDequantInput& input, float* output);
void neon_fused_attention(const FusedAttentionInput& input);

} // namespace tq
