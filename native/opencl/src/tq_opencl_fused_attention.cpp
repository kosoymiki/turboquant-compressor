/**
 * TurboQuant OpenCL — fused attention host code.
 * CPU reference: online softmax over compressed KV, no full dequant buffer.
 * Real OpenCL dispatch with profiling.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include <CL/cl.h>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

namespace tq {

static inline uint32_t fused_extract_bits(const uint8_t* packed, int coord, int bits) {
    uint32_t bit_pos = (uint32_t)coord * (uint32_t)bits;
    uint32_t byte_idx = bit_pos >> 3;
    uint32_t bit_off = bit_pos & 7u;
    uint32_t word = packed[byte_idx];
    if (bit_off + (uint32_t)bits > 8u) word |= ((uint32_t)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & ((1u << bits) - 1u);
}

__attribute__((unused))
static inline uint32_t fused_popcount32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    return ((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u >> 24;
}

void fused_attention_cpu_reference(const FusedAttentionInput& input) {
    int dim = input.mse.dim;
    int tokens = input.mse.tokens;
    int val_packed_stride = (input.value.dim * input.value.bits + 7) / 8;
    int n_groups = (input.value.dim + input.value.group_size - 1) / input.value.group_size;

    std::vector<float> scores(tokens);
    mse_score_cpu_reference(input.mse, scores.data());

    QjlScoreInput qjl_in = input.qjl;
    qjl_in.base_scores = scores.data();
    qjl_score_cpu_reference(qjl_in);

    for (int n = 0; n < tokens; n++) scores[n] *= input.sm_scale;

    float m_i = -1e30f;
    float l_i = 0.0f;
    std::vector<float> acc(dim, 0.0f);

    for (int n = 0; n < tokens; n++) {
        float s = scores[n];
        float m_new = std::max(m_i, s);
        float alpha = std::exp(m_i - m_new);
        float p = std::exp(s - m_new);
        l_i = l_i * alpha + p;

        for (int j = 0; j < dim; j++) acc[j] *= alpha;

        const uint8_t* v_packed = input.value.packed_values + n * val_packed_stride;
        for (int j = 0; j < dim; j++) {
            uint32_t raw = fused_extract_bits(v_packed, j, input.value.bits);
            int g = j / input.value.group_size;
            float val = (float)raw * input.value.scales[n * n_groups + g]
                       + input.value.zeros[n * n_groups + g];
            acc[j] += p * val;
        }

        m_i = m_new;
    }

    if (l_i > 0.0f) {
        for (int j = 0; j < dim; j++) input.output[j] = acc[j] / l_i;
    } else {
        std::memset(input.output, 0, dim * sizeof(float));
    }
}

TqStatus fused_attention_opencl(const FusedAttentionInput& input) {
    return fused_attention_tiled_opencl(input);
}

TqStatus fused_attention_opencl_profiled(const FusedAttentionInput& input, uint64_t* kernel_ns) {
    return fused_attention_tiled_opencl_profiled(input, kernel_ns);
}

} // namespace tq
