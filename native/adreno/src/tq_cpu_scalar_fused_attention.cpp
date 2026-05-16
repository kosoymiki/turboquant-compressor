/**
 * TurboQuant — CPU scalar fused attention (online softmax, no full dequant buffer).
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_adreno.h"
#include <cmath>
#include <cstring>

namespace tq {

static inline uint32_t fa_extract_bits(const uint8_t* packed, int coord, int bits) {
    uint32_t bit_pos = (uint32_t)coord * (uint32_t)bits;
    uint32_t byte_idx = bit_pos >> 3;
    uint32_t bit_off = bit_pos & 7u;
    uint32_t word = packed[byte_idx];
    if (bit_off + (uint32_t)bits > 8u) word |= ((uint32_t)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & ((1u << bits) - 1u);
}

static inline uint32_t popcount32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    return ((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u >> 24;
}

void cpu_scalar_fused_attention(const FusedAttentionInput& input) {
    int dim = input.mse.dim;
    int tokens = input.mse.tokens;
    int proj_words = (input.projections + 31) / 32;
    int val_packed_stride = (dim * input.val_bits + 7) / 8;
    int val_n_groups = (dim + input.val_group_size - 1) / input.val_group_size;

    float m_i = -1e30f, l_i = 0.0f;
    memset(input.output, 0, dim * sizeof(float));

    for (int n = 0; n < tokens; n++) {
        // MSE score
        const uint8_t* k_packed = input.mse.packed_indices + n * input.mse.packed_stride;
        float dot = 0.0f;
        for (int j = 0; j < dim; j++) {
            uint32_t idx = fa_extract_bits(k_packed, j, input.mse.bits);
            dot += input.mse.query_rotated[j] * input.mse.centroids[idx];
        }
        float score = dot * input.mse.norms[n];

        // QJL correction
        const uint32_t* r_signs = input.residual_signs + n * proj_words;
        uint32_t hamming = 0;
        for (int w = 0; w < proj_words; w++)
            hamming += popcount32(input.query_signs[w] ^ r_signs[w]);
        float sign_dot = (float)input.projections - 2.0f * (float)hamming;
        score += input.qjl_scale * input.residual_norms[n] * sign_dot;
        score *= input.sm_scale;

        // Online softmax
        float m_new = score > m_i ? score : m_i;
        float alpha = std::exp(m_i - m_new);
        float p = std::exp(score - m_new);
        l_i = l_i * alpha + p;

        // Rescale + accumulate value
        const uint8_t* v_packed = input.val_packed + n * val_packed_stride;
        for (int j = 0; j < dim; j++) {
            input.output[j] *= alpha;
            uint32_t raw = fa_extract_bits(v_packed, j, input.val_bits);
            int g = j / input.val_group_size;
            float val = (float)raw * input.val_scales[n * val_n_groups + g]
                       + input.val_zeros[n * val_n_groups + g];
            input.output[j] += p * val;
        }
        m_i = m_new;
    }

    if (l_i > 0.0f) {
        for (int j = 0; j < dim; j++) input.output[j] /= l_i;
    }
}

} // namespace tq
