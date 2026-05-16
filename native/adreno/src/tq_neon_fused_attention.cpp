/**
 * TurboQuant — NEON fused attention (delegates to scalar with NEON value dequant).
 * Full NEON fused path uses scalar online softmax + NEON inner loops.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_adreno.h"

#ifdef TQ_HAS_NEON
#include <arm_neon.h>
#endif

#include <cmath>
#include <cstring>

namespace tq {

static inline uint32_t nfa_extract_bits(const uint8_t* packed, int coord, int bits) {
    uint32_t bit_pos = (uint32_t)coord * (uint32_t)bits;
    uint32_t byte_idx = bit_pos >> 3;
    uint32_t bit_off = bit_pos & 7u;
    uint32_t word = packed[byte_idx];
    if (bit_off + (uint32_t)bits > 8u) word |= ((uint32_t)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & ((1u << bits) - 1u);
}

static inline uint32_t nfa_popcount32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    return ((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u >> 24;
}

void neon_fused_attention(const FusedAttentionInput& input) {
#ifdef TQ_HAS_NEON
    int dim = input.mse.dim;
    int tokens = input.mse.tokens;
    int proj_words = (input.projections + 31) / 32;
    int val_packed_stride = (dim * input.val_bits + 7) / 8;
    int val_n_groups = (dim + input.val_group_size - 1) / input.val_group_size;

    float m_i = -1e30f, l_i = 0.0f;
    memset(input.output, 0, dim * sizeof(float));

    for (int n = 0; n < tokens; n++) {
        // MSE score with NEON
        const uint8_t* k_packed = input.mse.packed_indices + n * input.mse.packed_stride;
        float32x4_t acc = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 3 < dim; j += 4) {
            float cv[4];
            for (int k = 0; k < 4; k++)
                cv[k] = input.mse.centroids[nfa_extract_bits(k_packed, j + k, input.mse.bits)];
            acc = vmlaq_f32(acc, vld1q_f32(input.mse.query_rotated + j), vld1q_f32(cv));
        }
        float dot = vaddvq_f32(acc);
        for (; j < dim; j++)
            dot += input.mse.query_rotated[j] * input.mse.centroids[nfa_extract_bits(k_packed, j, input.mse.bits)];

        float score = dot * input.mse.norms[n];

        // QJL
        const uint32_t* r_signs = input.residual_signs + n * proj_words;
        uint32_t hamming = 0;
        for (int w = 0; w < proj_words; w++)
            hamming += nfa_popcount32(input.query_signs[w] ^ r_signs[w]);
        score += input.qjl_scale * input.residual_norms[n] * ((float)input.projections - 2.0f * (float)hamming);
        score *= input.sm_scale;

        // Online softmax
        float m_new = score > m_i ? score : m_i;
        float alpha = std::exp(m_i - m_new);
        float p = std::exp(score - m_new);
        l_i = l_i * alpha + p;

        // Rescale + accumulate with NEON
        float32x4_t v_alpha = vdupq_n_f32(alpha);
        float32x4_t v_p = vdupq_n_f32(p);
        const uint8_t* v_packed = input.val_packed + n * val_packed_stride;

        j = 0;
        for (; j + 3 < dim; j += 4) {
            float32x4_t prev = vld1q_f32(input.output + j);
            prev = vmulq_f32(prev, v_alpha);
            float vv[4];
            for (int k = 0; k < 4; k++) {
                uint32_t raw = nfa_extract_bits(v_packed, j + k, input.val_bits);
                int g = (j + k) / input.val_group_size;
                vv[k] = (float)raw * input.val_scales[n * val_n_groups + g]
                        + input.val_zeros[n * val_n_groups + g];
            }
            prev = vmlaq_f32(prev, v_p, vld1q_f32(vv));
            vst1q_f32(input.output + j, prev);
        }
        for (; j < dim; j++) {
            input.output[j] *= alpha;
            uint32_t raw = nfa_extract_bits(v_packed, j, input.val_bits);
            int g = j / input.val_group_size;
            float val = (float)raw * input.val_scales[n * val_n_groups + g]
                       + input.val_zeros[n * val_n_groups + g];
            input.output[j] += p * val;
        }
        m_i = m_new;
    }

    if (l_i > 0.0f) {
        float32x4_t v_inv = vdupq_n_f32(1.0f / l_i);
        int j = 0;
        for (; j + 3 < dim; j += 4)
            vst1q_f32(input.output + j, vmulq_f32(vld1q_f32(input.output + j), v_inv));
        for (; j < dim; j++)
            input.output[j] /= l_i;
    }
#else
    cpu_scalar_fused_attention(input);
#endif
}

} // namespace tq
