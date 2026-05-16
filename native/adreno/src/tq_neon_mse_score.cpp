/**
 * TurboQuant — NEON MSE score (ARM SIMD accelerated).
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_adreno.h"

#ifdef TQ_HAS_NEON
#include <arm_neon.h>
#endif

namespace tq {

static inline uint32_t neon_extract_bits(const uint8_t* packed, int coord, int bits) {
    uint32_t bit_pos = (uint32_t)coord * (uint32_t)bits;
    uint32_t byte_idx = bit_pos >> 3;
    uint32_t bit_off = bit_pos & 7u;
    uint32_t word = packed[byte_idx];
    if (bit_off + (uint32_t)bits > 8u) word |= ((uint32_t)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & ((1u << bits) - 1u);
}

void neon_mse_score(const MseInput& input, float* scores) {
#ifdef TQ_HAS_NEON
    for (int n = 0; n < input.tokens; n++) {
        const uint8_t* packed = input.packed_indices + n * input.packed_stride;
        float32x4_t acc = vdupq_n_f32(0.0f);
        int j = 0;

        // Process 4 coords at a time with NEON
        for (; j + 3 < input.dim; j += 4) {
            float centroid_vals[4];
            for (int k = 0; k < 4; k++) {
                uint32_t idx = neon_extract_bits(packed, j + k, input.bits);
                centroid_vals[k] = input.centroids[idx];
            }
            float32x4_t c = vld1q_f32(centroid_vals);
            float32x4_t q = vld1q_f32(input.query_rotated + j);
            acc = vmlaq_f32(acc, q, c);
        }

        // Horizontal sum
        float dot = vaddvq_f32(acc);

        // Scalar tail
        for (; j < input.dim; j++) {
            uint32_t idx = neon_extract_bits(packed, j, input.bits);
            dot += input.query_rotated[j] * input.centroids[idx];
        }

        scores[n] = dot * input.norms[n];
    }
#else
    cpu_scalar_mse_score(input, scores);
#endif
}

} // namespace tq
