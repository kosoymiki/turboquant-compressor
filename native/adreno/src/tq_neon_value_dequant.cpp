/**
 * TurboQuant — NEON value dequantization (ARM SIMD accelerated).
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_adreno.h"

#ifdef TQ_HAS_NEON
#include <arm_neon.h>
#endif

namespace tq {

static inline uint32_t nvd_extract_bits(const uint8_t* packed, int coord, int bits) {
    uint32_t bit_pos = (uint32_t)coord * (uint32_t)bits;
    uint32_t byte_idx = bit_pos >> 3;
    uint32_t bit_off = bit_pos & 7u;
    uint32_t word = packed[byte_idx];
    if (bit_off + (uint32_t)bits > 8u) word |= ((uint32_t)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & ((1u << bits) - 1u);
}

void neon_value_dequant(const ValueDequantInput& input, float* output) {
#ifdef TQ_HAS_NEON
    int n_groups = (input.dim + input.group_size - 1) / input.group_size;
    int packed_stride = (input.dim * input.bits + 7) / 8;

    for (int n = 0; n < input.tokens; n++) {
        const uint8_t* packed = input.packed_values + n * packed_stride;
        float* out = output + n * input.dim;

        for (int g = 0; g < n_groups; g++) {
            float scale = input.scales[n * n_groups + g];
            float zero = input.zeros[n * n_groups + g];
            float32x4_t v_scale = vdupq_n_f32(scale);
            float32x4_t v_zero = vdupq_n_f32(zero);

            int j_start = g * input.group_size;
            int j_end = j_start + input.group_size;
            if (j_end > input.dim) j_end = input.dim;

            int j = j_start;
            for (; j + 3 < j_end; j += 4) {
                float raw_vals[4];
                for (int k = 0; k < 4; k++)
                    raw_vals[k] = (float)nvd_extract_bits(packed, j + k, input.bits);
                float32x4_t raw = vld1q_f32(raw_vals);
                float32x4_t result = vmlaq_f32(v_zero, raw, v_scale);
                vst1q_f32(out + j, result);
            }
            for (; j < j_end; j++) {
                uint32_t raw = nvd_extract_bits(packed, j, input.bits);
                out[j] = (float)raw * scale + zero;
            }
        }
    }
#else
    cpu_scalar_value_dequant(input, output);
#endif
}

} // namespace tq
