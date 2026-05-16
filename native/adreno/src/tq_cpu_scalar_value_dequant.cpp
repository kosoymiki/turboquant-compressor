/**
 * TurboQuant — CPU scalar value dequantization.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_adreno.h"

namespace tq {

static inline uint32_t vd_extract_bits(const uint8_t* packed, int coord, int bits) {
    uint32_t bit_pos = (uint32_t)coord * (uint32_t)bits;
    uint32_t byte_idx = bit_pos >> 3;
    uint32_t bit_off = bit_pos & 7u;
    uint32_t word = packed[byte_idx];
    if (bit_off + (uint32_t)bits > 8u) word |= ((uint32_t)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & ((1u << bits) - 1u);
}

void cpu_scalar_value_dequant(const ValueDequantInput& input, float* output) {
    int n_groups = (input.dim + input.group_size - 1) / input.group_size;
    int packed_stride = (input.dim * input.bits + 7) / 8;
    for (int n = 0; n < input.tokens; n++) {
        const uint8_t* packed = input.packed_values + n * packed_stride;
        for (int j = 0; j < input.dim; j++) {
            uint32_t raw = vd_extract_bits(packed, j, input.bits);
            int g = j / input.group_size;
            output[n * input.dim + j] = (float)raw * input.scales[n * n_groups + g]
                                        + input.zeros[n * n_groups + g];
        }
    }
}

} // namespace tq
