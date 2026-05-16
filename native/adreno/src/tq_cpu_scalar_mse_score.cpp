/**
 * TurboQuant — CPU scalar MSE score.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_adreno.h"

namespace tq {

static inline uint32_t extract_bits(const uint8_t* packed, int coord, int bits) {
    uint32_t bit_pos = (uint32_t)coord * (uint32_t)bits;
    uint32_t byte_idx = bit_pos >> 3;
    uint32_t bit_off = bit_pos & 7u;
    uint32_t word = packed[byte_idx];
    if (bit_off + (uint32_t)bits > 8u) word |= ((uint32_t)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & ((1u << bits) - 1u);
}

void cpu_scalar_mse_score(const MseInput& input, float* scores) {
    for (int n = 0; n < input.tokens; n++) {
        const uint8_t* packed = input.packed_indices + n * input.packed_stride;
        float dot = 0.0f;
        for (int j = 0; j < input.dim; j++) {
            uint32_t idx = extract_bits(packed, j, input.bits);
            dot += input.query_rotated[j] * input.centroids[idx];
        }
        scores[n] = dot * input.norms[n];
    }
}

} // namespace tq
