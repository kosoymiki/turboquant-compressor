/**
 * MSE score parity test — CPU reference vs expected values.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "../include/tq_opencl_types.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

namespace tq { void mse_score_cpu_reference(const MseScoreInput& input, float* scores); }

static void pack_index(uint8_t* packed, int coord, int bits, uint32_t val) {
    uint32_t bit_pos = (uint32_t)coord * (uint32_t)bits;
    uint32_t byte_idx = bit_pos >> 3;
    uint32_t bit_off = bit_pos & 7u;
    packed[byte_idx] |= (uint8_t)((val & ((1u << bits) - 1u)) << bit_off);
    if (bit_off + (uint32_t)bits > 8u)
        packed[byte_idx + 1] |= (uint8_t)(val >> (8u - bit_off));
}

int main() {
    const int tokens = 3, dim = 4, bits = 3;
    float centroids[8] = {-0.9f, -0.6f, -0.3f, 0.0f, 0.2f, 0.4f, 0.7f, 1.0f};
    float query[4] = {1.0f, 0.5f, -0.5f, -1.0f};
    float norms[3] = {2.0f, 1.0f, 0.5f};
    int packed_stride = (dim * bits + 7) / 8; // 2 bytes

    std::vector<uint8_t> packed(tokens * packed_stride, 0);

    // Token 0: indices [0, 2, 5, 7] -> centroids [-0.9, -0.3, 0.4, 1.0]
    pack_index(packed.data() + 0 * packed_stride, 0, bits, 0);
    pack_index(packed.data() + 0 * packed_stride, 1, bits, 2);
    pack_index(packed.data() + 0 * packed_stride, 2, bits, 5);
    pack_index(packed.data() + 0 * packed_stride, 3, bits, 7);

    // Token 1: indices [7, 7, 7, 7] -> all 1.0
    for (int j = 0; j < dim; j++)
        pack_index(packed.data() + 1 * packed_stride, j, bits, 7);

    // Token 2: indices [3, 3, 3, 3] -> all 0.0
    for (int j = 0; j < dim; j++)
        pack_index(packed.data() + 2 * packed_stride, j, bits, 3);

    // Expected:
    // Token 0: dot = 1.0*(-0.9) + 0.5*(-0.3) + (-0.5)*0.4 + (-1.0)*1.0 = -0.9 -0.15 -0.2 -1.0 = -2.25
    //          score = -2.25 * 2.0 = -4.5
    // Token 1: dot = 1.0*1.0 + 0.5*1.0 + (-0.5)*1.0 + (-1.0)*1.0 = 1.0 + 0.5 - 0.5 - 1.0 = 0.0
    //          score = 0.0 * 1.0 = 0.0
    // Token 2: dot = all centroids 0.0 -> 0.0, score = 0.0
    float expected[3] = {-4.5f, 0.0f, 0.0f};

    tq::MseScoreInput input;
    input.packed_indices = packed.data();
    input.norms = norms;
    input.query_rotated = query;
    input.centroids = centroids;
    input.tokens = tokens;
    input.dim = dim;
    input.bits = bits;
    input.packed_stride = packed_stride;

    float scores[3] = {};
    tq::mse_score_cpu_reference(input, scores);

    int pass = 0, fail = 0;
    for (int i = 0; i < tokens; i++) {
        float err = fabsf(scores[i] - expected[i]);
        if (err < 1e-5f) {
            printf("  PASS token %d: got %.6f expected %.6f\n", i, scores[i], expected[i]);
            pass++;
        } else {
            printf("  FAIL token %d: got %.6f expected %.6f (err=%.6f)\n", i, scores[i], expected[i], err);
            fail++;
        }
    }

    printf("\nMSE parity: %d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
