/**
 * QJL score parity test — CPU reference vs expected values.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "../include/tq_opencl_types.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

namespace tq { void qjl_score_cpu_reference(const QjlScoreInput& input); }

int main() {
    // 2 tokens, 32 projections (1 word each)
    const int tokens = 2, projections = 32;
    int proj_words = 1;

    // query_signs: 0xAAAAAAAA (alternating 1010...)
    uint32_t query_signs[1] = {0xAAAAAAAAu};

    // token 0 residual_signs: same as query -> hamming=0 -> sign_dot=32
    // token 1 residual_signs: inverted -> hamming=32 -> sign_dot=-32
    uint32_t residual_signs[2] = {0xAAAAAAAAu, 0x55555555u};

    float residual_norms[2] = {1.0f, 2.0f};
    float base_scores[2] = {10.0f, 5.0f};
    float qjl_scale = 0.5f;

    // Expected:
    // token 0: 10.0 + 0.5 * 1.0 * 32.0 = 10.0 + 16.0 = 26.0
    // token 1: 5.0 + 0.5 * 2.0 * (-32.0) = 5.0 - 32.0 = -27.0
    float expected[2] = {26.0f, -27.0f};

    tq::QjlScoreInput input;
    input.query_signs = query_signs;
    input.residual_signs = residual_signs;
    input.residual_norms = residual_norms;
    input.base_scores = base_scores;
    input.tokens = tokens;
    input.projections = projections;
    input.qjl_scale = qjl_scale;

    tq::qjl_score_cpu_reference(input);

    int pass = 0, fail = 0;
    for (int i = 0; i < tokens; i++) {
        float err = fabsf(base_scores[i] - expected[i]);
        if (err < 1e-5f) {
            printf("  PASS token %d: got %.6f expected %.6f\n", i, base_scores[i], expected[i]);
            pass++;
        } else {
            printf("  FAIL token %d: got %.6f expected %.6f (err=%.6f)\n", i, base_scores[i], expected[i], err);
            fail++;
        }
    }

    printf("\nQJL parity: %d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
