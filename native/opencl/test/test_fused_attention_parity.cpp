#include "../include/tq_opencl_types.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cstring>

namespace tq { void fused_attention_cpu_reference(const FusedAttentionInput& input); }

static void pack_3bit(uint8_t* packed, int coord, uint8_t val) {
    uint32_t bit_pos = (uint32_t)coord * 3u;
    uint32_t byte_idx = bit_pos >> 3;
    uint32_t bit_off = bit_pos & 7u;
    packed[byte_idx] |= (val & 0x7) << bit_off;
    if (bit_off + 3 > 8) packed[byte_idx + 1] |= val >> (8 - bit_off);
}

static void pack_4bit(uint8_t* packed, int coord, uint8_t val) {
    int byte_idx = coord >> 1;
    int shift = (coord & 1) * 4;
    packed[byte_idx] |= (val & 0xF) << shift;
}

int main() {
    const int tokens = 2, dim = 4, key_bits = 3, val_bits = 4;
    const int group_size = 4, n_groups = 1, projections = 32, proj_words = 1;
    int key_packed_stride = (dim * key_bits + 7) / 8;  // 2 bytes
    int val_packed_stride = (dim * val_bits + 7) / 8;  // 2 bytes

    // Key data
    float centroids[8] = {-0.9f, -0.6f, -0.3f, 0.0f, 0.2f, 0.4f, 0.7f, 1.0f};
    float query_rotated[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    float key_norms[2] = {1.0f, 1.0f};

    std::vector<uint8_t> key_packed(tokens * key_packed_stride, 0);
    // Token 0: all idx=6 (centroid=0.7)
    for (int j = 0; j < dim; j++) pack_3bit(key_packed.data(), j, 6);
    // Token 1: all idx=1 (centroid=-0.6)
    for (int j = 0; j < dim; j++) pack_3bit(key_packed.data() + key_packed_stride, j, 1);

    // QJL data — no correction (all same signs)
    uint32_t query_signs[1] = {0x0u};
    uint32_t residual_signs[2] = {0x0u, 0x0u};
    float residual_norms[2] = {0.0f, 0.0f};

    // Value data
    std::vector<uint8_t> val_packed(tokens * val_packed_stride, 0);
    float val_scales[2] = {0.1f, 0.1f};
    float val_zeros[2] = {0.0f, 0.0f};
    // Token 0 values: [8, 8, 8, 8] -> dequant = 0.8 each
    for (int j = 0; j < dim; j++) pack_4bit(val_packed.data(), j, 8);
    // Token 1 values: [2, 2, 2, 2] -> dequant = 0.2 each
    for (int j = 0; j < dim; j++) pack_4bit(val_packed.data() + val_packed_stride, j, 2);

    float sm_scale = 1.0f;
    float qjl_scale = 1.0f;
    float output[4] = {};

    tq::MseScoreInput mse;
    mse.packed_indices = key_packed.data();
    mse.norms = key_norms;
    mse.query_rotated = query_rotated;
    mse.centroids = centroids;
    mse.tokens = tokens;
    mse.dim = dim;
    mse.bits = key_bits;
    mse.packed_stride = key_packed_stride;

    tq::QjlScoreInput qjl;
    qjl.query_signs = query_signs;
    qjl.residual_signs = residual_signs;
    qjl.residual_norms = residual_norms;
    qjl.base_scores = nullptr; // set by fused
    qjl.tokens = tokens;
    qjl.projections = projections;
    qjl.qjl_scale = qjl_scale;

    tq::ValueDequantInput val;
    val.packed_values = val_packed.data();
    val.scales = val_scales;
    val.zeros = val_zeros;
    val.tokens = tokens;
    val.dim = dim;
    val.bits = val_bits;
    val.group_size = group_size;

    tq::FusedAttentionInput input;
    input.mse = mse;
    input.qjl = qjl;
    input.value = val;
    input.sm_scale = sm_scale;
    input.output = output;

    tq::fused_attention_cpu_reference(input);

    // MSE scores:
    // Token 0: dot = 0.5*0.7*4 = 1.4, score = 1.4*1.0 = 1.4
    // Token 1: dot = 0.5*(-0.6)*4 = -1.2, score = -1.2*1.0 = -1.2
    // QJL correction: 0 (residual_norms=0)
    // After sm_scale=1: scores = [1.4, -1.2]
    // Softmax: exp(1.4)/(exp(1.4)+exp(-1.2)) ≈ 4.055/(4.055+0.301) ≈ 0.931
    //          exp(-1.2)/(exp(1.4)+exp(-1.2)) ≈ 0.301/4.356 ≈ 0.069
    // Output ≈ 0.931*0.8 + 0.069*0.2 ≈ 0.745 + 0.014 ≈ 0.759 per dim

    int pass = 0, fail = 0;
    float expected_approx = 0.759f;
    for (int j = 0; j < dim; j++) {
        float err = fabsf(output[j] - expected_approx);
        if (err < 0.01f) { pass++; }
        else {
            printf("  FAIL [%d]: got %.6f expected ~%.3f (err=%.6f)\n", j, output[j], expected_approx, err);
            fail++;
        }
    }

    // Also verify output is finite and consistent
    bool all_same = true;
    for (int j = 1; j < dim; j++) {
        if (fabsf(output[j] - output[0]) > 1e-6f) all_same = false;
    }
    if (all_same && pass > 0) {
        printf("  All dims consistent: %.6f\n", output[0]);
    }

    printf("Fused attention parity: %d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
