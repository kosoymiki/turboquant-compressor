/**
 * TurboQuant QJL Score Kernel — popcount-based sign dot correction.
 * corrected[n] = base_scores[n] + qjl_scale * residual_norms[n] * sign_dot(query_signs, residual_signs[n])
 * sign_dot = projections - 2 * hamming_distance
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifdef TQ_HAS_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half tq_fp16_t;
typedef half4 tq_fp16_vec_t;
#define TQ_FP16_AVAILABLE 1
#else
typedef float tq_fp16_t;
typedef float4 tq_fp16_vec_t;
#define TQ_FP16_AVAILABLE 0
#endif

#ifdef USE_SUBGROUPS
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

__kernel void tq_qjl_score(
    __global const uint* query_signs,      // [proj_words]
    __global const uint* residual_signs,   // [tokens * proj_words]
    __global const float* residual_norms,  // [tokens]
    __global float* scores,                // [tokens] — in/out, adds correction
    const int tokens,
    const int projections,
    const float qjl_scale,
    const int proj_words                   // ceil(projections / 32)
) {
    int tid = get_global_id(0);
    if (tid >= tokens) return;

    __global const uint* my_signs = residual_signs + tid * proj_words;

    // XOR + popcount = hamming distance
    uint hamming = 0;
    for (int w = 0; w < proj_words; w++) {
        uint xor_val = query_signs[w] ^ my_signs[w];
        hamming += popcount(xor_val);
    }

    // sign_dot = projections - 2 * hamming_distance
    float sign_dot = (float)projections - 2.0f * (float)hamming;

    // Correction
    scores[tid] += qjl_scale * residual_norms[tid] * sign_dot;
}

// Tiled variant with local reduction (for large proj_words)
__kernel void tq_qjl_score_tiled(
    __global const uint* query_signs,
    __global const uint* residual_signs,
    __global const float* residual_norms,
    __global float* scores,
    const int tokens,
    const int projections,
    const float qjl_scale,
    const int proj_words,
    __local uint* scratch
) {
    int token_id = get_group_id(0);
    if (token_id >= tokens) return;

    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    __global const uint* my_signs = residual_signs + token_id * proj_words;

    uint partial_hamming = 0;
    for (int w = lid; w < proj_words; w += local_size) {
        uint xor_val = query_signs[w] ^ my_signs[w];
        partial_hamming += popcount(xor_val);
    }

    scratch[lid] = partial_hamming;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = local_size >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        float sign_dot = (float)projections - 2.0f * (float)scratch[0];
        scores[token_id] += qjl_scale * residual_norms[token_id] * sign_dot;
    }
}
