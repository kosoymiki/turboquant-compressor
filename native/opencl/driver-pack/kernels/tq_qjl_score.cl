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

#ifdef TQ_HAS_INTEGER_DOT_PRODUCT
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

inline char tq_sign_char_from_word_bit(uint word, uint bit) {
    return (word & (1u << bit)) ? (char)-1 : (char)1;
}

#ifdef TQ_HAS_INTEGER_DOT_PRODUCT
inline int tq_sign_dot_word_intdot(uint query_word, uint residual_word, uint valid_bits) {
    int total = 0;
    uint bit = 0;
    for (; bit + 3u < valid_bits; bit += 4u) {
        char4 q = (char4)(
            tq_sign_char_from_word_bit(query_word, bit + 0u),
            tq_sign_char_from_word_bit(query_word, bit + 1u),
            tq_sign_char_from_word_bit(query_word, bit + 2u),
            tq_sign_char_from_word_bit(query_word, bit + 3u));
        char4 r = (char4)(
            tq_sign_char_from_word_bit(residual_word, bit + 0u),
            tq_sign_char_from_word_bit(residual_word, bit + 1u),
            tq_sign_char_from_word_bit(residual_word, bit + 2u),
            tq_sign_char_from_word_bit(residual_word, bit + 3u));
        total += dot(q, r);
    }
    for (; bit < valid_bits; ++bit) {
        total += (int)tq_sign_char_from_word_bit(query_word, bit) * (int)tq_sign_char_from_word_bit(residual_word, bit);
    }
    return total;
}
#endif

inline int tq_sign_dot_word(uint query_word, uint residual_word, uint valid_bits) {
#ifdef TQ_HAS_INTEGER_DOT_PRODUCT
    return tq_sign_dot_word_intdot(query_word, residual_word, valid_bits);
#else
    uint mask = valid_bits >= 32u ? 0xFFFFFFFFu : ((1u << valid_bits) - 1u);
    uint hamming = popcount((query_word ^ residual_word) & mask);
    return (int)valid_bits - 2 * (int)hamming;
#endif
}

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

    int sign_dot = 0;
    for (int w = 0; w < proj_words; w++) {
        const int bit_base = w * 32;
        const uint valid_bits = (uint)max(0, min(32, projections - bit_base));
        if (valid_bits == 0u) break;
        sign_dot += tq_sign_dot_word(query_signs[w], my_signs[w], valid_bits);
    }

    // Correction
    scores[tid] += qjl_scale * residual_norms[tid] * (float)sign_dot;
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
    __local int* scratch
) {
    int token_id = get_group_id(0);
    if (token_id >= tokens) return;

    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    __global const uint* my_signs = residual_signs + token_id * proj_words;

    int partial_sign_dot = 0;
    for (int w = lid; w < proj_words; w += local_size) {
        const int bit_base = w * 32;
        const uint valid_bits = (uint)max(0, min(32, projections - bit_base));
        if (valid_bits == 0u) continue;
        partial_sign_dot += tq_sign_dot_word(query_signs[w], my_signs[w], valid_bits);
    }

#ifdef USE_SUBGROUPS
    int subgroup_sum = sub_group_reduce_add(partial_sign_dot);
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = subgroup_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        int total = 0;
        int subgroup_count = (int)get_num_sub_groups();
        for (int sg = 0; sg < subgroup_count; ++sg) {
            total += (int)scratch[sg];
        }
        scores[token_id] += qjl_scale * residual_norms[token_id] * (float)total;
    }
#else
    scratch[lid] = partial_sign_dot;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = local_size >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        scores[token_id] += qjl_scale * residual_norms[token_id] * (float)((int)scratch[0]);
    }
#endif
}
