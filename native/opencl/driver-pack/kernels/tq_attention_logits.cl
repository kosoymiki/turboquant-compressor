/**
 * TurboQuant Tiled Attention — compute logits (MSE + QJL) in parallel.
 * One work-item per token. Output: scores[tokens].
 * This is the same as running mse_score + qjl_score but fused into one kernel.
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

inline char logits_sign_char_from_word_bit(uint word, uint bit) {
    return (word & (1u << bit)) ? (char)-1 : (char)1;
}

#ifdef TQ_HAS_INTEGER_DOT_PRODUCT
inline int logits_sign_dot_word_intdot(uint query_word, uint residual_word, uint valid_bits) {
    int total = 0;
    uint bit = 0;
    for (; bit + 3u < valid_bits; bit += 4u) {
        char4 q = (char4)(
            logits_sign_char_from_word_bit(query_word, bit + 0u),
            logits_sign_char_from_word_bit(query_word, bit + 1u),
            logits_sign_char_from_word_bit(query_word, bit + 2u),
            logits_sign_char_from_word_bit(query_word, bit + 3u));
        char4 r = (char4)(
            logits_sign_char_from_word_bit(residual_word, bit + 0u),
            logits_sign_char_from_word_bit(residual_word, bit + 1u),
            logits_sign_char_from_word_bit(residual_word, bit + 2u),
            logits_sign_char_from_word_bit(residual_word, bit + 3u));
        total += dot(q, r);
    }
    for (; bit < valid_bits; ++bit) {
        total += (int)logits_sign_char_from_word_bit(query_word, bit) *
                 (int)logits_sign_char_from_word_bit(residual_word, bit);
    }
    return total;
}
#endif

inline int logits_sign_dot_word(uint query_word, uint residual_word, uint valid_bits) {
#ifdef TQ_HAS_INTEGER_DOT_PRODUCT
    return logits_sign_dot_word_intdot(query_word, residual_word, valid_bits);
#else
    uint mask = valid_bits >= 32u ? 0xFFFFFFFFu : ((1u << valid_bits) - 1u);
    uint hamming = popcount((query_word ^ residual_word) & mask);
    return (int)valid_bits - 2 * (int)hamming;
#endif
}

inline uint logits_extract_4bit(__global const uchar* packed, uint coord) {
    uint byte_idx = coord >> 1;
    uint shift = (coord & 1u) * 4u;
    return ((uint)packed[byte_idx] >> shift) & 0xFu;
}

inline uint logits_extract_3bit(__global const uchar* packed, uint coord) {
    uint bit_pos = coord * 3u;
    uint byte_idx = bit_pos >> 3;
    uint bit_off = bit_pos & 7u;
    uint word = (uint)packed[byte_idx];
    if (bit_off + 3 > 8) word |= ((uint)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & 0x7u;
}

inline uint logits_extract_2bit(__global const uchar* packed, uint coord) {
    uint byte_idx = coord >> 2;
    uint shift = (coord & 3u) * 2u;
    return ((uint)packed[byte_idx] >> shift) & 0x3u;
}

inline float logits_dot_chunk4_4bit(
    __global const float* query_rotated,
    __global const uchar* packed,
    __constant const float* centroids,
    int base
) {
    float4 q = vload4(0, query_rotated + base);
    float4 c = (float4)(
        centroids[logits_extract_4bit(packed, (uint)(base + 0))],
        centroids[logits_extract_4bit(packed, (uint)(base + 1))],
        centroids[logits_extract_4bit(packed, (uint)(base + 2))],
        centroids[logits_extract_4bit(packed, (uint)(base + 3))]);
    return dot(q, c);
}

inline float logits_dot_chunk4_3bit(
    __global const float* query_rotated,
    __global const uchar* packed,
    __constant const float* centroids,
    int base
) {
    float4 q = vload4(0, query_rotated + base);
    float4 c = (float4)(
        centroids[logits_extract_3bit(packed, (uint)(base + 0))],
        centroids[logits_extract_3bit(packed, (uint)(base + 1))],
        centroids[logits_extract_3bit(packed, (uint)(base + 2))],
        centroids[logits_extract_3bit(packed, (uint)(base + 3))]);
    return dot(q, c);
}

inline float logits_dot_chunk4_2bit(
    __global const float* query_rotated,
    __global const uchar* packed,
    __constant const float* centroids,
    int base
) {
    float4 q = vload4(0, query_rotated + base);
    float4 c = (float4)(
        centroids[logits_extract_2bit(packed, (uint)(base + 0))],
        centroids[logits_extract_2bit(packed, (uint)(base + 1))],
        centroids[logits_extract_2bit(packed, (uint)(base + 2))],
        centroids[logits_extract_2bit(packed, (uint)(base + 3))]);
    return dot(q, c);
}

__kernel void tq_attention_logits(
    __global const float* query_rotated,     // [dim]
    __global const uchar* key_packed,        // [tokens * key_packed_stride]
    __global const float* key_norms,         // [tokens]
    __constant const float* centroids,       // [2^key_bits]
    __global const uint* query_signs,        // [proj_words]
    __global const uint* residual_signs,     // [tokens * proj_words]
    __global const float* residual_norms,    // [tokens]
    __global float* scores,                  // [tokens] output
    const int tokens,
    const int dim,
    const int key_bits,
    const int key_packed_stride,
    const int projections,
    const int proj_words,
    const float qjl_scale,
    const float sm_scale
) {
    int tid = get_global_id(0);
    if (tid >= tokens) return;

    // MSE score
    __global const uchar* my_packed = key_packed + tid * key_packed_stride;
    float dot = 0.0f;
    int j = 0;
    if (key_bits == 4) {
        #pragma unroll 4
        for (; j + 3 < dim; j += 4)
            dot += logits_dot_chunk4_4bit(query_rotated, my_packed, centroids, j);
        for (; j < dim; ++j)
            dot += query_rotated[j] * centroids[logits_extract_4bit(my_packed, (uint)j)];
    } else if (key_bits == 3) {
        #pragma unroll 4
        for (; j + 3 < dim; j += 4)
            dot += logits_dot_chunk4_3bit(query_rotated, my_packed, centroids, j);
        for (; j < dim; ++j)
            dot += query_rotated[j] * centroids[logits_extract_3bit(my_packed, (uint)j)];
    } else {
        #pragma unroll 4
        for (; j + 3 < dim; j += 4)
            dot += logits_dot_chunk4_2bit(query_rotated, my_packed, centroids, j);
        for (; j < dim; ++j)
            dot += query_rotated[j] * centroids[logits_extract_2bit(my_packed, (uint)j)];
    }
    float score = dot * key_norms[tid];

    // QJL correction
    __global const uint* my_signs = residual_signs + tid * proj_words;
    int sign_dot_i = 0;
    for (int w = 0; w < proj_words; w++) {
        const int bit_base = w * 32;
        const uint valid_bits = (uint)max(0, min(32, projections - bit_base));
        if (valid_bits == 0u) break;
        sign_dot_i += logits_sign_dot_word(query_signs[w], my_signs[w], valid_bits);
    }
    float sign_dot = (float)sign_dot_i;
    score += qjl_scale * residual_norms[tid] * sign_dot;

    scores[tid] = score * sm_scale;
}
