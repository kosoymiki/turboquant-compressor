/**
 * TurboQuant Tiled Attention — compute logits (MSE + QJL) in parallel.
 * One work-item per token. Output: scores[tokens].
 * This is the same as running mse_score + qjl_score but fused into one kernel.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

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
    if (key_bits == 4) {
        for (int j = 0; j < dim; j++)
            dot += query_rotated[j] * centroids[logits_extract_4bit(my_packed, (uint)j)];
    } else if (key_bits == 3) {
        for (int j = 0; j < dim; j++)
            dot += query_rotated[j] * centroids[logits_extract_3bit(my_packed, (uint)j)];
    } else {
        for (int j = 0; j < dim; j++)
            dot += query_rotated[j] * centroids[logits_extract_2bit(my_packed, (uint)j)];
    }
    float score = dot * key_norms[tid];

    // QJL correction
    __global const uint* my_signs = residual_signs + tid * proj_words;
    uint hamming = 0;
    for (int w = 0; w < proj_words; w++)
        hamming += popcount(query_signs[w] ^ my_signs[w]);
    float sign_dot = (float)projections - 2.0f * (float)hamming;
    score += qjl_scale * residual_norms[tid] * sign_dot;

    scores[tid] = score * sm_scale;
}
