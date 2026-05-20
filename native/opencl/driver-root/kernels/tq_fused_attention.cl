/**
 * TurboQuant Fused Attention Kernel — online softmax over compressed KV.
 * No full K/V dequantization buffer. Streams scores + values in tiles.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// Reuse extraction helpers
inline uint fused_extract_4bit(__global const uchar* packed, uint coord) {
    uint byte_idx = coord >> 1;
    uint shift = (coord & 1u) * 4u;
    return ((uint)packed[byte_idx] >> shift) & 0xFu;
}

inline uint fused_extract_2bit(__global const uchar* packed, uint coord) {
    uint byte_idx = coord >> 2;
    uint shift = (coord & 3u) * 2u;
    return ((uint)packed[byte_idx] >> shift) & 0x3u;
}

inline uint fused_extract_3bit(__global const uchar* packed, uint coord) {
    uint bit_pos = coord * 3u;
    uint byte_idx = bit_pos >> 3;
    uint bit_off = bit_pos & 7u;
    uint word = (uint)packed[byte_idx];
    if (bit_off + 3 > 8) word |= ((uint)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & 0x7u;
}

inline uint fused_popcount32(uint x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    return ((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u >> 24;
}

__kernel void tq_fused_attention(
    __global const float* query_rotated,     // [dim]
    __global const uint* query_signs,        // [proj_words]
    __global const uchar* key_packed,        // [tokens * key_packed_stride]
    __global const float* key_norms,         // [tokens]
    __global const uint* residual_signs,     // [tokens * proj_words]
    __global const float* residual_norms,    // [tokens]
    __constant const float* centroids,       // [2^key_bits]
    __global const uchar* val_packed,        // [tokens * val_packed_stride]
    __global const float* val_scales,        // [tokens * n_groups]
    __global const float* val_zeros,         // [tokens * n_groups]
    __global float* output,                  // [dim]
    const int tokens,
    const int dim,
    const int key_bits,
    const int key_packed_stride,
    const int val_bits,
    const int val_packed_stride,
    const int group_size,
    const int n_groups,
    const int projections,
    const int proj_words,
    const float qjl_scale,
    const float sm_scale
) {
    // Single work-group fused attention with online softmax
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    // Online softmax state
    float m_i = -1e30f;
    float l_i = 0.0f;

    // Per-dimension accumulator (each thread handles a slice)
    // We process tokens sequentially, accumulate per-dim
    // This kernel is designed for small token counts or as reference

    for (int n = 0; n < tokens; n++) {
        // --- Compute MSE score ---
        __global const uchar* k_packed = key_packed + n * key_packed_stride;
        float mse_dot = 0.0f;
        for (int j = lid; j < dim; j += local_size) {
            uint idx;
            if (key_bits == 4) idx = fused_extract_4bit(k_packed, (uint)j);
            else if (key_bits == 3) idx = fused_extract_3bit(k_packed, (uint)j);
            else idx = fused_extract_2bit(k_packed, (uint)j);
            mse_dot += query_rotated[j] * centroids[idx];
        }

        // Reduce mse_dot across work-group (simplified: single thread for now)
        // For production: use local memory reduction
        // This reference version runs with local_size=1
        float score = mse_dot * key_norms[n];

        // --- QJL correction ---
        __global const uint* r_signs = residual_signs + n * proj_words;
        uint hamming = 0;
        for (int w = lid; w < proj_words; w += local_size) {
            hamming += fused_popcount32(query_signs[w] ^ r_signs[w]);
        }
        float sign_dot = (float)projections - 2.0f * (float)hamming;
        score += qjl_scale * residual_norms[n] * sign_dot;
        score *= sm_scale;

        // --- Online softmax update ---
        float m_new = (score > m_i) ? score : m_i;
        float alpha = exp(m_i - m_new);
        float p = exp(score - m_new);
        l_i = l_i * alpha + p;

        // --- Dequant value and accumulate ---
        __global const uchar* v_packed = val_packed + n * val_packed_stride;
        for (int j = lid; j < dim; j += local_size) {
            uint raw;
            if (val_bits == 4) raw = fused_extract_4bit(v_packed, (uint)j);
            else if (val_bits == 3) raw = fused_extract_3bit(v_packed, (uint)j);
            else raw = fused_extract_2bit(v_packed, (uint)j);
            int g = j / group_size;
            float val = (float)raw * val_scales[n * n_groups + g] + val_zeros[n * n_groups + g];

            // Rescale previous accumulation and add new
            output[j] = output[j] * alpha + p * val;
        }

        m_i = m_new;
    }

    // Final normalization
    if (l_i > 0.0f) {
        for (int j = lid; j < dim; j += local_size) {
            output[j] /= l_i;
        }
    }
}
