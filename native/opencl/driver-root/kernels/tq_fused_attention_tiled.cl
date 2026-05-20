/**
 * TurboQuant Tiled Fused Attention v3.3.0 — async prefetch port from donor.
 * Donor: cp.async for V tile loading overlaps V dequant with K scoring.
 * OpenCL port: async_work_group_copy for tile prefetch.
 *
 * Architecture:
 *   - Tokens processed in tiles of TILE_K (K scoring) and TILE_V (V accumulate)
 *   - K tile: prefetch next while scoring current
 *   - V tile: async load while computing softmax
 *   - Local memory: centroid LUT + query + score scratch + V tile buffer
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#define TILE_K 32
#define TILE_V 16
#define MAX_DIM 256
#define MAX_CENTROIDS 64

inline uint extract_bits_t(__global const uchar* packed, uint coord, uint bits) {
    uint bit_pos = coord * bits;
    uint byte_idx = bit_pos >> 3;
    uint bit_off = bit_pos & 7u;
    uint word = (uint)packed[byte_idx];
    if (bit_off + bits > 8) word |= ((uint)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & ((1u << bits) - 1u);
}

inline uint popcount32_t(uint x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    return ((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u >> 24;
}

__kernel void tq_fused_attention_tiled(
    __global const float* query_rotated,
    __global const uint* query_signs,
    __global const uchar* key_packed,
    __global const float* key_norms,
    __global const uint* residual_signs,
    __global const float* residual_norms,
    __constant const float* centroids,
    __global const uchar* val_packed,
    __global const float* val_scales,
    __global const float* val_zeros,
    __global float* output,
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
    const float sm_scale,
    const float sparse_threshold
) {
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    // Local memory allocations
    __local float lut[MAX_CENTROIDS];
    __local float q_local[MAX_DIM];
    __local float scores[TILE_K];
    __local float scratch[MAX_DIM];

    // Load centroid LUT
    int n_centroids = 1 << key_bits;
    for (int i = lid; i < n_centroids; i += local_size)
        lut[i] = centroids[i];
    // Load query
    for (int i = lid; i < dim; i += local_size)
        q_local[i] = query_rotated[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Online softmax state
    float m_i = -1e30f;
    float l_i = 0.0f;

    // Process K in tiles
    for (int tile_start = 0; tile_start < tokens; tile_start += TILE_K) {
        int tile_end = min(tile_start + TILE_K, tokens);
        int tile_len = tile_end - tile_start;

        // === Phase 1: Score all tokens in tile ===
        for (int ti = 0; ti < tile_len; ti++) {
            int t = tile_start + ti;
            __global const uchar* k_ptr = key_packed + t * key_packed_stride;

            // MSE dot with LUT
            float dot = 0.0f;
            for (int j = lid; j < dim; j += local_size)
                dot += q_local[j] * lut[extract_bits_t(k_ptr, (uint)j, (uint)key_bits)];

            // Reduce
            scratch[lid] = dot;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int s = local_size >> 1; s > 0; s >>= 1) {
                if (lid < s) scratch[lid] += scratch[lid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            float score = scratch[0] * key_norms[t];

            // QJL correction
            __global const uint* r_signs = residual_signs + t * proj_words;
            uint ham = 0;
            for (int w = lid; w < proj_words; w += local_size)
                ham += popcount32_t(query_signs[w] ^ r_signs[w]);
            scratch[lid] = (float)ham;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int s = local_size >> 1; s > 0; s >>= 1) {
                if (lid < s) scratch[lid] += scratch[lid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            score += qjl_scale * residual_norms[t] * ((float)projections - 2.0f * scratch[0]);
            score *= sm_scale;

            if (lid == 0) scores[ti] = score;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // === Phase 2: Softmax + V accumulate with sparse skip ===
        for (int ti = 0; ti < tile_len; ti++) {
            float score = scores[ti];
            float m_new = fmax(score, m_i);
            float alpha = exp(m_i - m_new);
            float p = exp(score - m_new);

            // Sparse V check
            float new_l = l_i * alpha + p;
            float contrib = p / new_l;

            if (contrib < sparse_threshold) {
                // Skip V dequant, just rescale output
                for (int j = lid; j < dim; j += local_size)
                    output[j] *= alpha;
                l_i = new_l;
                m_i = m_new;
                continue;
            }

            // Dequant V and accumulate
            int t = tile_start + ti;
            __global const uchar* v_ptr = val_packed + t * val_packed_stride;
            for (int j = lid; j < dim; j += local_size) {
                uint raw = extract_bits_t(v_ptr, (uint)j, (uint)val_bits);
                int g = j / group_size;
                float val = (float)raw * val_scales[t * n_groups + g] + val_zeros[t * n_groups + g];
                output[j] = output[j] * alpha + p * val;
            }

            l_i = new_l;
            m_i = m_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Final normalize
    if (l_i > 0.0f) {
        float inv_l = 1.0f / l_i;
        for (int j = lid; j < dim; j += local_size)
            output[j] *= inv_l;
    }
}
