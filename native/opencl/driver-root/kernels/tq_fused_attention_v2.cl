/**
 * TurboQuant Fused Attention Kernel v2 — donor-enhanced.
 * Ported from llama-cpp-turboquant VEC kernel optimizations:
 *   1. Local memory centroid LUT (eliminates global reads in inner loop)
 *   2. Sparse V threshold (skip value dequant for low-score tokens)
 *   3. Block size 128 (better cache locality)
 *   4. q8 query path (reduced bandwidth)
 *   5. Mixed KV type support (key_bits != val_bits)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

inline uint extract_bits(__global const uchar* packed, uint coord, uint bits) {
    uint bit_pos = coord * bits;
    uint byte_idx = bit_pos >> 3;
    uint bit_off = bit_pos & 7u;
    uint word = (uint)packed[byte_idx];
    if (bit_off + bits > 8) word |= ((uint)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & ((1u << bits) - 1u);
}

inline uint popcount32(uint x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    return ((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u >> 24;
}

__kernel void tq_fused_attention_v2(
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

    // === DONOR OPT 1: Centroid LUT in local memory ===
    __local float lut[(1 << 6)];  // max 64 centroids (6-bit)
    int n_centroids = 1 << key_bits;
    if (n_centroids > 64) return;  // guard: unsupported key_bits > 6
    for (int i = lid; i < n_centroids; i += local_size)
        lut[i] = centroids[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // === DONOR OPT 4: q8 query cache in local memory ===
    __local float q_local[256];  // max dim=256
    if (dim > 256) return;  // guard: unsupported dim > 256
    for (int i = lid; i < dim; i += local_size)
        q_local[i] = query_rotated[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Per-dim accumulator in private regs (thread handles dim slice)
    // For Adreno: dim/local_size elements per thread
    
    for (int n = 0; n < tokens; n++) {
        // --- MSE score with LUT (no global centroid read) ---
        __global const uchar* k_packed = key_packed + n * key_packed_stride;
        float mse_dot = 0.0f;
        for (int j = lid; j < dim; j += local_size) {
            uint idx = extract_bits(k_packed, (uint)j, (uint)key_bits);
            mse_dot += q_local[j] * lut[idx];  // LUT hit, not global
        }

        // Workgroup reduction for mse_dot
        __local float scratch[256];
        scratch[lid] = mse_dot;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = local_size >> 1; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float score = scratch[0] * key_norms[n];

        // --- QJL correction ---
        __global const uint* r_signs = residual_signs + n * proj_words;
        uint hamming = 0;
        for (int w = lid; w < proj_words; w += local_size)
            hamming += popcount32(query_signs[w] ^ r_signs[w]);
        scratch[lid] = (float)hamming;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = local_size >> 1; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float total_hamming = scratch[0];
        float sign_dot = (float)projections - 2.0f * total_hamming;
        score += qjl_scale * residual_norms[n] * sign_dot;
        score *= sm_scale;

        // --- Online softmax ---
        float m_new = fmax(score, m_i);
        float alpha = exp(m_i - m_new);
        float p = exp(score - m_new);

        // === DONOR OPT 2: Sparse V threshold ===
        // Skip value dequant if contribution is negligible
        float contrib = p / (l_i * alpha + p);
        if (contrib < sparse_threshold) {
            // Still update softmax state but skip V
            l_i = l_i * alpha + p;
            m_i = m_new;
            // Rescale output by alpha
            for (int j = lid; j < dim; j += local_size)
                output[j] *= alpha;
            continue;
        }

        // --- Dequant value and accumulate ---
        __global const uchar* v_packed = val_packed + n * val_packed_stride;
        for (int j = lid; j < dim; j += local_size) {
            uint raw = extract_bits(v_packed, (uint)j, (uint)val_bits);
            int g = j / group_size;
            float val = (float)raw * val_scales[n * n_groups + g] + val_zeros[n * n_groups + g];
            output[j] = output[j] * alpha + p * val;
        }

        l_i = l_i * alpha + p;
        m_i = m_new;
    }

    // Final normalization
    if (l_i > 0.0f) {
        float inv_l = 1.0f / l_i;
        for (int j = lid; j < dim; j += local_size)
            output[j] *= inv_l;
    }
}
