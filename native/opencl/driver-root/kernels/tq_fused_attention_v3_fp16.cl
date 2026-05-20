/**
 * TurboQuant Fused Attention Kernel v3 — fp16 + vectorized for Adreno 730.
 * Improvements over v2:
 *   - cl_khr_fp16: 2x throughput on Adreno ALU
 *   - vload4/vstore4: 128-bit memory path
 *   - Attention-sink protection (first N tokens never sparse-skipped)
 *   - Guards for dim/bits overflow
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline uint extract_bits(__global const uchar* packed, uint coord, uint bits) {
    uint bit_pos = coord * bits;
    uint byte_idx = bit_pos >> 3;
    uint bit_off = bit_pos & 7u;
    uint word = (uint)packed[byte_idx];
    if (bit_off + bits > 8) word |= ((uint)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & ((1u << bits) - 1u);
}

__kernel void tq_fused_attention_v3_fp16(
    __global const half* query_rotated,
    __global const uchar* key_packed,
    __global const half* key_norms,
    __constant const half* centroids,
    __global const uchar* val_packed,
    __global const half* val_scales,
    __global const half* val_zeros,
    __global half* output,
    const int tokens,
    const int dim,
    const int key_bits,
    const int key_packed_stride,
    const int val_bits,
    const int val_packed_stride,
    const int group_size,
    const int n_groups,
    const float sm_scale,
    const float sparse_threshold,
    const int sink_tokens
) {
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    if (dim > 256 || key_bits > 6) return;

    // Centroid LUT in local memory (fp16)
    __local half lut[64];
    int n_centroids = 1 << key_bits;
    for (int i = lid; i < n_centroids; i += local_size)
        lut[i] = centroids[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Query cache in local memory (fp16)
    __local half q_local[256];
    for (int i = lid; i < dim; i += local_size)
        q_local[i] = query_rotated[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Online softmax
    float m_i = -INFINITY;
    float l_i = 0.0f;

    __local float scratch[256];

    for (int n = 0; n < tokens; n++) {
        // Score: dot(q, dequant(k)) * norm
        __global const uchar* k_packed = key_packed + n * key_packed_stride;
        float mse_dot = 0.0f;

        // Vectorized: process 4 dims per iteration where possible
        int dim4 = (dim / 4) * 4;
        for (int j = lid * 4; j < dim4; j += local_size * 4) {
            half4 q4 = vload4(0, q_local + j);
            half c0 = lut[extract_bits(k_packed, (uint)j, (uint)key_bits)];
            half c1 = lut[extract_bits(k_packed, (uint)(j+1), (uint)key_bits)];
            half c2 = lut[extract_bits(k_packed, (uint)(j+2), (uint)key_bits)];
            half c3 = lut[extract_bits(k_packed, (uint)(j+3), (uint)key_bits)];
            mse_dot += (float)(q4.x * c0 + q4.y * c1 + q4.z * c2 + q4.w * c3);
        }
        // Remainder
        for (int j = dim4 + lid; j < dim; j += local_size) {
            uint idx = extract_bits(k_packed, (uint)j, (uint)key_bits);
            mse_dot += (float)(q_local[j] * lut[idx]);
        }

        // Workgroup reduction
        scratch[lid] = mse_dot;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = local_size >> 1; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float score = scratch[0] * (float)key_norms[n] * sm_scale;

        // Online softmax update
        float m_new = fmax(score, m_i);
        float alpha = exp(m_i - m_new);
        float p = exp(score - m_new);

        // Sparse V: skip if contribution negligible (but never skip sink tokens)
        float contrib = p / (l_i * alpha + p);
        if (n >= sink_tokens && contrib < sparse_threshold) {
            l_i = l_i * alpha + p;
            m_i = m_new;
            for (int j = lid; j < dim; j += local_size)
                output[j] = (half)((float)output[j] * alpha);
            continue;
        }

        // Dequant value (fp16) and accumulate
        __global const uchar* v_packed = val_packed + n * val_packed_stride;
        for (int j = lid; j < dim; j += local_size) {
            uint raw = extract_bits(v_packed, (uint)j, (uint)val_bits);
            int g = j / group_size;
            float val = (float)raw * (float)val_scales[n * n_groups + g] + (float)val_zeros[n * n_groups + g];
            output[j] = (half)((float)output[j] * alpha + p * val);
        }

        l_i = l_i * alpha + p;
        m_i = m_new;
    }

    // Final normalization
    if (l_i > 0.0f) {
        float inv_l = 1.0f / l_i;
        for (int j = lid; j < dim; j += local_size)
            output[j] = (half)((float)output[j] * inv_l);
    }
}
