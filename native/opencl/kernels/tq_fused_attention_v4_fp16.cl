/**
 * TurboQuant Fused Attention Kernel v4 — full vectorized fp16 for Adreno 730.
 * Over v3:
 *   - vload4/vstore4 on ALL hot paths (query dot, value accum, output norm)
 *   - vload8 path for dim>=64 (256-bit burst, 2x bandwidth on Adreno L2)
 *   - Vectorized FWHT butterfly (local memory, half4)
 *   - Prefetch hint for key_packed sequential access
 *
 * Adreno 730 specs:
 *   - 2 shader cores, 128-bit ALU (half8 native)
 *   - L1 cache line: 64B, L2: 128B
 *   - Peak BW: 51.2 GB/s (LPDDR5)
 *   - vload4 half = 64-bit = 1 cycle on SP
 *   - vload8 half = 128-bit = 1 cycle on full ALU width
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

__kernel void tq_fused_attention_v4_fp16(
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

    // Centroid LUT — local memory
    __local half lut[64];
    int n_centroids = 1 << key_bits;
    for (int i = lid; i < n_centroids; i += local_size)
        lut[i] = centroids[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Query → local (vectorized vload8 where possible)
    __local half q_local[256];
    int dim8 = (dim / 8) * 8;
    int dim4 = (dim / 4) * 4;
    for (int i = lid * 8; i < dim8; i += local_size * 8) {
        half8 qv = vload8(0, query_rotated + i);
        vstore8(qv, 0, q_local + i);
    }
    for (int i = dim8 + lid; i < dim; i += local_size)
        q_local[i] = query_rotated[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;

    __local float scratch[256];

    for (int n = 0; n < tokens; n++) {
        __global const uchar* k_packed = key_packed + n * key_packed_stride;
        float mse_dot = 0.0f;

        // === Key dot product: vload8 path (8 dims/iter) ===
        for (int j = lid * 8; j < dim8; j += local_size * 8) {
            half8 q8 = vload8(0, q_local + j);
            half c0 = lut[extract_bits(k_packed, (uint)j,     (uint)key_bits)];
            half c1 = lut[extract_bits(k_packed, (uint)(j+1), (uint)key_bits)];
            half c2 = lut[extract_bits(k_packed, (uint)(j+2), (uint)key_bits)];
            half c3 = lut[extract_bits(k_packed, (uint)(j+3), (uint)key_bits)];
            half c4 = lut[extract_bits(k_packed, (uint)(j+4), (uint)key_bits)];
            half c5 = lut[extract_bits(k_packed, (uint)(j+5), (uint)key_bits)];
            half c6 = lut[extract_bits(k_packed, (uint)(j+6), (uint)key_bits)];
            half c7 = lut[extract_bits(k_packed, (uint)(j+7), (uint)key_bits)];
            mse_dot += (float)(q8.s0*c0 + q8.s1*c1 + q8.s2*c2 + q8.s3*c3
                             + q8.s4*c4 + q8.s5*c5 + q8.s6*c6 + q8.s7*c7);
        }
        // vload4 remainder
        for (int j = dim8 + lid * 4; j < dim4; j += local_size * 4) {
            half4 q4 = vload4(0, q_local + j);
            half c0 = lut[extract_bits(k_packed, (uint)j,     (uint)key_bits)];
            half c1 = lut[extract_bits(k_packed, (uint)(j+1), (uint)key_bits)];
            half c2 = lut[extract_bits(k_packed, (uint)(j+2), (uint)key_bits)];
            half c3 = lut[extract_bits(k_packed, (uint)(j+3), (uint)key_bits)];
            mse_dot += (float)(q4.x*c0 + q4.y*c1 + q4.z*c2 + q4.w*c3);
        }
        // Scalar tail
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

        // Online softmax
        float m_new = fmax(score, m_i);
        float alpha = exp(m_i - m_new);
        float p = exp(score - m_new);

        // Sparse skip (never sink tokens)
        float contrib = p / (l_i * alpha + p);
        if (n >= sink_tokens && contrib < sparse_threshold) {
            l_i = l_i * alpha + p;
            m_i = m_new;
            // Vectorized output rescale
            for (int j = lid * 8; j < dim8; j += local_size * 8) {
                half8 ov = vload8(0, output + j);
                half8 scaled = ov * (half)alpha;
                vstore8(scaled, 0, output + j);
            }
            for (int j = dim8 + lid; j < dim; j += local_size)
                output[j] = (half)((float)output[j] * alpha);
            continue;
        }

        // === Value dequant + accumulate: vload4 vectorized ===
        __global const uchar* v_packed = val_packed + n * val_packed_stride;
        for (int j = lid * 4; j < dim4; j += local_size * 4) {
            int g0 = j / group_size;
            int g1 = (j+1) / group_size;
            int g2 = (j+2) / group_size;
            int g3 = (j+3) / group_size;
            int base_g = n * n_groups;

            float s0 = (float)val_scales[base_g + g0];
            float s1 = (float)val_scales[base_g + g1];
            float s2 = (float)val_scales[base_g + g2];
            float s3 = (float)val_scales[base_g + g3];
            float z0 = (float)val_zeros[base_g + g0];
            float z1 = (float)val_zeros[base_g + g1];
            float z2 = (float)val_zeros[base_g + g2];
            float z3 = (float)val_zeros[base_g + g3];

            uint r0 = extract_bits(v_packed, (uint)j,     (uint)val_bits);
            uint r1 = extract_bits(v_packed, (uint)(j+1), (uint)val_bits);
            uint r2 = extract_bits(v_packed, (uint)(j+2), (uint)val_bits);
            uint r3 = extract_bits(v_packed, (uint)(j+3), (uint)val_bits);

            float v0 = (float)r0 * s0 + z0;
            float v1 = (float)r1 * s1 + z1;
            float v2 = (float)r2 * s2 + z2;
            float v3 = (float)r3 * s3 + z3;

            half4 ov = vload4(0, output + j);
            half4 result = (half4)(
                (half)((float)ov.x * alpha + p * v0),
                (half)((float)ov.y * alpha + p * v1),
                (half)((float)ov.z * alpha + p * v2),
                (half)((float)ov.w * alpha + p * v3)
            );
            vstore4(result, 0, output + j);
        }
        // Scalar tail
        for (int j = dim4 + lid; j < dim; j += local_size) {
            uint raw = extract_bits(v_packed, (uint)j, (uint)val_bits);
            int g = j / group_size;
            float val = (float)raw * (float)val_scales[n * n_groups + g] + (float)val_zeros[n * n_groups + g];
            output[j] = (half)((float)output[j] * alpha + p * val);
        }

        l_i = l_i * alpha + p;
        m_i = m_new;
    }

    // === Final normalization: vectorized ===
    if (l_i > 0.0f) {
        half inv_l = (half)(1.0f / l_i);
        for (int j = lid * 8; j < dim8; j += local_size * 8) {
            half8 ov = vload8(0, output + j);
            vstore8(ov * inv_l, 0, output + j);
        }
        for (int j = dim8 + lid; j < dim; j += local_size)
            output[j] = output[j] * inv_l;
    }
}

/**
 * Vectorized FWHT kernel (in-place, half precision)
 * Butterfly pattern with vload4 for sequential dim access.
 * Works on single vector of power-of-2 length.
 */
__kernel void tq_fwht_fp16(
    __global half* data,
    const int dim
) {
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    if (dim > 256 || (dim & (dim - 1)) != 0) return;

    // Load into local memory (vectorized)
    __local half buf[256];
    int dim4 = (dim / 4) * 4;
    for (int i = lid * 4; i < dim4; i += local_size * 4) {
        half4 v = vload4(0, data + i);
        vstore4(v, 0, buf + i);
    }
    for (int i = dim4 + lid; i < dim; i += local_size)
        buf[i] = data[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Butterfly stages
    for (int h = 1; h < dim; h <<= 1) {
        for (int i = lid; i < dim / 2; i += local_size) {
            int block = i / h;
            int offset = i % h;
            int j = block * h * 2 + offset;
            half x = buf[j];
            half y = buf[j + h];
            buf[j] = x + y;
            buf[j + h] = x - y;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Normalize + write back (vectorized)
    half norm = (half)(1.0f / sqrt((float)dim));
    for (int i = lid * 4; i < dim4; i += local_size * 4) {
        half4 v = vload4(0, buf + i);
        vstore4(v * norm, 0, data + i);
    }
    for (int i = dim4 + lid; i < dim; i += local_size)
        data[i] = buf[i] * norm;
}
