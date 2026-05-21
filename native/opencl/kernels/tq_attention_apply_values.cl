/**
 * TurboQuant Tiled Attention — softmax + weighted value accumulation.
 * Parallelized over dim: each work-item handles one output dimension.
 * Processes tokens sequentially with online softmax (no full V materialization).
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

inline uint val_tiled_extract_4bit(__global const uchar* packed, uint coord) {
    uint byte_idx = coord >> 1;
    uint shift = (coord & 1u) * 4u;
    return ((uint)packed[byte_idx] >> shift) & 0xFu;
}

inline uint val_tiled_extract_3bit(__global const uchar* packed, uint coord) {
    uint bit_pos = coord * 3u;
    uint byte_idx = bit_pos >> 3;
    uint bit_off = bit_pos & 7u;
    uint word = (uint)packed[byte_idx];
    if (bit_off + 3 > 8) word |= ((uint)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & 0x7u;
}

inline uint val_tiled_extract_2bit(__global const uchar* packed, uint coord) {
    uint byte_idx = coord >> 2;
    uint shift = (coord & 3u) * 2u;
    return ((uint)packed[byte_idx] >> shift) & 0x3u;
}

__kernel void tq_attention_apply_values(
    __global const float* scores,            // [tokens] — pre-computed logits (scaled)
    __global const uchar* val_packed,        // [tokens * val_packed_stride]
    __global const float* val_scales,        // [tokens * n_groups]
    __global const float* val_zeros,         // [tokens * n_groups]
    __global float* output,                  // [dim]
    const int tokens,
    const int dim,
    const int val_bits,
    const int val_packed_stride,
    const int group_size,
    const int n_groups
) {
    int j = get_global_id(0);
    if (j >= dim) return;

    int g = j / group_size;

    float m_i = -1e30f;
    float l_i = 0.0f;
    float acc = 0.0f;

    for (int n = 0; n < tokens; n++) {
        float s = scores[n];
        float m_new = (s > m_i) ? s : m_i;
        float alpha = exp(m_i - m_new);
        float p = exp(s - m_new);

        // Dequant value for this dimension
        __global const uchar* vp = val_packed + n * val_packed_stride;
        uint raw;
        if (val_bits == 4) raw = val_tiled_extract_4bit(vp, (uint)j);
        else if (val_bits == 3) raw = val_tiled_extract_3bit(vp, (uint)j);
        else raw = val_tiled_extract_2bit(vp, (uint)j);
        float val = (float)raw * val_scales[n * n_groups + g] + val_zeros[n * n_groups + g];

        acc = acc * alpha + p * val;
        l_i = l_i * alpha + p;
        m_i = m_new;
    }

    output[j] = (l_i > 0.0f) ? acc / l_i : 0.0f;
}
