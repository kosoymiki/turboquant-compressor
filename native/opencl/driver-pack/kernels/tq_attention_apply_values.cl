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

#ifndef TQ_ITEMS_PER_THREAD
#define TQ_ITEMS_PER_THREAD 1
#endif

#ifndef TQ_TILE_K
#define TQ_TILE_K 8
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
    float m_i = -1e30f;
    float l_i = 0.0f;
    int base_j = get_global_id(0) * TQ_ITEMS_PER_THREAD;
    if (base_j >= dim) return;

    float acc[TQ_ITEMS_PER_THREAD];
    int groups[TQ_ITEMS_PER_THREAD];
    int valid = 0;
    for (int item = 0; item < TQ_ITEMS_PER_THREAD; ++item) {
        int j = base_j + item;
        if (j < dim) {
            acc[item] = 0.0f;
            groups[item] = j / group_size;
            valid++;
        } else {
            acc[item] = 0.0f;
            groups[item] = 0;
        }
    }

#if TQ_USE_GMEM_TILING
    __local float score_tile[TQ_TILE_K];
    int lid = get_local_id(0);
    for (int n0 = 0; n0 < tokens; n0 += TQ_TILE_K) {
        if (lid < TQ_TILE_K && (n0 + lid) < tokens) {
            score_tile[lid] = scores[n0 + lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int tile_count = min(TQ_TILE_K, tokens - n0);
        for (int t = 0; t < tile_count; ++t) {
            int n = n0 + t;
            float s = score_tile[t];
            float m_new = (s > m_i) ? s : m_i;
            float alpha = exp(m_i - m_new);
            float p = exp(s - m_new);
            __global const uchar* vp = val_packed + n * val_packed_stride;
            for (int item = 0; item < valid; ++item) {
                int j = base_j + item;
                uint raw;
                if (val_bits == 4) raw = val_tiled_extract_4bit(vp, (uint)j);
                else if (val_bits == 3) raw = val_tiled_extract_3bit(vp, (uint)j);
                else raw = val_tiled_extract_2bit(vp, (uint)j);
                float val = mad((float)raw, val_scales[n * n_groups + groups[item]], val_zeros[n * n_groups + groups[item]]);
                acc[item] = acc[item] * alpha + p * val;
            }
            l_i = l_i * alpha + p;
            m_i = m_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#else
    #pragma unroll 4
    for (int n = 0; n < tokens; n++) {
        float s = scores[n];
        float m_new = (s > m_i) ? s : m_i;
        float alpha = exp(m_i - m_new);
        float p = exp(s - m_new);
        __global const uchar* vp = val_packed + n * val_packed_stride;
        for (int item = 0; item < valid; ++item) {
            int j = base_j + item;
            uint raw;
            if (val_bits == 4) raw = val_tiled_extract_4bit(vp, (uint)j);
            else if (val_bits == 3) raw = val_tiled_extract_3bit(vp, (uint)j);
            else raw = val_tiled_extract_2bit(vp, (uint)j);
            float val = mad((float)raw, val_scales[n * n_groups + groups[item]], val_zeros[n * n_groups + groups[item]]);
            acc[item] = acc[item] * alpha + p * val;
        }
        l_i = l_i * alpha + p;
        m_i = m_new;
    }
#endif

    for (int item = 0; item < valid; ++item) {
        int j = base_j + item;
        output[j] = (l_i > 0.0f) ? acc[item] / l_i : 0.0f;
    }
}
