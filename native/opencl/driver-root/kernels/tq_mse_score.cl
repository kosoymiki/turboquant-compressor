/**
 * TurboQuant MSE Score Kernel — compute approximate dot scores from packed indices.
 * score[n] = norms[n] * sum_j( query_rotated[j] * centroids[unpack(packed[n], j)] )
 *
 * Supports 2/3/4-bit packing. No full dequantization buffer.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// Extract bits from packed byte array
inline uint tq_extract_bits(__global const uchar* packed, uint byte_offset, uint bit_offset_in_byte, uint bits) {
    uint val = (uint)packed[byte_offset];
    if (bit_offset_in_byte + bits > 8) {
        val |= ((uint)packed[byte_offset + 1]) << 8;
    }
    return (val >> bit_offset_in_byte) & ((1u << bits) - 1u);
}

// 4-bit extraction (2 values per byte, aligned)
inline uint tq_extract_4bit(__global const uchar* packed, uint coord) {
    uint byte_idx = coord >> 1;
    uint shift = (coord & 1u) * 4u;
    return ((uint)packed[byte_idx] >> shift) & 0xFu;
}

// 2-bit extraction (4 values per byte, aligned)
inline uint tq_extract_2bit(__global const uchar* packed, uint coord) {
    uint byte_idx = coord >> 2;
    uint shift = (coord & 3u) * 2u;
    return ((uint)packed[byte_idx] >> shift) & 0x3u;
}

// 3-bit extraction (unaligned, general path)
inline uint tq_extract_3bit(__global const uchar* packed, uint coord) {
    uint bit_pos = coord * 3u;
    uint byte_idx = bit_pos >> 3;
    uint bit_off = bit_pos & 7u;
    uint word = (uint)packed[byte_idx];
    if (bit_off + 3 > 8) {
        word |= ((uint)packed[byte_idx + 1]) << 8;
    }
    return (word >> bit_off) & 0x7u;
}

__kernel void tq_mse_score(
    __global const float* query_rotated,   // [dim]
    __global const uchar* packed_indices,  // [tokens * packed_stride]
    __global const float* norms,           // [tokens]
    __constant const float* centroids,     // [2^bits]
    __global float* scores,                // [tokens]
    const int tokens,
    const int dim,
    const int bits,
    const int packed_stride
) {
    int tid = get_global_id(0);
    if (tid >= tokens) return;

    __global const uchar* my_packed = packed_indices + tid * packed_stride;
    float dot = 0.0f;

    if (bits == 4) {
        for (int j = 0; j < dim; j++) {
            uint idx = tq_extract_4bit(my_packed, (uint)j);
            dot += query_rotated[j] * centroids[idx];
        }
    } else if (bits == 2) {
        for (int j = 0; j < dim; j++) {
            uint idx = tq_extract_2bit(my_packed, (uint)j);
            dot += query_rotated[j] * centroids[idx];
        }
    } else if (bits == 3) {
        for (int j = 0; j < dim; j++) {
            uint idx = tq_extract_3bit(my_packed, (uint)j);
            dot += query_rotated[j] * centroids[idx];
        }
    } else {
        // General fallback
        for (int j = 0; j < dim; j++) {
            uint bit_pos = (uint)j * (uint)bits;
            uint byte_idx = bit_pos >> 3;
            uint bit_off = bit_pos & 7u;
            uint word = (uint)my_packed[byte_idx];
            if (bit_off + (uint)bits > 8u) {
                word |= ((uint)my_packed[byte_idx + 1]) << 8;
            }
            uint idx = (word >> bit_off) & ((1u << bits) - 1u);
            dot += query_rotated[j] * centroids[idx];
        }
    }

    scores[tid] = dot * norms[tid];
}

// Tiled variant: one work-group per token, local reduction over dim
__kernel void tq_mse_score_tiled(
    __global const float* query_rotated,
    __global const uchar* packed_indices,
    __global const float* norms,
    __constant const float* centroids,
    __global float* scores,
    const int tokens,
    const int dim,
    const int bits,
    const int packed_stride,
    __local float* scratch
) {
    int token_id = get_group_id(0);
    if (token_id >= tokens) return;

    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    __global const uchar* my_packed = packed_indices + token_id * packed_stride;
    float partial = 0.0f;

    for (int j = lid; j < dim; j += local_size) {
        uint idx;
        if (bits == 4) idx = tq_extract_4bit(my_packed, (uint)j);
        else if (bits == 2) idx = tq_extract_2bit(my_packed, (uint)j);
        else if (bits == 3) idx = tq_extract_3bit(my_packed, (uint)j);
        else {
            uint bit_pos = (uint)j * (uint)bits;
            uint byte_idx = bit_pos >> 3;
            uint bit_off = bit_pos & 7u;
            uint word = (uint)my_packed[byte_idx];
            if (bit_off + (uint)bits > 8u) word |= ((uint)my_packed[byte_idx + 1]) << 8;
            idx = (word >> bit_off) & ((1u << bits) - 1u);
        }
        partial += query_rotated[j] * centroids[idx];
    }

    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction
    for (int s = local_size >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        scores[token_id] = scratch[0] * norms[token_id];
    }
}
