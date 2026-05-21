/**
 * TurboQuant Value Dequantization Kernel — unpack 2/4-bit grouped values.
 * output[n][j] = packed_value[n][j] * scales[n][group] + zeros[n][group]
 *
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

// 4-bit extraction
inline uint tq_val_extract_4bit(__global const uchar* packed, uint coord) {
    uint byte_idx = coord >> 1;
    uint shift = (coord & 1u) * 4u;
    return ((uint)packed[byte_idx] >> shift) & 0xFu;
}

// 2-bit extraction
inline uint tq_val_extract_2bit(__global const uchar* packed, uint coord) {
    uint byte_idx = coord >> 2;
    uint shift = (coord & 3u) * 2u;
    return ((uint)packed[byte_idx] >> shift) & 0x3u;
}

__kernel void tq_value_dequant(
    __global const uchar* packed_values,  // [tokens * packed_stride]
    __global const float* scales,         // [tokens * n_groups]
    __global const float* zeros,          // [tokens * n_groups]
    __global float* output,               // [tokens * dim]
    const int tokens,
    const int dim,
    const int bits,
    const int group_size,
    const int packed_stride,
    const int n_groups
) {
    int tid = get_global_id(0);  // token index
    if (tid >= tokens) return;

    __global const uchar* my_packed = packed_values + tid * packed_stride;
    __global const float* my_scales = scales + tid * n_groups;
    __global const float* my_zeros = zeros + tid * n_groups;
    __global float* my_output = output + tid * dim;

    for (int j = 0; j < dim; j++) {
        uint raw;
        if (bits == 4) raw = tq_val_extract_4bit(my_packed, (uint)j);
        else if (bits == 2) raw = tq_val_extract_2bit(my_packed, (uint)j);
        else {
            // General path
            uint bit_pos = (uint)j * (uint)bits;
            uint byte_idx = bit_pos >> 3;
            uint bit_off = bit_pos & 7u;
            uint word = (uint)my_packed[byte_idx];
            if (bit_off + (uint)bits > 8u) word |= ((uint)my_packed[byte_idx + 1]) << 8;
            raw = (word >> bit_off) & ((1u << bits) - 1u);
        }

        int g = j / group_size;
        my_output[j] = (float)raw * my_scales[g] + my_zeros[g];
    }
}

// Weighted accumulation variant: directly accumulate p[n] * dequant(v[n])
// Avoids materializing full dequantized value matrix
__kernel void tq_value_weighted_accum(
    __global const uchar* packed_values,
    __global const float* scales,
    __global const float* zeros,
    __global const float* weights,        // [tokens] — softmax probabilities
    __global float* accum,                // [dim] — output accumulator
    const int tokens,
    const int dim,
    const int bits,
    const int group_size,
    const int packed_stride,
    const int n_groups
) {
    int j = get_global_id(0);  // dimension index
    if (j >= dim) return;

    int g = j / group_size;
    float sum = 0.0f;

    for (int n = 0; n < tokens; n++) {
        __global const uchar* my_packed = packed_values + n * packed_stride;
        uint raw;
        if (bits == 4) raw = tq_val_extract_4bit(my_packed, (uint)j);
        else if (bits == 2) raw = tq_val_extract_2bit(my_packed, (uint)j);
        else {
            uint bit_pos = (uint)j * (uint)bits;
            uint byte_idx = bit_pos >> 3;
            uint bit_off = bit_pos & 7u;
            uint word = (uint)my_packed[byte_idx];
            if (bit_off + (uint)bits > 8u) word |= ((uint)my_packed[byte_idx + 1]) << 8;
            raw = (word >> bit_off) & ((1u << bits) - 1u);
        }

        float val = (float)raw * scales[n * n_groups + g] + zeros[n * n_groups + g];
        sum += weights[n] * val;
    }

    accum[j] += sum;
}
