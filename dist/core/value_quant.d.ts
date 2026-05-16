/**
 * Value quantization — symmetric group quantization for value vectors.
 * Port of donor turboquant/kv_cache.py quantize_values/dequantize_values.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
export interface ValueQuantized {
    /** Bit-packed quantized values, flat Uint8Array */
    data: Uint8Array;
    /** Per-group scale factors */
    scales: Float32Array;
    /** Per-group zero points */
    zeros: Float32Array;
    /** Number of tokens */
    numTokens: number;
    /** Head dimension (unpacked) */
    headDim: number;
    /** Quantization bits */
    bits: 2 | 3 | 4;
    /** Group size used */
    groupSize: number;
}
/**
 * Quantize value vectors using symmetric group quantization.
 * @param values - flat Float32Array of shape (numTokens * headDim)
 * @param numTokens - number of tokens
 * @param headDim - dimension per token
 * @param bits - 2 or 4
 * @param groupSize - elements per quantization group
 */
export declare function quantizeValues(values: Float32Array, numTokens: number, headDim: number, bits?: 2 | 3 | 4, groupSize?: number): ValueQuantized;
/**
 * Dequantize value vectors from bit-packed format.
 */
export declare function dequantizeValues(vq: ValueQuantized): Float32Array;
/**
 * Concatenate two ValueQuantized along the token dimension.
 */
export declare function concatValueQuantized(a: ValueQuantized, b: ValueQuantized): ValueQuantized;
