/* tslint:disable */
/* eslint-disable */

/**
 * Fast Walsh-Hadamard Transform (in-place, power-of-2)
 * Butterfly is data-dependent (not SIMD-able), normalize uses f32x4.
 */
export function fwht(data: Float32Array): void;

/**
 * Batch FWHT: apply to N vectors of same dimension
 */
export function fwht_batch(data: Float32Array, dim: number): void;

/**
 * Dequantize packed indices back to f32 using centroids
 */
export function lloyd_max_dequantize(packed: Uint8Array, centroids: Float32Array, bits: number, count: number): Float32Array;

/**
 * Lloyd-Max quantize: find nearest centroid index for each value
 * centroids: sorted array of centroid values
 * Returns packed indices (bits_per_value per element)
 */
export function lloyd_max_quantize(values: Float32Array, centroids: Float32Array, bits: number): Uint8Array;

/**
 * Bit-pack 4 values (2-bit each) into 1 byte — SIMD-friendly layout
 */
export function pack_2bit(indices: Uint8Array): Uint8Array;

/**
 * QJL 1-bit sign projection: dot product signs with random matrix row
 */
export function qjl_sign_project(vector: Float32Array, random_matrix: Float32Array, proj_dim: number): Uint8Array;

/**
 * Exposed SIMD dot product for JS
 */
export function simd_dot(a: Float32Array, b: Float32Array): number;

/**
 * Unpack 2-bit values
 */
export function unpack_2bit(packed: Uint8Array, count: number): Uint8Array;
