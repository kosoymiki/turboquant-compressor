/**
 * RaBitQ baseline — 1-bit sign/hypercube quantization for symmetric comparison.
 * Not a claim that TurboQuant > RaBitQ; this is a fair baseline.
 *
 * Algorithm: normalize → rotate → sign(x) → popcount scoring.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
export interface RaBitQEncoded {
    /** Packed sign bits (ceil(d/8) bytes) */
    signs: Uint8Array;
    /** Original L2 norm */
    norm: number;
    /** Dimension */
    dimension: number;
}
/**
 * Encode a vector using RaBitQ: normalize → rotate → sign quantize.
 */
export declare function rabitqEncode(vector: Float32Array, seed?: number): RaBitQEncoded;
/**
 * Popcount-based inner product estimation between query and encoded vector.
 * score ≈ norm * (2 * popcount(xor(q_signs, v_signs)) / d - 1) * correction
 */
export declare function rabitqScore(query: Float32Array, encoded: RaBitQEncoded, seed?: number): number;
/**
 * Batch score: query against multiple encoded vectors.
 */
export declare function rabitqBatchScore(query: Float32Array, encoded: RaBitQEncoded[], seed?: number): number[];
/**
 * Brute-force exact inner product for baseline comparison.
 */
export declare function exactInnerProduct(a: Float32Array, b: Float32Array): number;
/**
 * Recall@k: fraction of true top-k that appear in approximate top-k.
 */
export declare function recallAtK(query: Float32Array, database: Float32Array[], encoded: RaBitQEncoded[], k: number, seed?: number): number;
