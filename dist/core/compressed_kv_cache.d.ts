/**
 * Compressed KV Cache v3.3.0 — full donor port (llama-cpp-turboquant).
 *
 * Donor optimizations ported:
 *   1. Mixed KV types (different bits for K and V)
 *   2. Block size 128 (donor default, better cache locality)
 *   3. Precomputed scaled centroids per block (eliminates per-element multiply)
 *   4. Sparse V threshold (skip value dequant for low-contribution tokens)
 *   5. q8 query quantization path (reduced bandwidth in scoring)
 *   6. Tiled token processing (process in blocks of TILE_SIZE)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
import { ValueQuantized } from './value_quant.js';
export interface KeyQuantized {
    indices: Uint8Array;
    norms: Float32Array;
    numTokens: number;
    headDim: number;
    bits: number;
}
export interface CompressedKVCacheConfig {
    headDim: number;
    keyBits?: 1 | 2 | 3 | 4 | 5 | 6;
    valueBits?: 2 | 3 | 4;
    valueGroupSize?: number;
    bufferSize?: number;
    rotationSeed?: number;
    sparseThreshold?: number;
    useQ8Query?: boolean;
    /** RotateKV: reorder channels by outlier magnitude before rotation */
    outlierAwareRotation?: boolean;
    /** Number of initial tokens protected from sparse V skip (attention sinks) */
    sinkTokens?: number;
}
export declare class CompressedKVCache {
    private readonly headDim;
    private readonly keyBits;
    private readonly valueBits;
    private readonly valueGroupSize;
    private readonly bufferSize;
    private readonly sparseThreshold;
    private readonly useQ8Query;
    private readonly outlierAwareRotation;
    private readonly sinkTokens;
    private readonly rotationEngine;
    private readonly codebook;
    private readonly quantizer;
    private readonly centroidValues;
    private channelOrder;
    private seqLen;
    private keyQuantized;
    private valueQuantized;
    private keyBuffer;
    private valueBuffer;
    private bufferTokenCount;
    private sparseSkipCount;
    private totalScoreCount;
    constructor(config: CompressedKVCacheConfig);
    /**
     * RotateKV: compute channel reorder by outlier magnitude.
     * Sorts channels so high-variance ones are adjacent → FWHT spreads them evenly.
     */
    private computeChannelOrder;
    private reorderChannels;
    private inverseReorderChannels;
    private quantizeKeys;
    /**
     * Donor-enhanced attention scoring with sparse V skip.
     * Processes tokens in tiles of TILE_SIZE for cache locality.
     */
    computeAttention(query: Float32Array, kq: KeyQuantized, vq: ValueQuantized): Float32Array;
    dequantizeKeys(kq: KeyQuantized): Float32Array;
    prefill(keys: Float32Array, values: Float32Array, numTokens: number): void;
    append(key: Float32Array, value: Float32Array): void;
    private flushBuffer;
    getSeqLength(): number;
    getKeyQuantized(): KeyQuantized | null;
    getValueQuantized(): ValueQuantized | null;
    getKeyBuffer(): Float32Array | null;
    getValueBuffer(): Float32Array | null;
    getBufferTokenCount(): number;
    getSparseStats(): {
        skipped: number;
        total: number;
        ratio: number;
    };
    memoryBytes(): {
        quantizedKeys: number;
        quantizedValues: number;
        buffer: number;
        total: number;
    };
}
