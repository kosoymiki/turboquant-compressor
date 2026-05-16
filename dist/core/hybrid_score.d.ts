/**
 * Hybrid attention scoring — compressed history + exact recent buffer.
 * Port of donor turboquant/score.py compute_hybrid_attention.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
import { CompressedKVCache } from './compressed_kv_cache.js';
export interface HybridAttentionResult {
    /** Output vector (headDim) */
    output: Float32Array;
    /** Number of tokens attended over */
    totalTokens: number;
    /** Tokens from compressed history */
    compressedTokens: number;
    /** Tokens from exact buffer */
    bufferTokens: number;
}
/**
 * Compute attention output for a single query over the compressed KV cache.
 *
 * query: Float32Array of length headDim
 * cache: CompressedKVCache instance
 * scale: attention scale factor (default 1/sqrt(headDim))
 */
export declare function computeHybridAttention(query: Float32Array, cache: CompressedKVCache, scale?: number): HybridAttentionResult;
