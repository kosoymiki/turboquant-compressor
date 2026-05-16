/**
 * Hybrid attention scoring — compressed history + exact recent buffer.
 * Port of donor turboquant/score.py compute_hybrid_attention.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { CompressedKVCache } from './compressed_kv_cache.js';
import { dequantizeValues } from './value_quant.js';

const MIN_HISTORY_FOR_TQ = 16;

/**
 * Compute dot product between two vectors.
 */
function dot(a: Float32Array, b: Float32Array, len: number): number {
  let sum = 0;
  for (let i = 0; i < len; i++) {
    sum += a[i]! * b[i]!;
  }
  return sum;
}

/**
 * Softmax in-place over a Float32Array.
 */
function softmaxInPlace(arr: Float32Array): void {
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i]! > max) max = arr[i]!;
  }
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    arr[i] = Math.exp(arr[i]! - max);
    sum += arr[i]!;
  }
  const invSum = 1 / sum;
  for (let i = 0; i < arr.length; i++) {
    arr[i] = arr[i]! * invSum;
  }
}

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
export function computeHybridAttention(
  query: Float32Array,
  cache: CompressedKVCache,
  scale?: number
): HybridAttentionResult {
  const headDim = query.length;
  const s = scale ?? 1 / Math.sqrt(headDim);

  const keyQ = cache.getKeyQuantized();
  const valQ = cache.getValueQuantized();
  const keyBuf = cache.getKeyBuffer();
  const valBuf = cache.getValueBuffer();
  const bufTokens = cache.getBufferTokenCount();

  const hasHistory = keyQ !== null && keyQ.numTokens >= MIN_HISTORY_FOR_TQ;
  const hasRecent = keyBuf !== null && bufTokens > 0;

  if (!hasHistory && !hasRecent) {
    return { output: new Float32Array(headDim), totalTokens: 0, compressedTokens: 0, bufferTokens: 0 };
  }

  // Dequantize compressed keys/values if available
  let histKeys: Float32Array | null = null;
  let histVals: Float32Array | null = null;
  let nHist = 0;

  if (hasHistory) {
    histKeys = cache.dequantizeKeys(keyQ!);
    histVals = dequantizeValues(valQ!);
    nHist = keyQ!.numTokens;
  }

  const nBuf = hasRecent ? bufTokens : 0;
  const totalTokens = nHist + nBuf;

  // Compute attention scores
  const scores = new Float32Array(totalTokens);

  // Compressed history scores
  for (let t = 0; t < nHist; t++) {
    const keyOffset = t * headDim;
    let score = 0;
    for (let i = 0; i < headDim; i++) {
      score += query[i]! * histKeys![keyOffset + i]!;
    }
    scores[t] = score * s;
  }

  // Buffer scores
  for (let t = 0; t < nBuf; t++) {
    const keyOffset = t * headDim;
    let score = 0;
    for (let i = 0; i < headDim; i++) {
      score += query[i]! * keyBuf![keyOffset + i]!;
    }
    scores[nHist + t] = score * s;
  }

  // Softmax
  softmaxInPlace(scores);

  // Weighted sum of values
  const output = new Float32Array(headDim);

  for (let t = 0; t < nHist; t++) {
    const w = scores[t]!;
    const valOffset = t * headDim;
    for (let i = 0; i < headDim; i++) {
      output[i] = (output[i] ?? 0) + w * histVals![valOffset + i]!;
    }
  }

  for (let t = 0; t < nBuf; t++) {
    const w = scores[nHist + t]!;
    const valOffset = t * headDim;
    for (let i = 0; i < headDim; i++) {
      output[i] = (output[i] ?? 0) + w * valBuf![valOffset + i]!;
    }
  }

  return { output, totalTokens, compressedTokens: nHist, bufferTokens: nBuf };
}
