/**
 * Compressed KV Cache v4.5.0 — TurboQuant implementation.
 *
 * Current implementation characteristics:
 *   1. Mixed KV types (different bits for K and V)
 *   2. Block size 128 for cache-local scoring
 *   3. Precomputed scaled centroids per block
 *   4. Sparse V thresholding for low-contribution tokens
 *   5. q8 query quantization path
 *   6. Tiled token processing (TILE_SIZE)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { RotationEngine, FWHT_SIGN } from './rotation.js';
import { TurboQuantBetaCodebook, TurboQuantCodebookQuantizer } from './codebook.js';
import { quantizeValues, dequantizeValues, concatValueQuantized, ValueQuantized } from './value_quant.js';

// === Runtime constants ===
const BLOCK_SIZE = 128;          // Current tuned default (was 32)
const TILE_SIZE = 64;            // Process tokens in tiles
const SPARSE_V_DEFAULT = 0.001;  // Skip V if contrib < this
const ATTENTION_SINK_TOKENS = 4; // RotateKV: protect first N tokens from sparse skip

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

function concatKeyQuantized(a: KeyQuantized, b: KeyQuantized): KeyQuantized {
  const numTokens = a.numTokens + b.numTokens;
  const indices = new Uint8Array(a.indices.length + b.indices.length);
  indices.set(a.indices, 0);
  indices.set(b.indices, a.indices.length);
  const norms = new Float32Array(a.norms.length + b.norms.length);
  norms.set(a.norms, 0);
  norms.set(b.norms, a.norms.length);
  return { indices, norms, numTokens, headDim: a.headDim, bits: a.bits };
}

// q8 query quantization path
interface Q8Vector {
  quantized: Int8Array;
  scale: number;
}

function quantizeQ8(vec: Float32Array): Q8Vector {
  let amax = 0;
  for (let i = 0; i < vec.length; i++) {
    const a = Math.abs(vec[i]!);
    if (a > amax) amax = a;
  }
  // Guard: zero vector → scale=1e-10 prevents division artifacts downstream
  if (amax < 1e-30) {
    return { quantized: new Int8Array(vec.length), scale: 1e-10 };
  }
  const scale = amax / 127;
  const invScale = 127 / amax;
  const quantized = new Int8Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    quantized[i] = Math.round(vec[i]! * invScale);
  }
  return { quantized, scale };
}

// Precomputed scaled centroids
function precomputeScaledCentroids(
  centroids: Float32Array,
  queryQ8: Q8Vector
): Float32Array {
  // centroid[i] * q8.scale = effective centroid for q8 dot product
  const scaled = new Float32Array(centroids.length);
  for (let i = 0; i < centroids.length; i++) {
    scaled[i] = centroids[i]! * queryQ8.scale;
  }
  return scaled;
}

export class CompressedKVCache {
  private readonly headDim: number;
  private readonly keyBits: 1 | 2 | 3 | 4 | 5 | 6;
  private readonly valueBits: 2 | 3 | 4;
  private readonly valueGroupSize: number;
  private readonly bufferSize: number;
  private readonly sparseThreshold: number;
  private readonly useQ8Query: boolean;
  private readonly outlierAwareRotation: boolean;
  private readonly sinkTokens: number;

  private readonly rotationEngine: RotationEngine;
  private readonly codebook: TurboQuantBetaCodebook;
  private readonly quantizer: TurboQuantCodebookQuantizer;
  private readonly centroidValues: Float32Array;
  private channelOrder: Uint16Array | null = null; // RotateKV: outlier-sorted channel indices

  private seqLen: number = 0;
  private keyQuantized: KeyQuantized | null = null;
  private valueQuantized: ValueQuantized | null = null;

  private keyBuffer: Float32Array | null = null;
  private valueBuffer: Float32Array | null = null;
  private bufferTokenCount: number = 0;

  // Stats for sparse V optimization
  private sparseSkipCount: number = 0;
  private totalScoreCount: number = 0;

  constructor(config: CompressedKVCacheConfig) {
    this.headDim = config.headDim;
    this.keyBits = config.keyBits ?? 4;
    this.valueBits = config.valueBits ?? 2;
    const rawGroupSize = Math.min(config.valueGroupSize ?? BLOCK_SIZE, config.headDim);
    // Ensure groupSize divides headDim — fall back to headDim if not
    this.valueGroupSize = (config.headDim % rawGroupSize === 0) ? rawGroupSize : config.headDim;
    this.bufferSize = config.bufferSize ?? 128;
    this.sparseThreshold = config.sparseThreshold ?? SPARSE_V_DEFAULT;
    this.useQ8Query = config.useQ8Query ?? true;  // Donor default: on
    this.outlierAwareRotation = config.outlierAwareRotation ?? true;  // RotateKV default: on
    this.sinkTokens = config.sinkTokens ?? ATTENTION_SINK_TOKENS;

    this.rotationEngine = new RotationEngine(this.headDim, config.rotationSeed ?? 42, FWHT_SIGN);
    this.codebook = new TurboQuantBetaCodebook(this.headDim, this.keyBits);
    this.codebook.compute();
    this.quantizer = this.codebook.getQuantizer();

    // Cache centroid values for LUT
    const nCentroids = 1 << this.keyBits;
    this.centroidValues = new Float32Array(nCentroids);
    for (let i = 0; i < nCentroids; i++) {
      this.centroidValues[i] = this.quantizer.dequantize(i);
    }
  }

  /**
   * RotateKV: compute channel reorder by outlier magnitude.
   * Sorts channels so high-variance ones are adjacent → FWHT spreads them evenly.
   */
  private computeChannelOrder(keys: Float32Array, numTokens: number): Uint16Array {
    const d = this.headDim;
    // Compute per-channel variance across tokens
    const variance = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      let sum = 0, sum2 = 0;
      for (let t = 0; t < numTokens; t++) {
        const v = keys[t * d + i]!;
        sum += v;
        sum2 += v * v;
      }
      const mean = sum / numTokens;
      variance[i] = sum2 / numTokens - mean * mean;
    }
    // Sort indices by variance descending → interleave high/low for FWHT balance
    const order = new Uint16Array(d);
    for (let i = 0; i < d; i++) order[i] = i;
    order.sort((a, b) => variance[b]! - variance[a]!);
    return order;
  }

  private reorderChannels(vec: Float32Array, order: Uint16Array): Float32Array {
    const out = new Float32Array(vec.length);
    for (let i = 0; i < order.length; i++) out[i] = vec[order[i]!]!;
    return out;
  }

  private inverseReorderChannels(vec: Float32Array, order: Uint16Array): Float32Array {
    const out = new Float32Array(vec.length);
    for (let i = 0; i < order.length; i++) out[order[i]!] = vec[i]!;
    return out;
  }

  private quantizeKeys(keys: Float32Array, numTokens: number): KeyQuantized {
    const d = this.headDim;
    const norms = new Float32Array(numTokens);
    const bytesPerToken = Math.ceil(d * this.keyBits / 8);
    const indices = new Uint8Array(numTokens * bytesPerToken);

    // RotateKV: compute channel order on first batch
    if (this.outlierAwareRotation && !this.channelOrder && numTokens >= 4) {
      this.channelOrder = this.computeChannelOrder(keys, numTokens);
    }

    for (let t = 0; t < numTokens; t++) {
      const offset = t * d;
      let norm = 0;
      for (let i = 0; i < d; i++) norm += keys[offset + i]! * keys[offset + i]!;
      norm = Math.sqrt(norm);
      norms[t] = norm;

      const invNorm = norm > 1e-10 ? 1 / norm : 0;
      let normalized = new Float32Array(d);
      for (let i = 0; i < d; i++) normalized[i] = keys[offset + i]! * invNorm;

      // RotateKV: reorder channels before rotation
      if (this.channelOrder) {
        normalized = this.reorderChannels(normalized, this.channelOrder) as Float32Array<ArrayBuffer>;
      }

      const rotated = this.rotationEngine.rotate(normalized);

      for (let i = 0; i < d; i++) {
        const idx = this.quantizer.quantizeIndex(rotated[i]!);
        const bitOffset = t * bytesPerToken * 8 + i * this.keyBits;
        const bytePos = Math.floor(bitOffset / 8);
        const bitPos = bitOffset % 8;
        indices[bytePos]! |= (idx << bitPos) & 0xFF;
        if (bitPos + this.keyBits > 8) {
          indices[bytePos + 1]! |= (idx >> (8 - bitPos)) & 0xFF;
        }
      }
    }
    return { indices, norms, numTokens, headDim: d, bits: this.keyBits };
  }

  /**
   * Donor-enhanced attention scoring with sparse V skip.
   * Processes tokens in tiles of TILE_SIZE for cache locality.
   */
  computeAttention(
    query: Float32Array,
    kq: KeyQuantized,
    vq: ValueQuantized
  ): Float32Array {
    const d = this.headDim;
    const output = new Float32Array(d);
    const smScale = 1.0 / Math.sqrt(d);

    // Donor opt 5: q8 query path
    // RotateKV: reorder query channels to match key encoding
    let queryForRotation = query;
    if (this.channelOrder) {
      queryForRotation = this.reorderChannels(query, this.channelOrder);
    }
    const rotatedQuery = this.rotationEngine.rotate(queryForRotation);
    const bytesPerToken = Math.ceil(d * kq.bits / 8);
    const kMask = (1 << kq.bits) - 1;
    let scoreFn: (tokenIdx: number) => number;

    if (this.useQ8Query) {
      const q8 = quantizeQ8(rotatedQuery);
      const scaledCentroids = precomputeScaledCentroids(this.centroidValues, q8);

      scoreFn = (t: number) => {
        let dot = 0;
        for (let i = 0; i < d; i++) {
          const bitOffset = t * bytesPerToken * 8 + i * kq.bits;
          const bytePos = Math.floor(bitOffset / 8);
          const bitPos = bitOffset % 8;
          let idx = (kq.indices[bytePos]! >> bitPos) & kMask;
          if (bitPos + kq.bits > 8) idx |= ((kq.indices[bytePos + 1]! << (8 - bitPos)) & kMask);
          dot += q8.quantized[i]! * scaledCentroids[idx]!;
        }
        return dot * kq.norms[t]! * smScale;
      };
    } else {
      scoreFn = (t: number) => {
        let dot = 0;
        for (let i = 0; i < d; i++) {
          const bitOffset = t * bytesPerToken * 8 + i * kq.bits;
          const bytePos = Math.floor(bitOffset / 8);
          const bitPos = bitOffset % 8;
          let idx = (kq.indices[bytePos]! >> bitPos) & kMask;
          if (bitPos + kq.bits > 8) idx |= ((kq.indices[bytePos + 1]! << (8 - bitPos)) & kMask);
          dot += rotatedQuery[i]! * this.centroidValues[idx]!;
        }
        return dot * kq.norms[t]! * smScale;
      };
    }

    // Online softmax with sparse V (donor opt 2 + 6: tiled)
    let m_i = -Infinity;
    let l_i = 0;

    const nTokens = kq.numTokens;
    const nGroups = Math.ceil(d / this.valueGroupSize);
    const valsPerByte = this.valueBits === 2 ? 4 : 2;
    const vPackedDim = Math.ceil(d / valsPerByte);

    for (let tileStart = 0; tileStart < nTokens; tileStart += TILE_SIZE) {
      const tileEnd = Math.min(tileStart + TILE_SIZE, nTokens);

      for (let t = tileStart; t < tileEnd; t++) {
        const score = scoreFn(t);
        this.totalScoreCount++;

        const m_new = Math.max(score, m_i);
        const alpha = Math.exp(m_i - m_new);
        const p = Math.exp(score - m_new);

        // Donor opt 2: sparse V threshold
        // RotateKV: attention sinks (first N tokens) never skipped
        const isSink = t < this.sinkTokens;
        const contrib = p / (l_i * alpha + p);
        if (!isSink && contrib < this.sparseThreshold) {
          l_i = l_i * alpha + p;
          m_i = m_new;
          for (let j = 0; j < d; j++) output[j]! *= alpha;
          this.sparseSkipCount++;
          continue;
        }

        // Dequant value — consistent with value_quant.ts packing
        for (let j = 0; j < d; j++) {
          const flatIdx = j;
          const byteIdx = t * vPackedDim + Math.floor(flatIdx / valsPerByte);
          const shift = (flatIdx % valsPerByte) * this.valueBits;
          const vMask = (1 << this.valueBits) - 1;
          const raw = (vq.data[byteIdx]! >> shift) & vMask;
          const g = Math.floor(j / this.valueGroupSize);
          const val = raw * vq.scales[t * nGroups + g]! + vq.zeros[t * nGroups + g]!;
          output[j] = output[j]! * alpha + p * val;
        }

        l_i = l_i * alpha + p;
        m_i = m_new;
      }
    }

    // Normalize
    if (l_i > 0) {
      const invL = 1 / l_i;
      for (let j = 0; j < d; j++) output[j]! *= invL;
    }
    return output;
  }

  dequantizeKeys(kq: KeyQuantized): Float32Array {
    const d = kq.headDim;
    const bytesPerToken = Math.ceil(d * kq.bits / 8);
    const mask = (1 << kq.bits) - 1;
    const result = new Float32Array(kq.numTokens * d);

    for (let t = 0; t < kq.numTokens; t++) {
      const rotated = new Float32Array(d);
      for (let i = 0; i < d; i++) {
        const bitOffset = t * bytesPerToken * 8 + i * kq.bits;
        const bytePos = Math.floor(bitOffset / 8);
        const bitPos = bitOffset % 8;
        let idx = (kq.indices[bytePos]! >> bitPos) & mask;
        if (bitPos + kq.bits > 8) idx |= ((kq.indices[bytePos + 1]! << (8 - bitPos)) & mask);
        rotated[i] = this.centroidValues[idx]!;
      }
      let unrotated = this.rotationEngine.inverseRotate(rotated);
      // RotateKV: inverse channel reorder
      if (this.channelOrder) {
        unrotated = this.inverseReorderChannels(unrotated, this.channelOrder);
      }
      const norm = kq.norms[t]!;
      for (let i = 0; i < d; i++) result[t * d + i] = unrotated[i]! * norm;
    }
    return result;
  }

  prefill(keys: Float32Array, values: Float32Array, numTokens: number): void {
    const d = this.headDim;
    this.seqLen = numTokens;

    if (numTokens <= this.bufferSize) {
      this.keyBuffer = new Float32Array(keys);
      this.valueBuffer = new Float32Array(values);
      this.bufferTokenCount = numTokens;
      return;
    }

    const nQuant = numTokens - this.bufferSize;
    this.keyQuantized = this.quantizeKeys(keys.subarray(0, nQuant * d), nQuant);
    this.valueQuantized = quantizeValues(
      values.subarray(0, nQuant * d), nQuant, d, this.valueBits, this.valueGroupSize
    );
    this.keyBuffer = new Float32Array(keys.subarray(nQuant * d));
    this.valueBuffer = new Float32Array(values.subarray(nQuant * d));
    this.bufferTokenCount = this.bufferSize;
  }

  append(key: Float32Array, value: Float32Array): void {
    const d = this.headDim;
    this.seqLen++;

    if (this.keyBuffer === null) {
      this.keyBuffer = new Float32Array(key);
      this.valueBuffer = new Float32Array(value);
      this.bufferTokenCount = 1;
    } else {
      const newKeyBuf = new Float32Array(this.keyBuffer.length + d);
      newKeyBuf.set(this.keyBuffer, 0);
      newKeyBuf.set(key, this.keyBuffer.length);
      this.keyBuffer = newKeyBuf;

      const newValBuf = new Float32Array(this.valueBuffer!.length + d);
      newValBuf.set(this.valueBuffer!, 0);
      newValBuf.set(value, this.valueBuffer!.length);
      this.valueBuffer = newValBuf;
      this.bufferTokenCount++;
    }

    if (this.bufferTokenCount > this.bufferSize) this.flushBuffer();
  }

  private flushBuffer(): void {
    const d = this.headDim;
    const nFlush = this.bufferTokenCount - this.bufferSize;
    if (nFlush <= 0) return;

    const newKeyQ = this.quantizeKeys(new Float32Array(this.keyBuffer!.subarray(0, nFlush * d)), nFlush);
    const newValQ = quantizeValues(
      new Float32Array(this.valueBuffer!.subarray(0, nFlush * d)), nFlush, d, this.valueBits, this.valueGroupSize
    );

    if (this.keyQuantized === null) {
      this.keyQuantized = newKeyQ;
      this.valueQuantized = newValQ;
    } else {
      this.keyQuantized = concatKeyQuantized(this.keyQuantized, newKeyQ);
      this.valueQuantized = concatValueQuantized(this.valueQuantized!, newValQ);
    }

    this.keyBuffer = new Float32Array(this.keyBuffer!.subarray(nFlush * d));
    this.valueBuffer = new Float32Array(this.valueBuffer!.subarray(nFlush * d));
    this.bufferTokenCount = this.bufferSize;
  }

  getSeqLength(): number { return this.seqLen; }
  getKeyQuantized(): KeyQuantized | null { return this.keyQuantized; }
  getValueQuantized(): ValueQuantized | null { return this.valueQuantized; }
  getKeyBuffer(): Float32Array | null { return this.keyBuffer; }
  getValueBuffer(): Float32Array | null { return this.valueBuffer; }
  getBufferTokenCount(): number { return this.bufferTokenCount; }

  getSparseStats(): { skipped: number; total: number; ratio: number } {
    return {
      skipped: this.sparseSkipCount,
      total: this.totalScoreCount,
      ratio: this.totalScoreCount > 0 ? this.sparseSkipCount / this.totalScoreCount : 0,
    };
  }

  memoryBytes(): { quantizedKeys: number; quantizedValues: number; buffer: number; total: number } {
    let quantizedKeys = 0, quantizedValues = 0, buffer = 0;
    if (this.keyQuantized) {
      quantizedKeys += this.keyQuantized.indices.byteLength + this.keyQuantized.norms.byteLength;
    }
    if (this.valueQuantized) {
      quantizedValues += this.valueQuantized.data.byteLength + this.valueQuantized.scales.byteLength + this.valueQuantized.zeros.byteLength;
    }
    if (this.keyBuffer) buffer += this.keyBuffer.byteLength;
    if (this.valueBuffer) buffer += this.valueBuffer.byteLength;
    return { quantizedKeys, quantizedValues, buffer, total: quantizedKeys + quantizedValues + buffer };
  }
}
