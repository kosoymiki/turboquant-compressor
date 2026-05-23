/**
 * TurboQuant compression tool.
 * Implements TurboQuant-inspired vector compression with rotation and quantization.
 */

import { RotationEngine } from '../core/rotation.js';
import { packValues, unpackValues } from '../core/pack.js';
import { UniformSymmetricCodebook } from '../core/quantizer.js';
import { encodeCompressedDatabase, encodeBase64 } from '../core/format.js';
import { validateCompressionParams, estimateCompressionMemory } from '../core/limits.js';
import { QJLResidualEstimator } from '../core/qjl.js';
import { TurboQuantBetaCodebook } from '../core/codebook.js';
import { EXPERIMENTAL_QJL_LEVEL, PUBLIC_ALGORITHM_LEVEL, algorithmLevelDescription } from '../core/algorithm_level.js';
import { parseCompressInput } from './validation.js';
import type { CompressResult } from './types.js';

export function compressVectors(
  input: {
    vectors: number[][];
    dimensions?: number;
    seed?: number;
    bitsPerValue?: number;
    codebookType?: 'uniform' | 'turboquant_beta';
    includeQJL?: boolean;
    rotationSeed?: number;
  }
): CompressResult {
  const parsed = parseCompressInput(input);

  const vectors = parsed.vectors;
  const dimensions = parsed.dimensions ?? vectors[0]!.length;
  const seed = parsed.seed ?? parsed.rotationSeed ?? 0;
  const bitsPerValue = parsed.bitsPerValue ?? 4;
  const requestedCodebookType = parsed.codebookType ?? 'turboquant_beta';
  const codebookType = requestedCodebookType === 'turboquant_beta' && bitsPerValue !== 8
    ? 'turboquant_beta'
    : 'uniform';
  const includeQJL = parsed.includeQJL ?? false;

  validateCompressionParams(dimensions, vectors.length, bitsPerValue);

  const rotation = new RotationEngine(dimensions!, seed);
  const uniformCodebook = new UniformSymmetricCodebook(bitsPerValue);
  const betaCodebook = codebookType === 'turboquant_beta'
    ? (() => {
        const cb = new TurboQuantBetaCodebook(dimensions!, bitsPerValue as 2 | 3 | 4);
        cb.compute();
        return cb;
      })()
    : null;

  function encodeWithConfiguredCodebook(values: Float32Array): Uint8Array {
    if (codebookType === 'uniform' || !betaCodebook) {
      return uniformCodebook.encode(values);
    }
    const paddedDimensions = Number.isInteger(dimensions) && dimensions > 0
      ? 1 << Math.ceil(Math.log2(dimensions))
      : values.length;
    const indices: number[] = [];
    for (let i = 0; i < paddedDimensions; i++) {
      indices.push(betaCodebook.quantizeIndex(values[i] ?? 0));
    }
    return packValues(indices, bitsPerValue);
  }

  function decodeWithConfiguredCodebook(packed: Uint8Array, count: number): Float32Array {
    if (codebookType === 'uniform' || !betaCodebook) {
      return uniformCodebook.decode(packed, count);
    }
    const out = new Float32Array(count);
    const indices = unpackValues(packed, count, bitsPerValue);
    for (let i = 0; i < count; i++) {
      out[i] = betaCodebook.dequantize(indices[i]!);
    }
    return out;
  }

  const warnings: string[] = [];
  if (!includeQJL) {
    warnings.push(`Note: includeQJL=true is recommended for production use. QJL residual correction provides unbiased dot-product estimation per arXiv:2504.19874.`);
  }
  if (codebookType === 'uniform') {
    warnings.push('Uniform quantizer has fixed step size — Beta Lloyd-Max codebook is recommended for production.');
  }

  const encodedVectors: Uint8Array[] = [];
  const norms = new Float32Array(vectors.length);
  let qjlSketchesB64: string | undefined;

  // QJL residual estimator (Hadamard + projection + 4-bit quantized sketch)
  const qjlEstimator = includeQJL ? new QJLResidualEstimator({ targetDimensions: Math.min(Math.max(256, Math.floor(dimensions! / 4), 512)), bitsPerValue: 4, seed, useHadamard: true }) : null;
  const qjlSketches: Uint8Array[] = [];

  for (let i = 0; i < vectors.length; i++) {
    const vector = vectors[i]!;
    const floatVector = new Float32Array(vector.length);
    for (let j = 0; j < vector.length; j++) {
      const val = vector[j];
      if (val !== undefined) {
        floatVector[j] = val;
      }
    }

    const rotated = rotation.rotate(floatVector);
    const norm = Math.sqrt(rotated.reduce((sum, v) => sum + v * v, 0));
    norms[i] = norm;
    const normalizedRotated = norm === 0
        ? new Float32Array(rotated.length)
        : (() => {
            const normalized = new Float32Array(rotated.length);
            for (let k = 0; k < rotated.length; k++) {
              const val = rotated[k];
              if (val !== undefined) {
                normalized[k] = val / norm;
              }
            }
            return normalized;
          })();
    const encoded = encodeWithConfiguredCodebook(normalizedRotated);
    encodedVectors.push(encoded);

    // QJL: compute residual and compress sign-bit sketch
    if (qjlEstimator) {
      const decoded = decodeWithConfiguredCodebook(encoded, normalizedRotated.length);
      const residual = new Float32Array(normalizedRotated.length);
      for (let k = 0; k < residual.length; k++) {
        residual[k] = (normalizedRotated[k] ?? 0) - (decoded[k] ?? 0);
      }
      const compressed = qjlEstimator.compress(residual);
      qjlSketches.push(compressed.sketch);
    }
  }

  // Pack QJL sketches into single base64 blob
  if (includeQJL && qjlSketches.length > 0) {
    const totalBytes = qjlSketches.reduce((s, sk) => s + sk.length, 0);
    const combined = new Uint8Array(totalBytes);
    let offset = 0;
    for (const sk of qjlSketches) {
      combined.set(sk, offset);
      offset += sk.length;
    }
    qjlSketchesB64 = encodeBase64(combined);
  }

  const binary = encodeCompressedDatabase(
    encodedVectors,
    dimensions!,
    bitsPerValue,
    seed,
    norms,
    includeQJL && qjlSketches.length > 0 ? Buffer.from(qjlSketchesB64!, 'base64') : undefined,
    codebookType
  );
  const compressedData = encodeBase64(binary);

  const originalSize = vectors.length * dimensions! * 4;
  const compressedSize = binary.length;
  const compressionRatio = originalSize / compressedSize;

  return {
    compressed_database_b64: compressedData,
    dimensions: dimensions!,
    vector_count: vectors.length,
    bits_per_value: bitsPerValue,
    codebook_type: codebookType,
    include_qjl: includeQJL,
    qjl_sketches_b64: qjlSketchesB64,
    algorithm_level: includeQJL ? EXPERIMENTAL_QJL_LEVEL : PUBLIC_ALGORITHM_LEVEL,
    original_bytes_estimate: originalSize,
    compressed_bytes: compressedSize,
    compression_ratio: compressionRatio,
    format_version: 3,
    warnings
  };
}
