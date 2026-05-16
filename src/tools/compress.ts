/**
 * TurboQuant compression tool.
 * Implements TurboQuant-inspired vector compression with rotation and quantization.
 */

import { RotationEngine } from '../core/rotation.js';
import { UniformSymmetricCodebook } from '../core/quantizer.js';
import { encodeCompressedDatabase, encodeBase64 } from '../core/format.js';
import { validateCompressionParams, estimateCompressionMemory } from '../core/limits.js';
import { QJLResidualEstimator } from '../core/qjl.js';
import { parseCompressInput } from './validation.js';
import type { CompressResult } from './types.js';

export function compressVectors(
  input: {
    vectors: number[][];
    dimensions?: number;
    seed?: number;
    bitsPerValue?: number;
    includeQJL?: boolean;
    rotationSeed?: number;
  }
): CompressResult {
  const parsed = parseCompressInput(input);

  const vectors = parsed.vectors;
  const dimensions = parsed.dimensions ?? vectors[0]!.length;
  const seed = parsed.seed ?? parsed.rotationSeed ?? 0;
  const bitsPerValue = parsed.bitsPerValue ?? 4;
  const includeQJL = parsed.includeQJL ?? false;

  validateCompressionParams(dimensions, vectors.length, bitsPerValue);

  const rotation = new RotationEngine(dimensions!, seed);
  const codebook = new UniformSymmetricCodebook(bitsPerValue);

  const warnings: string[] = [];
  warnings.push('LEVEL_1_TURBOQUANT: Production implementation with QJL sign-bit residual correction');

  const encodedVectors: Uint8Array[] = [];
  const norms = new Float32Array(vectors.length);
  let qjlSketchesB64: string | undefined;

  // QJL residual estimator (Hadamard + projection + 4-bit quantized sketch)
  const qjlEstimator = includeQJL ? new QJLResidualEstimator({ targetDimensions: Math.min(64, dimensions!), bitsPerValue: 4, seed, useHadamard: true }) : null;
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
    const encoded = codebook.encode(normalizedRotated);
    encodedVectors.push(encoded);

    // QJL: compute residual and compress sign-bit sketch
    if (qjlEstimator) {
      const decoded = codebook.decode(encoded);
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

  const binary = encodeCompressedDatabase(encodedVectors, dimensions!, bitsPerValue, seed, norms);
  const compressedData = encodeBase64(binary);

  const originalSize = vectors.length * dimensions! * 4;
  const compressedSize = binary.length;
  const compressionRatio = originalSize / compressedSize;

  return {
    compressed_database_b64: compressedData,
    dimensions: dimensions!,
    vector_count: vectors.length,
    bits_per_value: bitsPerValue,
    include_qjl: includeQJL,
    qjl_sketches_b64: qjlSketchesB64,
    algorithm_level: includeQJL ? 'LEVEL_1_TURBOQUANT_QJL' : 'LEVEL_1_TURBOQUANT',
    original_bytes_estimate: originalSize,
    compressed_bytes: compressedSize,
    compression_ratio: compressionRatio,
    format_version: 3,
    warnings
  };
}