/**
 * Experimental QJL-like residual sketch utilities.
 *
 * IMPORTANT: This is NOT paper-faithful QJL (Zandieh et al., 2024, arXiv:2406.03482).
 * Paper QJL uses:
 *   - Sign-bit quantization (1-bit) with zero overhead (no scale/zero stored)
 *   - Asymmetric estimator: QJL on keys, standard JL on queries → unbiased dot product
 *   - Structured JL via randomized Hadamard (O(d log d), not O(d²))
 *
 * This module uses dense random projection + multi-bit uniform quantization.
 * It is NOT wired into the public MCP compression/search tools.
 * Research-only (LEVEL_1). Do not claim unbiased-estimator guarantees.
 */

import { fwhtInPlace } from './hadamard.js';
import { UniformSymmetricCodebook } from './quantizer.js';
import { packValues, unpackValues } from './pack.js';

export interface QJLConfig {
  targetDimensions: number;
  bitsPerValue: number;
  seed: number;
  useHadamard: boolean;
}

export interface QJLCompressed {
  sketch: Uint8Array;
  dimensions: number;
  originalDimensions: number;
  scale: number;
}

export interface QJLStats {
  compressionRatio: number;
  distortion: number;
  sketchSize: number;
}

/**
 * Experimental QJL-like residual sketch research utility.
 *
 * Not paper-exact TurboQuant QJL.
 * Not wired into public MCP compression/search.
 * Does not make unbiased-estimator guarantees.
 */
export class QJLResidualEstimator {
  private config: QJLConfig;
  private projectionMatrix: Float32Array | null = null;
  private codebook: UniformSymmetricCodebook;
  private isInitialized: boolean = false;

  constructor(config?: Partial<QJLConfig>) {
    this.config = {
      targetDimensions: config?.targetDimensions ?? 64,
      bitsPerValue: config?.bitsPerValue ?? 4,
      seed: config?.seed ?? 42,
      useHadamard: config?.useHadamard ?? true
    };
    this.codebook = new UniformSymmetricCodebook(this.config.bitsPerValue);
  }

  /**
   * Initialize projection matrix for QJL.
   */
  initialize(originalDimensions: number): void {
    const { targetDimensions, seed, useHadamard } = this.config;

    // Ensure dimensions are power of 2 for Hadamard
    let paddedDims = originalDimensions;
    if (useHadamard) {
      paddedDims = 1;
      while (paddedDims < originalDimensions) {
        paddedDims *= 2;
      }
    }

    // Generate random projection matrix
    this.projectionMatrix = new Float32Array(paddedDims * targetDimensions);

    // Use simple seeded random for reproducibility
    let state = seed;
    const random = () => {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      return state / 0x7fffffff;
    };

    // Fill with random values scaled by 1/sqrt(targetDimensions)
    const scale = 1 / Math.sqrt(targetDimensions);
    for (let i = 0; i < this.projectionMatrix.length; i++) {
      this.projectionMatrix[i] = (random() * 2 - 1) * scale;
    }

    this.isInitialized = true;
  }

  /**
   * Compress residual vector using QJL.
   */
  compress(residual: Float32Array): QJLCompressed {
    if (!this.isInitialized) {
      this.initialize(residual.length);
    }

    const { targetDimensions, useHadamard } = this.config;
    const originalDimensions = residual.length;

    // Pad if necessary
    let padded = residual;
    if (useHadamard) {
      let paddedDims = 1;
      while (paddedDims < originalDimensions) {
        paddedDims *= 2;
      }
      if (paddedDims > originalDimensions) {
        padded = new Float32Array(paddedDims);
        padded.set(residual);
      }
    }

    // Apply Hadamard transform if enabled
    let processed = new Float32Array(padded);
    if (useHadamard) {
      processed = new Float32Array(padded);
      fwhtInPlace(processed);
    }

    // Project to lower dimensions
    const sketch = new Float32Array(targetDimensions);
    for (let i = 0; i < targetDimensions; i++) {
      let sum = 0;
      for (let j = 0; j < processed.length; j++) {
        sum += processed[j]! * this.projectionMatrix![i * processed.length + j]!;
      }
      sketch[i] = sum;
    }

    // Calculate scale factor for reconstruction
    const maxAbs = Math.max(...Array.from(sketch).map(Math.abs)) || 1;
    const scale = maxAbs > 1 ? 1 / maxAbs : 1;

    // Normalize and quantize
    const normalized = new Float32Array(targetDimensions);
    for (let i = 0; i < targetDimensions; i++) {
      normalized[i] = (sketch[i] ?? 0) * scale;
    }

    // Quantize using uniform codebook
    const quantized = this.codebook.encode(normalized);

    return {
      sketch: quantized,
      dimensions: targetDimensions,
      originalDimensions,
      scale
    };
  }

  /**
   * Reconstruct residual vector from compressed QJL sketch.
   */
  decompress(compressed: QJLCompressed): Float32Array {
    const { targetDimensions, useHadamard } = this.config;
    const { sketch, originalDimensions, scale } = compressed;

    // Decode quantized values
    const normalized = this.codebook.decode(sketch);

    // Denormalize
    const sketchValues = new Float32Array(targetDimensions);
    for (let i = 0; i < targetDimensions; i++) {
      sketchValues[i] = normalized[i]! / scale;
    }

    // Reconstruct using transpose of projection matrix
    let reconstructed = new Float32Array(originalDimensions);
    if (useHadamard && this.projectionMatrix) {
      // For Hadamard-based QJL, reconstruction is simpler
      const paddedDims = this.projectionMatrix.length / targetDimensions;
      const padded = new Float32Array(paddedDims);

      for (let j = 0; j < paddedDims; j++) {
        let sum = 0;
        for (let i = 0; i < targetDimensions; i++) {
          sum += sketchValues[i]! * this.projectionMatrix![i * paddedDims + j]!;
        }
        padded[j] = sum;
      }

      // Apply inverse Hadamard
      fwhtInPlace(padded);

      // Trim to original dimensions
      reconstructed = padded.slice(0, originalDimensions);
    } else {
      // Standard reconstruction
      for (let j = 0; j < originalDimensions; j++) {
        let sum = 0;
        for (let i = 0; i < targetDimensions; i++) {
          sum += sketchValues[i]! * this.projectionMatrix![i * originalDimensions + j]!;
        }
        reconstructed[j] = sum;
      }
    }

    return reconstructed;
  }

  /**
   * Estimate residual and get compression statistics.
   */
  estimateAndCompress(residual: Float32Array): { compressed: QJLCompressed; stats: QJLStats } {
    const compressed = this.compress(residual);

    // Calculate statistics
    const reconstructed = this.decompress(compressed);

    // Calculate distortion (MSE)
    let totalDistortion = 0;
    const n = Math.min(residual.length, reconstructed.length);
    for (let i = 0; i < n; i++) {
      const diff = residual[i]! - reconstructed[i]!;
      totalDistortion += diff * diff;
    }
    const distortion = n > 0 ? totalDistortion / n : 0;

    // Calculate compression ratio
    const originalSize = residual.length * 4; // Float32
    const sketchSize = compressed.sketch.length;
    const compressionRatio = originalSize / sketchSize;

    return {
      compressed,
      stats: {
        compressionRatio,
        distortion,
        sketchSize
      }
    };
  }

  /**
   * Get configuration.
   */
  getConfig(): QJLConfig {
    return { ...this.config };
  }

  /**
   * Check if initialized.
   */
  getIsInitialized(): boolean {
    return this.isInitialized;
  }
}

/**
 * Optimal QJL dimensions calculator.
 * Based on Johnson-Lindenstrauss lemma for given distortion tolerance.
 */
export function optimalQJLDimensions(
  nPoints: number,
  distortion: number = 0.1,
  delta: number = 0.1
): number {
  // JL lemma: k >= 4 * log(n) / (2 * epsilon^2 - epsilon^3)
  // where epsilon is distortion, delta is failure probability
  const epsilon = distortion;
  const k = (4 * Math.log(nPoints / delta)) / (2 * epsilon * epsilon - epsilon * epsilon * epsilon);
  return Math.ceil(k);
}

/**
 * Estimate QJL compression ratio.
 */
export function estimateQJLCompressionRatio(
  originalDimensions: number,
  targetDimensions: number,
  bitsPerValue: number
): number {
  const originalBits = originalDimensions * 32; // Float32
  const compressedBits = targetDimensions * bitsPerValue;
  return originalBits / compressedBits;
}