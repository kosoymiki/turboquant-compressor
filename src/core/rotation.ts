/**
 * Rotation engine using FWHT-based sign flip pattern.
 * Mobile-friendly rotation without floating-point operations.
 */

import { Mulberry32 } from './prng.js';
import { normalizedFwhtInPlace, isPowerOfTwo, nextPowerOfTwo, padVector } from './hadamard.js';

export const FWHT_SIGN = 'FWHT_SIGN';
export const DENSE_QR_DEBUG = 'DENSE_QR_DEBUG';
export const NONE = 'NONE';

export type RotationMode = typeof FWHT_SIGN | typeof DENSE_QR_DEBUG | typeof NONE;

export class RotationEngine {
  private dimensions: number;
  private paddedDimensions: number;
  private seed: number;
  private signFlipPattern: Float32Array;
  private mode: RotationMode;

  constructor(dimensions: number, seed: number = 0, mode: RotationMode = FWHT_SIGN) {
    if (mode === DENSE_QR_DEBUG) {
      throw new Error('DENSE_QR_DEBUG is not implemented in v4.0.0; use FWHT_SIGN or NONE');
    }
    this.dimensions = dimensions;
    this.paddedDimensions = isPowerOfTwo(dimensions) ? dimensions : nextPowerOfTwo(dimensions);
    this.seed = seed;
    this.mode = mode;
    this.signFlipPattern = this.computeSignFlipPattern();
  }

  private computeSignFlipPattern(): Float32Array {
    const prng = new Mulberry32(this.seed);
    const pattern = new Float32Array(this.paddedDimensions);

    if (this.mode === FWHT_SIGN) {
      for (let i = 0; i < this.paddedDimensions; i++) {
        pattern[i] = prng.randomFloat() < 0.5 ? -1 : 1;
      }
    } else {
      for (let i = 0; i < this.paddedDimensions; i++) {
        pattern[i] = prng.gaussian();
      }
    }

    return pattern;
  }

  rotate(vector: Float32Array): Float32Array {
    if (vector.length > this.dimensions) {
      throw new Error(`Vector length ${vector.length} exceeds dimensions ${this.dimensions}`);
    }

    const padded = padVector(vector, this.paddedDimensions);

    if (this.mode === NONE) {
      return padded.slice(0, this.dimensions);
    } else if (this.mode === FWHT_SIGN) {
      for (let i = 0; i < this.paddedDimensions; i++) {
        padded[i]! *= this.signFlipPattern[i]!;
      }
      normalizedFwhtInPlace(padded);
    } else {
      for (let i = 0; i < this.paddedDimensions; i++) {
        let sum = 0;
        for (let j = 0; j < this.paddedDimensions; j++) {
          sum += padded[j]! * this.signFlipPattern[j * this.paddedDimensions + i]!;
        }
        padded[i]! = sum;
      }
    }

    // If padded > original dims, trim and re-normalize to preserve unit-norm property
    if (this.paddedDimensions > this.dimensions) {
      const trimmed = padded.slice(0, this.dimensions);
      let norm2 = 0;
      for (let i = 0; i < this.dimensions; i++) norm2 += trimmed[i]! * trimmed[i]!;
      if (norm2 > 1e-20) {
        const invNorm = 1 / Math.sqrt(norm2);
        for (let i = 0; i < this.dimensions; i++) trimmed[i]! *= invNorm;
      }
      return trimmed;
    }

    return padded;
  }

  inverseRotate(rotated: Float32Array): Float32Array {
    const result = new Float32Array(this.paddedDimensions);

    if (this.mode === NONE) {
      return rotated.slice(0, this.dimensions);
    } else if (this.mode === FWHT_SIGN) {
      // FWHT is its own inverse (up to normalization by n)
      const temp = new Float32Array(rotated);
      normalizedFwhtInPlace(temp);

      // Reverse sign flips
      for (let i = 0; i < this.paddedDimensions; i++) {
        result[i] = temp[i]! * this.signFlipPattern[i]!;
      }
    } else {
      // Use transpose of the matrix
      for (let i = 0; i < this.paddedDimensions; i++) {
        let sum = 0;
        for (let j = 0; j < this.paddedDimensions; j++) {
          sum += rotated[j]! * this.signFlipPattern[i * this.paddedDimensions + j]!;
        }
        result[i] = sum;
      }
    }

    return result.slice(0, this.dimensions);
  }
}