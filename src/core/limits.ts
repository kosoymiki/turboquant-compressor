/**
 * Mobile constraints and Termux compatibility limits.
 */

import { isPowerOfTwo, nextPowerOfTwo } from './hadamard.js';

export const MAX_DIMENSIONS = 8192;
export const MAX_VECTORS_DEFAULT = 4096;
export const MAX_BITS_PER_VALUE = 8;
export const MIN_BITS_PER_VALUE = 2;

export const TERMUX_DEFAULT_DIMENSIONS = 1024;
export const TERMUX_DEFAULT_VECTORS = 128;
export const TERMUX_MAX_MEMORY_MB = 256;

export function estimateCompressionMemory(
  dimensions: number,
  vectorCount: number,
  bitsPerValue: number
): number {
  const bytesPerVector = Math.ceil(dimensions * bitsPerValue / 8);
  const dataSize = vectorCount * bytesPerVector;
  const overheadFactor = 1.5;
  return Math.ceil(dataSize * overheadFactor / (1024 * 1024));
}

export function validateCompressionParams(
  dimensions: number,
  vectorCount: number,
  bitsPerValue: number
): void {
  if (dimensions < 1 || dimensions > MAX_DIMENSIONS) {
    throw new Error(`Dimensions must be 1-${MAX_DIMENSIONS}, got ${dimensions}`);
  }
  // Removed power-of-two requirement - RotationEngine handles auto-padding
  const paddedDimensions = isPowerOfTwo(dimensions) ? dimensions : nextPowerOfTwo(dimensions);
  if (paddedDimensions > MAX_DIMENSIONS) {
    throw new Error(`Padded dimensions ${paddedDimensions} exceeds maximum ${MAX_DIMENSIONS}`);
  }
  if (vectorCount < 1 || vectorCount > MAX_VECTORS_DEFAULT) {
    throw new Error(`Vector count must be 1-${MAX_VECTORS_DEFAULT}, got ${vectorCount}`);
  }
  if (bitsPerValue < MIN_BITS_PER_VALUE || bitsPerValue > MAX_BITS_PER_VALUE) {
    throw new Error(`Bits per value must be ${MIN_BITS_PER_VALUE}-${MAX_BITS_PER_VALUE}, got ${bitsPerValue}`);
  }
}

export function validateSearchParams(
  dimensions: number,
  k: number,
  maxVectors: number
): void {
  if (k < 1 || k > maxVectors) {
    throw new Error(`k must be 1-${maxVectors}, got ${k}`);
  }
}

