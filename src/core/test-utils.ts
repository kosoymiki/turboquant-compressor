/**
 * Deterministic test utilities using seeded PRNG.
 */

import { Mulberry32 } from './prng.js';

export function deterministicFloatArray(
  length: number,
  seed: number = 123,
  min: number = -1,
  max: number = 1
): Float32Array {
  const rng = new Mulberry32(seed);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = min + (max - min) * rng.randomFloat();
  }
  return out;
}

export function deterministicIntArray(
  length: number,
  maxExclusive: number,
  seed: number = 123
): number[] {
  const rng = new Mulberry32(seed);
  return Array.from({ length }, () => Math.floor(rng.randomFloat() * maxExclusive));
}

export function deterministicNumber(
  seed: number = 123,
  min: number = 0,
  max: number = 1
): number {
  const rng = new Mulberry32(seed);
  return min + (max - min) * rng.randomFloat();
}