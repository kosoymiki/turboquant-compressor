/**
 * Fast Walsh-Hadamard Transform implementation.
 * Mobile-friendly O(n log n) rotation without floating-point operations.
 */

export function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

export function nextPowerOfTwo(n: number): number {
  if (n <= 0) return 1;
  if (isPowerOfTwo(n)) return n;
  return Math.pow(2, Math.floor(Math.log2(n)) + 1);
}

export function fwhtInPlace(data: Float32Array): void {
  const n = data.length;
  if (!isPowerOfTwo(n)) {
    throw new Error(`FWHT requires power of two length, got ${n}`);
  }

  let len = 1;
  while (len < n) {
    for (let i = 0; i < n; i += len * 2) {
      for (let j = 0; j < len; j++) {
        const u = data[i + j]!;
        const v = data[i + j + len]!;
        data[i + j] = u + v;
        data[i + j + len] = u - v;
      }
    }
    len *= 2;
  }
}

export function normalizedFwhtInPlace(data: Float32Array): void {
  const n = data.length;
  fwhtInPlace(data);
  const scale = 1 / Math.sqrt(n);
  for (let i = 0; i < n; i++) {
    data[i]! *= scale;
  }
}

export function padVector(vector: Float32Array, targetLength: number): Float32Array {
  if (vector.length > targetLength) {
    throw new Error(`Vector length ${vector.length} exceeds target ${targetLength}`);
  }
  const padded = new Float32Array(targetLength);
  for (let i = 0; i < vector.length; i++) {
    const val = vector[i];
    if (val !== undefined) {
      padded[i] = val;
    }
  }
  return padded;
}

export function l2Norm(vector: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < vector.length; i++) {
    const v = vector[i];
    if (v !== undefined) {
      sum += v * v;
    }
  }
  return Math.sqrt(sum);
}

export function normalizeVector(vector: Float32Array): Float32Array {
  const norm = l2Norm(vector);
  if (norm === 0) return vector;
  const normalized = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i++) {
    const val = vector[i];
    if (val !== undefined) {
      normalized[i] = val / norm;
    }
  }
  return normalized;
}