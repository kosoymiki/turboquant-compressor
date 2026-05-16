import {
  isPowerOfTwo, nextPowerOfTwo, fwhtInPlace, normalizedFwhtInPlace,
  padVector, l2Norm, normalizeVector
} from './hadamard.js';

describe('Hadamard', () => {
  test('isPowerOfTwo', () => {
    expect(isPowerOfTwo(1)).toBe(true);
    expect(isPowerOfTwo(2)).toBe(true);
    expect(isPowerOfTwo(4)).toBe(true);
    expect(isPowerOfTwo(8)).toBe(true);
    expect(isPowerOfTwo(3)).toBe(false);
    expect(isPowerOfTwo(0)).toBe(false);
    expect(isPowerOfTwo(16)).toBe(true);
  });

  test('nextPowerOfTwo', () => {
    expect(nextPowerOfTwo(1)).toBe(1);
    expect(nextPowerOfTwo(2)).toBe(2);
    expect(nextPowerOfTwo(3)).toBe(4);
    expect(nextPowerOfTwo(5)).toBe(8);
    expect(nextPowerOfTwo(8)).toBe(8);
    expect(nextPowerOfTwo(0)).toBe(1);
  });

  test('fwhtInPlace', () => {
    const data = new Float32Array([1, 0, 0, 0]);
    fwhtInPlace(data);
    expect(Array.from(data)).toEqual([1, 1, 1, 1]);
  });

  test('normalizedFwhtInPlace', () => {
    const data = new Float32Array([1, 0, 0, 0]);
    normalizedFwhtInPlace(data);
    const expected = [0.5, 0.5, 0.5, 0.5];
    data.forEach((v, i) => expect(Math.abs(v - expected[i]!)).toBeLessThan(0.0001));
  });

  test('padVector', () => {
    expect(Array.from(padVector(new Float32Array([1, 2, 3]), 8))).toEqual([1, 2, 3, 0, 0, 0, 0, 0]);
  });

  test('l2Norm', () => {
    expect(l2Norm(new Float32Array([3, 4]))).toBe(5);
    expect(l2Norm(new Float32Array([1, 1, 1, 1]))).toBe(2);
  });

  test('normalizeVector', () => {
    const normalized = normalizeVector(new Float32Array([3, 4]));
    expect(Math.abs(normalized[0]! - 0.6)).toBeLessThan(0.0001);
    expect(Math.abs(normalized[1]! - 0.8)).toBeLessThan(0.0001);
  });
});