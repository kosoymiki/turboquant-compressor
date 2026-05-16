import { validateCompressionParams, validateSearchParams, estimateCompressionMemory } from './limits.js';

describe('Limits', () => {
  test('validateCompressionParams accepts valid input', () => {
    expect(() => validateCompressionParams(1024, 128, 4)).not.toThrow();
  });

  test('validateCompressionParams rejects invalid dimensions', () => {
    expect(() => validateCompressionParams(0, 128, 4)).toThrow();
    expect(() => validateCompressionParams(8193, 128, 4)).toThrow();
  });

  test('validateCompressionParams accepts non-power-of-two dimensions (RotationEngine handles padding)', () => {
    expect(() => validateCompressionParams(100, 128, 4)).not.toThrow();
    expect(() => validateCompressionParams(1023, 128, 4)).not.toThrow();
    expect(() => validateCompressionParams(1536, 128, 4)).not.toThrow();
    expect(() => validateCompressionParams(3072, 128, 4)).not.toThrow();
  });

  test('validateCompressionParams rejects invalid bits', () => {
    expect(() => validateCompressionParams(1024, 128, 1)).toThrow();
    expect(() => validateCompressionParams(1024, 128, 16)).toThrow();
  });

  test('validateSearchParams', () => {
    expect(() => validateSearchParams(1024, 5, 100)).not.toThrow();
    expect(() => validateSearchParams(1024, 0, 100)).toThrow();
    expect(() => validateSearchParams(1024, 101, 100)).toThrow();
  });

  test('estimateCompressionMemory', () => {
    const mb = estimateCompressionMemory(1024, 128, 4);
    expect(mb).toBeGreaterThan(0);
  });
});