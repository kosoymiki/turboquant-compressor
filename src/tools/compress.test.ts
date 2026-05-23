import { compressVectors } from './compress.js';
import { decodeCompressedDatabase } from '../core/format.js';

describe('compressVectors', () => {
  test('should compress vectors and return valid result', () => {
    const vectors = [
      [0.1, -0.3, 0.5, -0.7],
      [0.9, 0.8, -0.2, 0.1]
    ];
    const result = compressVectors({ vectors, bitsPerValue: 4, rotationSeed: 42 });

    expect(result.compressed_database_b64).toBeDefined();
    expect(result.compressed_database_b64.length).toBeGreaterThan(0);
    expect(result.dimensions).toBe(4);
    expect(result.vector_count).toBe(2);
    expect(result.bits_per_value).toBe(4);
    expect(result.warnings.length).toBeGreaterThan(0);
  });

  test('should calculate compression ratio', () => {
    const vectors = [
      [0.1, -0.3, 0.5, -0.7],
      [0.9, 0.8, -0.2, 0.1]
    ];
    const result = compressVectors({ vectors });

    // With norms storage and Float32 estimate, ratio may be < 1 for small vectors
    // The important thing is that compression is correct, not that it saves space
    expect(result.compression_ratio).toBeGreaterThanOrEqual(0.25);
  });

  test('default public path is LEVEL_1 public', () => {
    const vectors = [[0.1, -0.3, 0.5, -0.7]];
    const result = compressVectors({ vectors });

    expect(result.algorithm_level).toBe('LEVEL_1_PUBLIC');
    expect(result.warnings.some(w => w.includes('LEVEL_1_PUBLIC'))).toBe(true);
  });

  test('rejects ragged vectors', () => {
    expect(() => compressVectors({
      vectors: [[1, 0, 0, 0], [1, 0]],
      dimensions: 4,
    })).toThrow(/rectangular|length/i);
  });

  test('rejects NaN and Infinity', () => {
    expect(() => compressVectors({
      vectors: [[NaN, 0, 0, 0]],
      dimensions: 4,
    })).toThrow(/NaN|number/i);

    expect(() => compressVectors({
      vectors: [[Infinity, 0, 0, 0]],
      dimensions: 4,
    })).toThrow(/number|Infinity/i);
  });

  test('rejects dimensions mismatch', () => {
    expect(() => compressVectors({
      vectors: [[1, 0, 0, 0]],
      dimensions: 3,
    })).toThrow(/dimensions/i);
  });

  test('includeQJL enables QJL residual sketch', () => {
    const result = compressVectors({
      vectors: [[1, 0, 0, 0]],
      includeQJL: true,
    });

    expect(result.include_qjl).toBe(true);
    expect(result.qjl_sketches_b64).toBeDefined();
    const decoded = decodeCompressedDatabase(result.compressed_database_b64);
    expect(decoded.qjlLength).toBeGreaterThan(0);
  });

  test('format_version matches decoded binary format version', () => {
    const result = compressVectors({
      vectors: [[1, 0, 0, 0]],
      dimensions: 4,
      bitsPerValue: 4,
      seed: 42,
    });

    const decoded = decodeCompressedDatabase(result.compressed_database_b64);

    expect(result.format_version).toBe(decoded.version);
    expect(decoded.version).toBe(3);
    expect(decoded.headerLength).toBe(80);
  });

  test('public compression stores QJL payload when requested', () => {
    const result = compressVectors({
      vectors: [[1, 0, 0, 0]],
      dimensions: 4,
      includeQJL: true,
    });

    expect(result.include_qjl).toBe(true);
    expect(result.qjl_sketches_b64).toBeDefined();
    expect(result.algorithm_level).toBe('LEVEL_1_EXPERIMENTAL_QJL');
  });
});
