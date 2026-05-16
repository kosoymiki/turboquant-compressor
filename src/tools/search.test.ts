import { searchVectors } from './search.js';
import { compressVectors } from './compress.js';

describe('searchVectors', () => {
  test('should find nearest neighbors', () => {
    const vectors = [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ];
    const compressed = compressVectors({ vectors, bitsPerValue: 4, rotationSeed: 42 });

    const query = [0.9, 0.1, 0.0, 0.0];
    const result = searchVectors(compressed.compressed_database_b64, query, { k: 2, metric: 'cosine' });

    expect(result.results.length).toBe(2);
    expect(result.results[0]!.score).toBeGreaterThanOrEqual(result.results[1]!.score);
    expect(result.vector_count).toBe(4);
  });

  test('should respect k parameter', () => {
    const vectors = [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0]
    ];
    const compressed = compressVectors({ vectors, bitsPerValue: 4, rotationSeed: 42 });

    const query = [0.5, 0.5, 0.0, 0.0];
    const result = searchVectors(compressed.compressed_database_b64, query, { k: 1, metric: 'cosine' });

    expect(result.results.length).toBe(1);
  });

  test('exact vector should rank itself first', () => {
    const vectors = [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ];
    const compressed = compressVectors({ vectors, bitsPerValue: 4, rotationSeed: 42 });

    // Query with exact copy of first vector
    const query = [1.0, 0.0, 0.0, 0.0];
    const result = searchVectors(compressed.compressed_database_b64, query, { k: 4, metric: 'cosine' });

    // First result should be index 0 (the exact match)
    expect(result.results[0]!.index).toBe(0);
    // Score should be maximal (near 1 for cosine similarity)
    expect(result.results[0]!.score).toBeGreaterThan(0.9);
  });

  test('object-form canonical search API should work', () => {
    const vectors = [[1, 0, 0, 0], [0, 1, 0, 0]];
    const compressed = compressVectors({ vectors, dimensions: 4, bitsPerValue: 4, seed: 42 });

    const result = searchVectors({
      compressed_database_b64: compressed.compressed_database_b64,
      query_vector: [1, 0, 0, 0],
      dimensions: 4,
      top_k: 1,
      metric: 'cosine',
    });

    expect(result.results[0]!.index).toBe(0);
  });

  test('dot product self-match should be near exact for 8-bit quantization', () => {
    const vectors = [[2, 0, 0, 0]];
    const compressed = compressVectors({ vectors, dimensions: 4, bitsPerValue: 8, seed: 42 });

    const result = searchVectors({
      compressed_database_b64: compressed.compressed_database_b64,
      query_vector: [2, 0, 0, 0],
      dimensions: 4,
      top_k: 1,
      metric: 'dot',
    });

    expect(result.results[0]!.score).toBeGreaterThan(3.9);
    expect(result.results[0]!.score).toBeLessThan(4.1);
  });

  test('should reject dimensions > 8192', () => {
    const vectors = [[1, 0, 0, 0]];
    const compressed = compressVectors({ vectors, dimensions: 4, bitsPerValue: 4, seed: 42 });

    expect(() => {
      searchVectors({
        compressed_database_b64: compressed.compressed_database_b64,
        query_vector: [1, 0, 0, 0],
        dimensions: 8193,
        top_k: 1,
      });
    }).toThrow();
  });

  test('should reject top_k > 100', () => {
    const vectors = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];
    const compressed = compressVectors({ vectors, dimensions: 4, bitsPerValue: 4, seed: 42 });

    expect(() => {
      searchVectors({
        compressed_database_b64: compressed.compressed_database_b64,
        query_vector: [1, 0, 0, 0],
        dimensions: 4,
        top_k: 101,
      });
    }).toThrow();
  });

  test('should reject k > 100 (legacy alias)', () => {
    const vectors = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];
    const compressed = compressVectors({ vectors, dimensions: 4, bitsPerValue: 4, seed: 42 });

    expect(() => {
      searchVectors({
        compressed_database_b64: compressed.compressed_database_b64,
        query_vector: [1, 0, 0, 0],
        dimensions: 4,
        k: 101,
      });
    }).toThrow();
  });

  test('should emit QJL warning when useQjl is true', () => {
    const vectors = [[1, 0, 0, 0], [0, 1, 0, 0]];
    const compressed = compressVectors({ vectors, dimensions: 4, bitsPerValue: 4, seed: 42 });

    const result = searchVectors({
      compressed_database_b64: compressed.compressed_database_b64,
      query_vector: [1, 0, 0, 0],
      dimensions: 4,
      top_k: 1,
      useQjl: true,
    });

    expect(result.warnings).toContain('useQjl was requested, but public LEVEL_0 databases store no QJL payload and no QJL correction is applied.');
  });

  test('returns sorted top_k results without returning all vectors', () => {
    const compressed = compressVectors({
      vectors: [
        [1, 0, 0, 0],
        [0.9, 0.1, 0, 0],
        [0, 1, 0, 0],
      ],
      dimensions: 4,
    });

    const result = searchVectors({
      compressed_database_b64: compressed.compressed_database_b64,
      query_vector: [1, 0, 0, 0],
      top_k: 2,
      metric: 'cosine',
    });

    expect(result.results).toHaveLength(2);
    expect(result.results[0]!.score).toBeGreaterThanOrEqual(result.results[1]!.score);
  });
});