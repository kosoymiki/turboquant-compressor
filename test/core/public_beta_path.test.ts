import { describe, it, expect } from '@jest/globals';
import { compressVectors } from '../../src/tools/compress.js';
import { searchVectors } from '../../src/tools/search.js';

describe('public compression/search path', () => {
  it('uses turboquant_beta by default for 4-bit vectors and preserves it in search metadata', () => {
    const vectors = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
    ];

    const compressed = compressVectors({
      vectors,
      dimensions: 4,
      bitsPerValue: 4,
      seed: 42,
    });

    expect(compressed.codebook_type).toBe('turboquant_beta');

    const search = searchVectors({
      compressed_database_b64: compressed.compressed_database_b64,
      query_vector: [1, 0, 0, 0],
      top_k: 2,
      metric: 'cosine',
    });

    expect(search.codebook_type).toBe('turboquant_beta');
    expect(search.results[0]?.index).toBe(0);
  });

  it('falls back to uniform for 8-bit vectors', () => {
    const compressed = compressVectors({
      vectors: [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
      ],
      dimensions: 4,
      bitsPerValue: 8,
      seed: 7,
    });

    expect(compressed.codebook_type).toBe('uniform');

    const search = searchVectors({
      compressed_database_b64: compressed.compressed_database_b64,
      query_vector: [0, 1, 0, 0],
      top_k: 1,
      metric: 'cosine',
    });

    expect(search.codebook_type).toBe('uniform');
    expect(search.results[0]?.index).toBe(1);
  });
});
