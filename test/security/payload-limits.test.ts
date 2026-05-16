import { describe, it, expect } from '@jest/globals';
import { parseCompressInput, parseSearchInput } from '../../src/tools/validation.js';

describe('payload-limits: compress input', () => {
  it('rejects empty vectors array', () => {
    expect(() => parseCompressInput({ vectors: [] })).toThrow();
  });

  it('rejects non-finite values', () => {
    expect(() => parseCompressInput({ vectors: [[Infinity, 0]] })).toThrow();
    expect(() => parseCompressInput({ vectors: [[NaN, 0]] })).toThrow();
  });

  it('rejects non-rectangular vectors', () => {
    expect(() => parseCompressInput({ vectors: [[1, 2], [3]] })).toThrow(/rectangular/);
  });

  it('rejects invalid bitsPerValue', () => {
    expect(() => parseCompressInput({ vectors: [[1, 0]], bitsPerValue: 5 as never })).toThrow();
  });

  it('rejects dimensions mismatch', () => {
    expect(() => parseCompressInput({ vectors: [[1, 0, 0]], dimensions: 2 })).toThrow(/dimensions/);
  });
});

describe('payload-limits: search input', () => {
  it('rejects missing compressed_database_b64', () => {
    expect(() => parseSearchInput({ query_vector: [1, 0] })).toThrow();
  });

  it('rejects missing query_vector', () => {
    expect(() => parseSearchInput({ compressed_database_b64: 'AAAA' })).toThrow();
  });

  it('rejects invalid base64', () => {
    expect(() => parseSearchInput({
      compressed_database_b64: '!!!invalid!!!',
      query_vector: [1, 0],
    })).toThrow(/base64/);
  });

  it('rejects non-finite query values', () => {
    expect(() => parseSearchInput({
      compressed_database_b64: 'AAAA',
      query_vector: [Infinity],
    })).toThrow();
  });

  it('rejects top_k above 100', () => {
    expect(() => parseSearchInput({
      compressed_database_b64: 'AAAA',
      query_vector: [1, 0],
      top_k: 101,
    })).toThrow();
  });
});
