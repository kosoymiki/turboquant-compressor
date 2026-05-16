import { describe, it, expect } from '@jest/globals';
import { buildProvenance } from '../../src/context/provenance.js';

describe('buildProvenance', () => {
  it('records all required fields', () => {
    const p = buildProvenance({
      vectorizerId: 'token_hash_fnv1a',
      vectorizerKind: 'token_hash',
      dimensions: 128,
      chunkBytes: 2048,
      bitsPerValue: 4,
      storageMode: 'inline_text',
      rootFingerprint: 'deadbeef',
      totalInputBytes: 4096,
      chunks: [
        { chunkId: 'chunk_0', path: 'src/a.ts', fingerprint: 'fp0', byteOffset: 0, byteLength: 1024, approximateTokens: 200 },
        { chunkId: 'chunk_1', path: 'src/b.ts', fingerprint: 'fp1', byteOffset: 1024, byteLength: 1024, approximateTokens: 180 },
      ],
    });

    expect(p.schemaVersion).toBe(1);
    expect(p.vectorizerId).toBe('token_hash_fnv1a');
    expect(p.vectorizerKind).toBe('token_hash');
    expect(p.dimensions).toBe(128);
    expect(p.chunkBytes).toBe(2048);
    expect(p.bitsPerValue).toBe(4);
    expect(p.storageMode).toBe('inline_text');
    expect(p.rootFingerprint).toBe('deadbeef');
    expect(p.totalInputBytes).toBe(4096);
    expect(p.chunkCount).toBe(2);
    expect(p.chunks).toHaveLength(2);
  });

  it('sets builtAt as valid ISO date', () => {
    const p = buildProvenance({
      vectorizerId: 'token_hash_fnv1a',
      vectorizerKind: 'token_hash',
      dimensions: 64,
      chunkBytes: 4096,
      bitsPerValue: 4,
      storageMode: 'preview_only',
      rootFingerprint: 'abc',
      totalInputBytes: 100,
      chunks: [],
    });
    const d = new Date(p.builtAt);
    expect(d.getTime()).not.toBeNaN();
  });

  it('handles empty chunks', () => {
    const p = buildProvenance({
      vectorizerId: 'token_hash_fnv1a',
      vectorizerKind: 'token_hash',
      dimensions: 32,
      chunkBytes: 256,
      bitsPerValue: 2,
      storageMode: 'external_store',
      rootFingerprint: '',
      totalInputBytes: 0,
      chunks: [],
    });
    expect(p.chunkCount).toBe(0);
    expect(p.chunks).toHaveLength(0);
  });
});
