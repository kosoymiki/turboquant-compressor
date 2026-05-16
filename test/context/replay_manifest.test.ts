import { describe, it, expect } from '@jest/globals';
import { buildProvenance } from '../../src/context/provenance.js';
import { buildReplayManifest, verifyReplayManifest } from '../../src/context/replay_manifest.js';

const baseProvenance = buildProvenance({
  vectorizerId: 'token_hash_fnv1a',
  vectorizerKind: 'token_hash',
  dimensions: 64,
  chunkBytes: 4096,
  bitsPerValue: 4,
  storageMode: 'preview_only',
  rootFingerprint: 'abc123',
  totalInputBytes: 1024,
  chunks: [
    { chunkId: 'chunk_0', path: 'a.ts', fingerprint: 'fp0', byteOffset: 0, byteLength: 512, approximateTokens: 100 },
  ],
});

describe('buildProvenance', () => {
  it('sets schemaVersion=1', () => {
    expect(baseProvenance.schemaVersion).toBe(1);
  });

  it('records vectorizer metadata', () => {
    expect(baseProvenance.vectorizerId).toBe('token_hash_fnv1a');
    expect(baseProvenance.vectorizerKind).toBe('token_hash');
  });

  it('records chunk count', () => {
    expect(baseProvenance.chunkCount).toBe(1);
    expect(baseProvenance.chunks).toHaveLength(1);
  });

  it('sets builtAt as ISO string', () => {
    expect(() => new Date(baseProvenance.builtAt)).not.toThrow();
  });
});

describe('buildReplayManifest', () => {
  const manifest = buildReplayManifest({
    provenance: baseProvenance,
    compressedDatabaseB64: 'AAAA',
  });

  it('sets schemaVersion=1', () => {
    expect(manifest.schemaVersion).toBe(1);
  });

  it('includes replayCommand with key params', () => {
    expect(manifest.replayCommand).toMatch(/--dimensions 64/);
    expect(manifest.replayCommand).toMatch(/--bits-per-value 4/);
    expect(manifest.replayCommand).toMatch(/token_hash_fnv1a/);
  });
});

describe('verifyReplayManifest', () => {
  const manifest = buildReplayManifest({
    provenance: baseProvenance,
    compressedDatabaseB64: 'AAAA',
  });

  it('passes with correct fingerprint', () => {
    const r = verifyReplayManifest(manifest, 'abc123');
    expect(r.ok).toBe(true);
    expect(r.error).toBeUndefined();
  });

  it('fails with wrong fingerprint', () => {
    const r = verifyReplayManifest(manifest, 'wrong');
    expect(r.ok).toBe(false);
    expect(r.error).toMatch(/rootFingerprint mismatch/);
  });

  it('fails with empty compressedDatabaseB64', () => {
    const m = buildReplayManifest({ provenance: baseProvenance, compressedDatabaseB64: '' });
    const r = verifyReplayManifest(m, 'abc123');
    expect(r.ok).toBe(false);
    expect(r.error).toMatch(/empty/);
  });
});
