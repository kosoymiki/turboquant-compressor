import { describe, it, expect } from '@jest/globals';
import { turboquantContextPackBuild } from '../../src/tools/context_pack_build.js';
import { decodeCompressedDatabase } from '../../src/core/format.js';

describe('context_pack_build defaults', () => {
  it('defaults to hashed_tfidf and turboquant_beta for 4-bit packs', () => {
    const result = turboquantContextPackBuild({
      files: [
        { path: 'docs/a.md', text: 'alpha beta gamma' },
        { path: 'src/b.ts', text: 'export const betaGamma = 1;' },
      ],
      dimensions: 16,
      chunkBytes: 256,
      bitsPerValue: 4,
      storageMode: 'preview_only',
    });

    expect(result.manifest.vectorizer.kind).toBe('hashed_tfidf');
    expect(result.manifest.provenance.vectorizerKind).toBe('hashed_tfidf');

    const db = decodeCompressedDatabase(result.manifest.compressedDatabaseB64);
    expect(db.codebookType).toBe('turboquant_beta');
  });
});
