import { describe, it, expect } from '@jest/globals';
import { turboquantContextPackBuild } from '../../src/tools/context_pack_build.js';
import { turboquantContextPackSearch } from '../../src/tools/context_pack_search.js';

describe('context_pack_search externalStore', () => {
  const testFiles = [
    { path: 'test1.md', text: 'First test document with unique content alpha.' },
    { path: 'test2.md', text: 'Second test document with unique content beta.' },
    { path: 'test3.md', text: 'Third test document with unique content gamma.' },
  ];

  it('returns source=external and verified=false without externalStore for external_store manifest', () => {
    const buildResult = turboquantContextPackBuild({
      files: testFiles,
      dimensions: 16,
      chunkBytes: 256,
      bitsPerValue: 4,
      storageMode: 'external_store',
    });

    const manifest = (buildResult as any).manifest;
    const searchResult = turboquantContextPackSearch({
      manifest,
      query: 'unique content',
      top_k: 3,
    });

    expect(searchResult.results).toHaveLength(3);
    searchResult.results.forEach((r: any) => {
      expect(r.source).toBe('external');
      expect(r.verified).toBe(false);
      expect(r.text_available).toBe(false);
      expect(r.warning).toContain('No externalStore provided');
    });
  });

  it('returns source=external and verified=true with matching fingerprint', () => {
    const buildResult = turboquantContextPackBuild({
      files: testFiles,
      dimensions: 16,
      chunkBytes: 256,
      bitsPerValue: 4,
      storageMode: 'external_store',
    });

    const manifest = (buildResult as any).manifest;
    const entries = manifest.chunks.map((chunk: any) => ({
      chunk_id: chunk.id,
      fingerprint: chunk.fingerprint,
      text: chunk.textPreview,
    }));

    const searchResult = turboquantContextPackSearch({
      manifest,
      query: 'unique content',
      top_k: 3,
      externalStore: {
        kind: 'inline_entries',
        entries,
      },
    });

    expect(searchResult.results).toHaveLength(3);
    searchResult.results.forEach((r: any) => {
      expect(r.source).toBe('external');
      expect(r.verified).toBe(true);
      expect(r.text_available).toBe(true);
    });
  });

  it('returns source=external and verified=false with fingerprint mismatch', () => {
    const buildResult = turboquantContextPackBuild({
      files: testFiles,
      dimensions: 16,
      chunkBytes: 256,
      bitsPerValue: 4,
      storageMode: 'external_store',
    });

    const manifest = (buildResult as any).manifest;
    const entries = manifest.chunks.map((chunk: any) => ({
      chunk_id: chunk.id,
      fingerprint: 'WRONG_FINGERPRINT_' + chunk.id,
      text: chunk.textPreview,
    }));

    const searchResult = turboquantContextPackSearch({
      manifest,
      query: 'unique content',
      top_k: 3,
      externalStore: {
        kind: 'inline_entries',
        entries,
      },
    });

    expect(searchResult.results).toHaveLength(3);
    searchResult.results.forEach((r: any) => {
      expect(r.source).toBe('external');
      expect(r.verified).toBe(false);
      expect(r.text_available).toBe(false); // text_available matches verified
      expect(r.warning).toContain('Fingerprint mismatch');
    });
  });

  it('returns source=preview and verified=false for preview_only storage mode', () => {
    const buildResult = turboquantContextPackBuild({
      files: testFiles,
      dimensions: 16,
      chunkBytes: 256,
      bitsPerValue: 4,
      storageMode: 'preview_only',
    });

    const manifest = (buildResult as any).manifest;
    const searchResult = turboquantContextPackSearch({
      manifest,
      query: 'unique content',
      top_k: 3,
    });

    expect(searchResult.results).toHaveLength(3);
    searchResult.results.forEach((r: any) => {
      expect(r.source).toBe('preview');
      expect(r.verified).toBe(false);
      expect(r.text_available).toBe(false);
    });
  });

  it('returns source=inline and verified=true for inline_text storage mode', () => {
    const buildResult = turboquantContextPackBuild({
      files: testFiles,
      dimensions: 16,
      chunkBytes: 256,
      bitsPerValue: 4,
      storageMode: 'inline_text',
    });

    const manifest = (buildResult as any).manifest;
    const searchResult = turboquantContextPackSearch({
      manifest,
      query: 'unique content',
      top_k: 3,
    });

    expect(searchResult.results).toHaveLength(3);
    searchResult.results.forEach((r: any) => {
      expect(r.source).toBe('inline');
      expect(r.verified).toBe(true);
      expect(r.text_available).toBe(true);
    });
  });
});