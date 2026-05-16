import { describe, it, expect } from '@jest/globals';
import { InlineResolver, PreviewResolver, ExternalStoreResolver } from '../../src/context/resolver.js';

describe('InlineResolver', () => {
  it('resolves known chunk with matching fingerprint', async () => {
    const r = new InlineResolver([{ id: 'chunk_0', text: 'hello world', fingerprint: 'fp1' }]);
    const result = await r.resolve('chunk_0', 'fp1');
    expect(result.source).toBe('inline');
    expect(result.text).toBe('hello world');
    expect(result.verified).toBe(true);
    expect(result.warning).toBeUndefined();
  });

  it('returns verified=false on fingerprint mismatch', async () => {
    const r = new InlineResolver([{ id: 'chunk_0', text: 'hello', fingerprint: 'fp1' }]);
    const result = await r.resolve('chunk_0', 'wrong');
    expect(result.verified).toBe(false);
    expect(result.warning).toMatch(/fingerprint mismatch/);
  });

  it('returns missing for unknown chunk_id', async () => {
    const r = new InlineResolver([]);
    const result = await r.resolve('chunk_99', 'fp');
    expect(result.source).toBe('missing');
    expect(result.text).toBe('');
    expect(result.verified).toBe(false);
  });
});

describe('PreviewResolver', () => {
  it('returns preview text with source=preview', async () => {
    const r = new PreviewResolver([{ id: 'chunk_0', textPreview: 'first 200 chars...' }]);
    const result = await r.resolve('chunk_0', 'any');
    expect(result.source).toBe('preview');
    expect(result.text).toBe('first 200 chars...');
    expect(result.verified).toBe(false);
    expect(result.warning).toMatch(/preview only/);
  });

  it('returns missing for unknown chunk_id', async () => {
    const r = new PreviewResolver([]);
    const result = await r.resolve('chunk_99', 'fp');
    expect(result.source).toBe('missing');
  });
});

describe('ExternalStoreResolver', () => {
  it('returns empty text with source=external and stable chunk_id in warning', async () => {
    const r = new ExternalStoreResolver();
    const result = await r.resolve('chunk_42', 'fp');
    expect(result.source).toBe('external');
    expect(result.text).toBe('');
    expect(result.verified).toBe(false);
    expect(result.warning).toMatch(/chunk_42/);
    expect(result.warning).toMatch(/external store/);
  });

  it('does not throw on any input', async () => {
    const r = new ExternalStoreResolver();
    await expect(r.resolve('', '')).resolves.toBeDefined();
  });
});
