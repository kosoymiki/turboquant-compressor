import { describe, it, expect } from '@jest/globals';
import { compressVectors } from '../../src/tools/compress.js';
import { turboquantContextPackBuild } from '../../src/tools/context_pack_build.js';

const SECRET_PATTERNS = [
  /sk-[A-Za-z0-9_-]{20,}/,
  /ant-[A-Za-z0-9_-]{20,}/,
  /ghp_[A-Za-z0-9]{36,}/,
  /ANTHROPIC_API_KEY\s*=\s*\S+/i,
  /Authorization:\s*Bearer/i,
];

function containsSecret(text: string): boolean {
  return SECRET_PATTERNS.some((p) => p.test(text));
}

describe('no-secret-output: compress tool', () => {
  it('compress output contains no secret patterns', () => {
    const result = compressVectors({ vectors: [[1, 0, 0, 0], [0, 1, 0, 0]] });
    const serialized = JSON.stringify(result);
    expect(containsSecret(serialized)).toBe(false);
  });

  it('compress warnings do not contain secrets', () => {
    const result = compressVectors({ vectors: [[1, 0]], includeQJL: true });
    for (const w of result.warnings ?? []) {
      expect(containsSecret(w)).toBe(false);
    }
  });
});

describe('no-secret-output: context pack build', () => {
  it('context pack build output contains no secret patterns', () => {
    const result = turboquantContextPackBuild({
      files: [{ path: 'test.ts', text: 'export function foo() {}' }],
      dimensions: 32,
    });
    const serialized = JSON.stringify(result);
    expect(containsSecret(serialized)).toBe(false);
  });
});
