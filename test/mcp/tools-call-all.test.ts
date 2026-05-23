import { describe, it, expect } from '@jest/globals';
import { spawnSync } from 'child_process';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '../..');
const conformanceScript = join(rootDir, 'scripts', 'mcp-conformance.mjs');
const serverPath = join(rootDir, 'dist', 'server.js');

describe('tools-call-all: the full MCP tool surface must succeed via conformance harness', () => {
  it('dist/server.js exists (run npm run build first)', () => {
    expect(existsSync(serverPath)).toBe(true);
  });

  it('mcp-conformance.mjs exits 0 — all tools/call paths pass', () => {
    const result = spawnSync('node', [conformanceScript], {
      cwd: rootDir,
      encoding: 'utf8',
      timeout: 60_000,
    });

    if (result.error) {
      throw new Error(`Failed to spawn conformance harness: ${result.error.message}`);
    }

    const output = (result.stdout ?? '') + (result.stderr ?? '');

    // Surface conformance output on failure for easy diagnosis
    if (result.status !== 0) {
      console.error('=== mcp-conformance.mjs output ===\n' + output);
    }

    expect(result.status).toBe(0);
  }, 60_000);
});
