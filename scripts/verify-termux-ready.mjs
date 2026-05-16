#!/usr/bin/env node
/**
 * verify-termux-ready: proves quick backend probe works without native deps,
 * no top-level torch/triton/vllm imports in runtime code.
 */

import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

function fail(msg) {
  console.error(`ERROR [verify-termux-ready]: ${msg}`);
  process.exit(1);
}

// 1. No top-level torch/triton/vllm imports in backend_probe
const probeSource = readFileSync(join(rootDir, 'src/runtime/backend_probe.ts'), 'utf8');
// Only match real TS/JS import statements, not string literals inside spawnSync args
const forbidden = [/^import\s.*from\s+['"]torch['"]/, /^import\s.*from\s+['"]triton['"]/, /^import\s.*from\s+['"]vllm['"]/];
for (const pat of forbidden) {
  if (pat.test(probeSource)) {
    fail(`backend_probe.ts has forbidden top-level import: ${pat}`);
  }
}

// 2. Quick probe works via dist
const res = spawnSync('node', [
  '--input-type=module',
  '-e',
  `import { probeBackends } from './dist/runtime/backend_probe.js';
   const r = probeBackends({ deep: false, timeoutMs: 250 });
   if (r.probeMode !== 'quick') throw new Error('not quick mode');
   if (r.recommendedBackend !== 'typescript_cpu') throw new Error('unexpected backend: ' + r.recommendedBackend);
   console.log(JSON.stringify(r));`
], {
  encoding: 'utf8',
  cwd: rootDir,
  timeout: 5000,
  stdio: ['ignore', 'pipe', 'pipe'],
});

if (res.error || res.status !== 0) {
  fail(`Quick probe failed: ${res.stderr || res.error}`);
}

const parsed = JSON.parse(res.stdout.trim());
if (parsed.elapsedMs > 3000) {
  fail(`Quick probe too slow: ${parsed.elapsedMs}ms`);
}

console.log(`[OK] verify-termux-ready: quick probe ${parsed.elapsedMs}ms, backend=${parsed.recommendedBackend}, no forbidden imports`);
