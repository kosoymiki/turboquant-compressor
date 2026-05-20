#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

function fail(msg) {
  console.error(`ERROR: ${msg}`);
  process.exit(1);
}

function run(script) {
  const res = spawnSync(process.execPath, [`scripts/${script}`], {
    stdio: 'inherit',
    cwd: rootDir,
  });
  if (res.status !== 0) process.exit(res.status ?? 1);
}

// Static verification gates
const staticChecks = [
  'verify-package-json.mjs',
  'verify-format-contract.mjs',
  'verify-scientific-claims.mjs',
  'verify-tests-determinism.mjs',
  'verify-mcp-schema.mjs',
  'verify-cost-claims.mjs',
  'verify-mcp-conformance.mjs',
  'verify-no-placeholders.mjs',
  'verify-host-matrix.mjs',
  'verify-tool-metadata.mjs',
  'verify-no-shell-executor.mjs',
  'verify-forensics-redaction.mjs',
  'verify-termux-ready.mjs',
  'verify-opencl-claims.mjs',
  'verify-release-evidence.mjs',
];

const requiredWasmAssets = [
  'dist/native/wasm/pkg/turboquant_wasm_bg.wasm',
  'dist/native/wasm/pkg/turboquant_wasm.js',
];
for (const rel of requiredWasmAssets) {
  const abs = join(rootDir, rel);
  if (!existsSync(abs)) fail(`required WASM release asset missing: ${rel}`);
}

for (const check of staticChecks) {
  run(check);
}

// Live MCP conformance — actually executes the harness
console.log('[GATE] running live mcp-conformance.mjs...');
run('mcp-conformance.mjs');
console.log('[OK] live mcp-conformance passed');

// Live transcript — verifies all 13 tools callable with isError=false
console.log('[GATE] running live mcp-transcript.mjs...');
run('mcp-transcript.mjs');

// Validate transcript output
const transcriptPath = join(rootDir, 'forensics', 'mcp-stdio-transcript.jsonl');
if (!existsSync(transcriptPath)) {
  fail('mcp-stdio-transcript.jsonl not found after transcript run');
}
const lines = readFileSync(transcriptPath, 'utf8').trim().split('\n').filter(Boolean);
const entries = lines.map((l) => JSON.parse(l));

const isErrorEntries = entries.filter(
  (e) => e.phase?.startsWith('tools/call') && e.response?.result?.isError === true
);
if (isErrorEntries.length > 0) {
  for (const e of isErrorEntries) {
    const text = e.response?.result?.content?.[0]?.text ?? '';
    console.error(`  isError=true in ${e.phase}: ${text.slice(0, 200)}`);
  }
  fail(`transcript contains ${isErrorEntries.length} isError=true tool call(s)`);
}

const timeoutEntries = entries.filter((e) => e.error?.includes('Timeout'));
if (timeoutEntries.length > 0) {
  fail(`transcript contains ${timeoutEntries.length} timeout(s)`);
}

console.log(`[OK] transcript verified: ${entries.length} entries, 0 isError, 0 timeouts`);
console.log('[OK] release verification suite passed');
