#!/usr/bin/env node
/**
 * Gate: mcp-conformance.mjs must be pure JS (no TS annotations) and server must be built.
 */

import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

function fail(msg) {
  console.error(`ERROR [verify-mcp-conformance]: ${msg}`);
  process.exit(1);
}

// 1. conformance script must exist
const conformancePath = join(rootDir, 'scripts', 'mcp-conformance.mjs');
if (!existsSync(conformancePath)) {
  fail('scripts/mcp-conformance.mjs not found');
}

// 2. must not contain TypeScript type annotations
const src = readFileSync(conformancePath, 'utf8');
const tsPatterns = [
  /:\s*number\b/,
  /:\s*string\b/,
  /:\s*any\b/,
  /:\s*boolean\b/,
  /:\s*void\b/,
  /Promise<[^>]+>/,
  /catch\s*\(\s*\w+\s*:\s*\w+\s*\)/,
];
for (const pat of tsPatterns) {
  if (pat.test(src)) {
    fail(`mcp-conformance.mjs contains TypeScript annotation matching ${pat} — must be pure JS`);
  }
}

// 3. dist/server.js must exist (server must be built)
const serverPath = join(rootDir, 'dist', 'server.js');
if (!existsSync(serverPath)) {
  fail('dist/server.js not found — run npm run build before release');
}

// 4. conformance script must reference all 13 expected tools
const EXPECTED_TOOLS = [
  'turboquant_compress',
  'turboquant_vector_search',
  'turboquant_cost_analyze',
  'turboquant_cache_plan',
  'turboquant_prompt_cache_lint',
  'turboquant_context_pack_build',
  'turboquant_context_pack_search',
  'turboquant_cli_mcp_profile',
  'turboquant_quantize',
  'turboquant_kv_analyze',
  'turboquant_backend_probe',
  'turboquant_opencl_probe',
  'turboquant_adreno_loader_probe',
];
for (const tool of EXPECTED_TOOLS) {
  if (!src.includes(tool)) {
    fail(`mcp-conformance.mjs does not reference tool: ${tool}`);
  }
}

console.log('[OK] verify-mcp-conformance: pure JS, server built, all 13 tools referenced');
