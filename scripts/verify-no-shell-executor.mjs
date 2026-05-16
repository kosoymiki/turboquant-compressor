#!/usr/bin/env node
/**
 * Gate: server source must not contain exec/spawn calls (no shell executor).
 * Scripts directory is excluded — only src/ and dist/ are checked.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

function fail(msg) {
  console.error(`ERROR [verify-no-shell-executor]: ${msg}`);
  process.exit(1);
}

function walk(dir, results = []) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) walk(full, results);
    else if (entry.endsWith('.js') || entry.endsWith('.ts')) results.push(full);
  }
  return results;
}

const SHELL_PATTERNS = [
  /\bexecSync\b/,
  /\bspawnSync\b/,
  /\bexec\s*\(/,
  /\bspawn\s*\(/,
  /child_process/,
  /shelljs/,
];

// Only check server source — not scripts (which legitimately use spawn)
// backend_probe.ts is excluded: it requires child_process by design (runtime detection)
const ALLOWED_SHELL_FILES = ['backend_probe.ts', 'backend_probe.js', 'opencl_probe.ts', 'opencl_probe.js', 'turboquant_adreno_loader_probe.ts', 'turboquant_adreno_loader_probe.js'];

const srcDir = join(rootDir, 'src');
const distDir = join(rootDir, 'dist');
const files = [...walk(srcDir), ...walk(distDir)];

let violations = 0;
for (const file of files) {
  const basename = file.split('/').pop();
  if (ALLOWED_SHELL_FILES.includes(basename)) continue;
  const src = readFileSync(file, 'utf8');
  for (const pat of SHELL_PATTERNS) {
    if (pat.test(src)) {
      console.error(`  FAIL ${file.replace(rootDir + '/', '')}: contains shell executor pattern: ${pat}`);
      violations++;
    }
  }
}

if (violations > 0) fail(`${violations} shell executor pattern(s) found in server source`);
console.log(`[OK] verify-no-shell-executor: ${files.length} files scanned, no shell executor`);
