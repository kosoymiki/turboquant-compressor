#!/usr/bin/env node
/**
 * Gate: no vitest imports or expect(true).toBe(true) placeholders in test files.
 */

import { readdirSync, readFileSync, statSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

function fail(msg) {
  console.error(`ERROR [verify-no-placeholders]: ${msg}`);
  process.exit(1);
}

function walk(dir, results = []) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) walk(full, results);
    else if (entry.endsWith('.test.ts') || entry.endsWith('.test.js')) results.push(full);
  }
  return results;
}

const testDir = join(rootDir, 'test');
const srcDir = join(rootDir, 'src');
const testFiles = [...walk(testDir), ...walk(srcDir)];

const PLACEHOLDER_PATTERNS = [
  { pat: /from ['"]vitest['"]/, label: 'vitest import' },
  { pat: /expect\(true\)\.toBe\(true\)/, label: 'expect(true).toBe(true) placeholder' },
  { pat: /it\.todo\(/, label: 'it.todo placeholder' },
  { pat: /test\.todo\(/, label: 'test.todo placeholder' },
  { pat: /\/\/ TODO.*test/i, label: 'TODO test comment' },
];

let violations = 0;
for (const file of testFiles) {
  const src = readFileSync(file, 'utf8');
  for (const { pat, label } of PLACEHOLDER_PATTERNS) {
    if (pat.test(src)) {
      console.error(`  FAIL ${file}: contains ${label}`);
      violations++;
    }
  }
}

if (violations > 0) {
  fail(`${violations} placeholder violation(s) found in test files`);
}

console.log(`[OK] verify-no-placeholders: ${testFiles.length} test files clean`);
