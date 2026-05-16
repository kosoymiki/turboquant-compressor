#!/usr/bin/env node
/**
 * Gate: forensic redaction removes known secret patterns.
 * Creates a temp file with fixture secrets, runs redact-forensics.mjs, verifies clean.
 */

import { writeFileSync, readFileSync, mkdirSync, rmSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

function fail(msg) {
  console.error(`ERROR [verify-forensics-redaction]: ${msg}`);
  process.exit(1);
}

const SECRET_PATTERNS = [
  /sk-ant-[A-Za-z0-9_-]{10,}/,
  /ghp_[A-Za-z0-9]{10,}/,
  /eyJ[A-Za-z0-9_-]{10,}/,
  /ANTHROPIC_API_KEY\s*=\s*(?!\[REDACTED\])\S+/,
  /nvapi-[A-Za-z0-9_-]{10,}/,
];

const FIXTURES = [
  'ANTHROPIC_API_KEY=sk-ant-api03-testkey1234567890abcdef',
  'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test',
  'GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz123456',
  'NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
  'normal log line: server started on port 3000',
  'another normal line without secrets',
];

// Create temp forensics dir with fixture file
const tmpDir = join(rootDir, 'forensics', '_redaction_test');
mkdirSync(tmpDir, { recursive: true });
const fixtureFile = join(tmpDir, 'test-secrets.txt');
writeFileSync(fixtureFile, FIXTURES.join('\n') + '\n');

// Run redact-forensics.mjs on the temp dir
const redactScript = join(rootDir, 'scripts', 'redact-forensics.mjs');
if (!existsSync(redactScript)) fail('scripts/redact-forensics.mjs not found');

const res = spawnSync(process.execPath, [redactScript, tmpDir], {
  stdio: 'pipe',
  cwd: rootDir,
});

if (res.status !== 0) {
  rmSync(tmpDir, { recursive: true, force: true });
  fail(`redact-forensics.mjs exited ${res.status}: ${res.stderr?.toString().slice(0, 200)}`);
}

// Check redacted file contains no secret patterns
const redacted = readFileSync(fixtureFile, 'utf8');
let violations = 0;
for (const pat of SECRET_PATTERNS) {
  if (pat.test(redacted)) {
    console.error(`  FAIL: secret pattern ${pat} still present after redaction`);
    console.error(`  Content: ${redacted.slice(0, 300)}`);
    violations++;
  }
}

// Cleanup
rmSync(tmpDir, { recursive: true, force: true });

if (violations > 0) fail(`${violations} secret pattern(s) survived redaction`);
console.log('[OK] verify-forensics-redaction: all secret patterns redacted');
