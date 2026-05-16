#!/usr/bin/env node

import { execSync } from 'node:child_process';

const out = execSync(
  `grep -RIn "Math\\.random" src --include="*.test.ts" || true`,
  { encoding: 'utf8' }
).trim();

if (out) {
  console.error('ERROR: Math.random is forbidden in tests. Use deterministic seeded PRNG.');
  console.error(out);
  process.exit(1);
}

console.log('[OK] tests are deterministic');