#!/usr/bin/env node

import { spawnSync } from 'node:child_process';

const scripts = ['smoke:api', 'smoke:mcp', 'smoke:numeric'];

for (const script of scripts) {
  console.log(`[SMOKE] Running npm run ${script}`);
  const result = spawnSync('npm', ['run', script], {
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });

  if (result.status !== 0) {
    console.error(`[SMOKE] ${script} failed`);
    process.exit(result.status ?? 1);
  }
}

console.log('[SMOKE] All smoke checks passed');