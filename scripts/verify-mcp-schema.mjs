#!/usr/bin/env node

import { spawnSync } from 'node:child_process';

const res = spawnSync(process.execPath, ['scripts/smoke-mcp.mjs'], {
  stdio: 'inherit'
});

process.exit(res.status ?? 1);