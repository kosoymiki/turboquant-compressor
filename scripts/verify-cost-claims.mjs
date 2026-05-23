#!/usr/bin/env node

import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const FORBIDDEN = [
  /reduces Claude Code token bills/i,
  /saves Claude Code tokens/i,
  /direct Claude Code cost reducer/i,
  /TurboQuant KV cache for Claude Code/i,
  /50[-–]90% savings guaranteed/i,
  /QJL-enabled/i,
  /near-zero loss/i,
  /production TurboQuant/i,
];

const ALLOWED_NEARBY = [
  /does not reduce Anthropic/i,
  /not QJL-enabled/i,
  /no QJL data is stored/i,
  /LEVEL_1_PUBLIC/i,
  /donor claim/i,
  /unverified donor claim/i,
  /must be measured/i,
  /self-hosted/i,
];

function files(root) {
  const out = [];
  function visit(p) {
    if (!existsSync(p)) return;
    const st = statSync(p);
    if (st.isDirectory()) {
      for (const e of readdirSync(p)) visit(join(p, e));
    } else if (/\.(md|ts|js|json)$/.test(p)) {
      out.push(p);
    }
  }
  visit(root);
  return out;
}

function fail(msg) {
  console.error(`ERROR: ${msg}`);
  process.exit(1);
}

for (const file of ['README.md', ...files('docs'), ...files('src')]) {
  if (!existsSync(file)) continue;
  const lines = readFileSync(file, 'utf8').split(/\r?\n/);

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? '';
    for (const pattern of FORBIDDEN) {
      if (!pattern.test(line)) continue;

      const window = [
        lines[i - 2] ?? '',
        lines[i - 1] ?? '',
        line,
        lines[i + 1] ?? '',
        lines[i + 2] ?? '',
      ].join('\n');

      if (!ALLOWED_NEARBY.some((p) => p.test(window))) {
        fail(`unqualified cost/science claim in ${file}:${i + 1}: ${line}`);
      }
    }
  }
}

console.log('[OK] cost/science claims verified');
