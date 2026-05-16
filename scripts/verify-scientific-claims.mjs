#!/usr/bin/env node

import { readFileSync } from 'node:fs';

const forbidden = [
  /paper-exact/i,
  /near-zero loss/i,
  /\bzero loss\b/i,
  /QJL-enabled/i,
  /unbiased QJL/i,
  /optimal codebooks/i,
  /production TurboQuant/i
];

const allowedLineContexts = [
  /not paper-exact/i,
  /not QJL-enabled/i,
  /not guaranteed/i,
  /research-only/i,
  /experimental/i,
  /no QJL data is stored/i,
  /does not use/i,
  /do not use/i
];

const files = [
  'README.md',
  'docs/FORMAT.md',
  'docs/ARCHITECTURE.md',
  'docs/PERFORMANCE.md',
  'docs/SCIENTIFIC_LIMITATIONS.md',
  'docs/QJL_ROADMAP.md',
  'package.json'
];

function fail(msg) {
  console.error(`ERROR: ${msg}`);
  process.exit(1);
}

for (const file of files) {
  let lines;
  try {
    lines = readFileSync(file, 'utf8').split(/\r?\n/);
  } catch {
    continue;
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? '';
    for (const pattern of forbidden) {
      if (!pattern.test(line)) continue;

      const window = [
        lines[i - 2] ?? '',
        lines[i - 1] ?? '',
        line,
        lines[i + 1] ?? '',
        lines[i + 2] ?? '',
      ].join('\n');

      const allowed = allowedLineContexts.some((allowedPattern) => allowedPattern.test(window));
      if (!allowed) {
        fail(`forbidden scientific claim in ${file}:${i + 1}: ${line}`);
      }
    }
  }
}

console.log('[OK] scientific claims verified');