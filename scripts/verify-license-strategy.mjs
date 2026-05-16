#!/usr/bin/env node
/**
 * Verifies license strategy compliance.
 */

import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

function fail(msg) {
  console.error(`ERROR: ${msg}`);
  process.exit(1);
}

// 1. Check LICENSE file
const licensePath = join(rootDir, 'LICENSE');
if (!existsSync(licensePath)) {
  fail('LICENSE file not found');
}
const license = readFileSync(licensePath, 'utf8');
if (!license.includes('GNU GENERAL PUBLIC LICENSE') || !license.includes('Version 3')) {
  fail('LICENSE must be GPL-3.0-or-later');
}

// 2. Check NOTICE file
const noticePath = join(rootDir, 'NOTICE');
if (!existsSync(noticePath)) {
  fail('NOTICE file not found');
}
const notice = readFileSync(noticePath, 'utf8');
if (!notice.includes('0xSero') && !notice.includes('turboquant')) {
  console.warn('[WARN] NOTICE should attribute 0xSero turboquant donor');
}

// 3. Check package.json license
const pkgPath = join(rootDir, 'package.json');
const pkg = JSON.parse(readFileSync(pkgPath, 'utf8'));
if (pkg.license !== 'GPL-3.0-or-later') {
  fail(`package.json license must be GPL-3.0-or-later, got: ${pkg.license}`);
}

// 4. Check LICENSE_STRATEGY.md exists
const strategyPath = join(rootDir, 'docs', 'LICENSE_STRATEGY.md');
if (!existsSync(strategyPath)) {
  fail('docs/LICENSE_STRATEGY.md not found');
}

console.log('[OK] License strategy verified:');
console.log('  - LICENSE: GPL-3.0-or-later');
console.log('  - NOTICE: present');
console.log('  - package.json: GPL-3.0-or-later');
console.log('  - docs/LICENSE_STRATEGY.md: present');
