#!/usr/bin/env node

import { cpSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const srcDir = join(rootDir, 'native', 'wasm', 'pkg');
const dstDir = join(rootDir, 'dist', 'native', 'wasm', 'pkg');

if (!existsSync(srcDir)) {
  console.error(`[copy-wasm-assets] source directory missing: ${srcDir}`);
  process.exit(1);
}

mkdirSync(dstDir, { recursive: true });
cpSync(srcDir, dstDir, { recursive: true });
console.log(`[copy-wasm-assets] copied ${srcDir} -> ${dstDir}`);
