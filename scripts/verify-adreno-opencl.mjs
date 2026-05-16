#!/usr/bin/env node
/**
 * Gate: verify OpenCL/Adreno readiness.
 * Passes if OpenCL available and probe succeeds, OR if --allow-unavailable and OpenCL missing.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { execSync, spawnSync } from 'child_process';
import { existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

const allowUnavailable = process.argv.includes('--allow-unavailable');

function fail(msg) {
  console.error(`FAIL [verify-adreno-opencl]: ${msg}`);
  process.exit(1);
}

// 1. Check dist built
const distProbe = join(rootDir, 'dist', 'native', 'opencl_probe.js');
if (!existsSync(distProbe)) {
  fail('dist/native/opencl_probe.js not found — run npm run build');
}

// 2. Run the probe via Node
let probeResult;
try {
  const out = execSync(
    `node -e "import('${distProbe.replace(/\\/g, '/')}').then(m => console.log(JSON.stringify(m.probeOpenCl({ deep: false }))))"`,
    { timeout: 5000, encoding: 'utf-8', cwd: rootDir }
  );
  probeResult = JSON.parse(out.trim());
} catch (e) {
  fail(`OpenCL probe execution failed: ${e.message}`);
}

console.log(`[opencl-probe] available=${probeResult.available}, adreno=${probeResult.adrenoDetected}, recommended=${probeResult.recommendedBackend}`);

// 3. If unavailable
if (!probeResult.available) {
  if (allowUnavailable) {
    console.log('[OK] verify-adreno-opencl: OpenCL unavailable (--allow-unavailable passed)');
    process.exit(0);
  } else {
    fail('OpenCL not available on this device. Use --allow-unavailable to skip.');
  }
}

// 4. If available, verify probe returned sane data
if (probeResult.probeTimeMs > 3000) {
  fail(`Probe took ${probeResult.probeTimeMs}ms — exceeds 3s budget`);
}

if (probeResult.libraryCandidates.filter(l => l.exists).length === 0) {
  fail('Probe says available but no library candidates exist');
}

console.log(`[OK] verify-adreno-opencl: OpenCL detected, probe ${probeResult.probeTimeMs}ms`);
if (probeResult.adrenoDetected) {
  console.log(`  Adreno GPU detected`);
}
if (probeResult.warnings.length > 0) {
  console.log(`  Warnings: ${probeResult.warnings.join('; ')}`);
}
