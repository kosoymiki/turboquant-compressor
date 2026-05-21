#!/usr/bin/env node
/**
 * Gate: verify OpenCL/Adreno readiness.
 * Passes if OpenCL available and probe succeeds, OR if --allow-unavailable and OpenCL missing.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { execSync, spawnSync } from 'child_process';
import { existsSync, mkdirSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const outPath = join(forensicsDir, 'opencl-adreno-report.json');

const allowUnavailable = process.argv.includes('--allow-unavailable');

mkdirSync(forensicsDir, { recursive: true });

function fail(msg) {
  console.error(`FAIL [verify-adreno-opencl]: ${msg}`);
  process.exit(1);
}

function writeReport(report) {
  writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`);
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

const report = {
  timestamp: new Date().toISOString(),
  state:
    !probeResult.available
      ? 'ADRENO_VENDOR_OPENCL_UNAVAILABLE'
      : probeResult.recommendedBackend === 'mesa_rusticl_kgsl'
        ? 'CUSTOM_RUSTICL_STACK_PROBED'
        : 'ADRENO_VENDOR_OPENCL_PROBED',
  all_pass: false,
  claim_safe: false,
  source: 'verify-adreno-opencl',
  available: probeResult.available,
  adrenoDetected: probeResult.adrenoDetected,
  recommendedBackend: probeResult.recommendedBackend,
  probeTimeMs: probeResult.probeTimeMs,
  libraryCandidates: probeResult.libraryCandidates,
  warnings: probeResult.warnings,
  opencl_probe: probeResult,
  production: probeResult.production,
};
writeReport(report);

// 3. If unavailable
if (!probeResult.available) {
  if (allowUnavailable) {
    console.log(`[verify-adreno-opencl] Report: ${outPath}`);
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
console.log(`[verify-adreno-opencl] Report: ${outPath}`);
if (probeResult.adrenoDetected) {
  console.log(`  Adreno GPU detected`);
}
if (probeResult.warnings.length > 0) {
  console.log(`  Warnings: ${probeResult.warnings.join('; ')}`);
}
