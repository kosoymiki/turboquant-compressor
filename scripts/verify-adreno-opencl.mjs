#!/usr/bin/env node
/**
 * Gate: verify OpenCL/Adreno readiness.
 * Passes if OpenCL available and probe succeeds, OR if --allow-unavailable and OpenCL missing.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { spawnSync } from 'child_process';
import { existsSync, mkdirSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const outPath = join(forensicsDir, 'opencl-adreno-report.json');
const safeRunner = join(rootDir, 'scripts', 'safe-runtime-pack-run.sh');

const allowUnavailable = process.argv.includes('--allow-unavailable');

mkdirSync(forensicsDir, { recursive: true });

function fail(msg) {
  console.error(`FAIL [verify-adreno-opencl]: ${msg}`);
  process.exit(1);
}

function writeReport(report) {
  writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`);
}

function normalizeProbe(probeResult) {
  const devices = Array.isArray(probeResult.devices) ? probeResult.devices : [];
  const adrenoDetected =
    probeResult.adrenoDetected ??
    devices.some((d) => d?.isAdreno || /adreno|fd[0-9]+/i.test(`${d?.name || ''} ${d?.vendor || ''}`));
  return {
    ...probeResult,
    adrenoDetected,
  };
}

let probeResult;
try {
  if (existsSync(safeRunner)) {
    const cliBin =
      process.env.TQ_OPENCL_CLI ||
      join(process.env.XDG_CACHE_HOME || join(process.env.HOME || '.', '.cache'),
           'turboquant', 'native-opencl-build', 'tq_opencl_cli');
    const result = spawnSync(safeRunner, ['probe'], {
      timeout: 15000,
      encoding: 'utf-8',
      cwd: rootDir,
      env: { ...process.env, TQ_OPENCL_CLI: cliBin },
    });
    if (result.status !== 0 || !result.stdout) {
      throw new Error(result.error?.message || result.stderr || `safe-runtime-pack-run exited with status ${result.status ?? 'timeout'}`);
    }
    probeResult = JSON.parse(result.stdout.trim());
  } else {
    const distProbe = join(rootDir, 'dist', 'native', 'opencl_probe.js');
    if (!existsSync(distProbe)) {
      fail('dist/native/opencl_probe.js not found — run npm run build');
    }
    const { probeOpenCl } = await import(distProbe);
    probeResult = probeOpenCl({ deep: false });
  }
} catch (e) {
  fail(`OpenCL probe execution failed: ${e.message}`);
}

probeResult = normalizeProbe(probeResult);

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
const probeBudgetMs =
  probeResult.recommendedBackend === 'mesa_rusticl_kgsl' ? 10000 : 3000;
if (probeResult.probeTimeMs > probeBudgetMs) {
  fail(`Probe took ${probeResult.probeTimeMs}ms — exceeds ${probeBudgetMs / 1000}s budget`);
}

const existingLibraries = Array.isArray(probeResult.libraryCandidates)
  ? probeResult.libraryCandidates.filter((l) => l.exists).length
  : null;
if (existingLibraries !== null && existingLibraries === 0) {
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
