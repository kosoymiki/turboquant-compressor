#!/usr/bin/env node
/**
 * OpenCL/Adreno benchmark runner — always writes forensics/opencl-adreno-report.json.
 * Handles: binary not built, OpenCL unavailable, namespace blocked, parity pass.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { spawnSync } from 'child_process';
import { existsSync, writeFileSync, readFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const buildDir = join(rootDir, 'native', 'opencl', 'build');
const forensicsDir = join(rootDir, 'forensics');
const safeRunner = join(rootDir, 'scripts', 'safe-runtime-pack-run.sh');

mkdirSync(forensicsDir, { recursive: true });

const allowUnavailable = process.argv.includes('--allow-unavailable');
const outPath = join(forensicsDir, 'opencl-adreno-report.json');
const termuxPrefix = process.env.PREFIX || dirname(dirname(process.execPath));

// Memory info
let memInfo = {};
try {
  const mem = readFileSync('/proc/meminfo', 'utf-8');
  const total = mem.match(/MemTotal:\s+(\d+)/);
  const avail = mem.match(/MemAvailable:\s+(\d+)/);
  if (total) memInfo.totalKb = parseInt(total[1]);
  if (avail) memInfo.availableKb = parseInt(avail[1]);
} catch { /* not available */ }

// Device info
let deviceInfo = {};
try {
  const props = ['ro.product.model', 'ro.soc.model', 'ro.board.platform'];
  for (const p of props) {
    const r = spawnSync('getprop', [p], { encoding: 'utf-8', timeout: 1000 });
    if (r.status === 0 && r.stdout.trim()) deviceInfo[p] = r.stdout.trim();
  }
} catch { /* not available */ }

function writeReport(report) {
  writeFileSync(outPath, JSON.stringify(report, null, 2));
  console.log(`[benchmark-opencl-adreno] State: ${report.state}`);
  console.log(`[benchmark-opencl-adreno] Report: ${outPath}`);
}

// Check binary exists
const cliBin = join(buildDir, 'tq_opencl_cli');
const resolvedCliBin = process.env.TQ_OPENCL_CLI || cliBin;
if (!existsSync(resolvedCliBin) && !existsSync(safeRunner)) {
  const report = {
    timestamp: new Date().toISOString(),
    state: 'ADRENO_NATIVE_NOT_BUILT',
    all_pass: false,
    claim_safe: false,
    recommended_next: 'npm run build:opencl',
    device: deviceInfo,
    memory: memInfo,
    parity_tests: [],
    opencl_probe: null,
  };
  writeReport(report);
  if (allowUnavailable) process.exit(0);
  console.error('FAIL: tq_opencl_cli not built. Use --allow-unavailable to skip.');
  process.exit(1);
}

// Run probe — set LD_LIBRARY_PATH to include vendor OpenCL dirs
const openclLdPath = [
  '/system/vendor/lib64',
  '/vendor/lib64',
  `${termuxPrefix}/lib`,
  process.env.LD_LIBRARY_PATH || '',
].filter(Boolean).join(':');

const probeCmd = existsSync(safeRunner)
  ? spawnSync(safeRunner, ['probe'], {
      encoding: 'utf-8',
      timeout: 15000,
      env: { ...process.env, TQ_OPENCL_CLI: resolvedCliBin, LD_LIBRARY_PATH: openclLdPath },
    })
  : spawnSync(resolvedCliBin, ['probe'], {
      encoding: 'utf-8',
      timeout: 5000,
      env: { ...process.env, LD_LIBRARY_PATH: openclLdPath },
    });
let probeJson;
try {
  probeJson = JSON.parse(probeCmd.stdout);
} catch {
  probeJson = { available: false, error: 'parse_failed' };
}

// Run parity tests
const tests = ['test_mse_parity', 'test_qjl_parity', 'test_value_dequant_parity', 'test_fused_attention_parity'];
const results = [];

for (const test of tests) {
  const bin = join(buildDir, test);
  if (!existsSync(bin)) {
    results.push({ test, status: 'not_built', elapsed_ms: 0, output: '' });
    continue;
  }
  const start = Date.now();
  const r = spawnSync(bin, [], { encoding: 'utf-8', timeout: 10000, env: { ...process.env, LD_LIBRARY_PATH: openclLdPath } });
  const elapsed = Date.now() - start;
  results.push({
    test,
    status: r.status === 0 ? 'pass' : 'fail',
    elapsed_ms: elapsed,
    output: (r.stdout || '').trim(),
  });
}

const testsRun = results.filter(r => r.status !== 'not_built').length;
const testsPassed = results.filter(r => r.status === 'pass').length;
const allPass = testsRun > 0 && testsRun === testsPassed;

const recommendedBackend = probeJson.recommendedBackend ?? 'unavailable';
const customStackReady = probeJson.available && recommendedBackend === 'mesa_rusticl_kgsl';
const vendorStackReady = probeJson.available && recommendedBackend === 'opencl_generic';

// Determine state
let state;
if (customStackReady) {
  state = allPass ? 'CUSTOM_RUSTICL_STACK_READY' : 'CUSTOM_RUSTICL_STACK_PARITY_FAIL';
} else if (vendorStackReady) {
  state = allPass ? 'ADRENO_VENDOR_OPENCL_READY' : 'ADRENO_VENDOR_OPENCL_PARITY_FAIL';
} else if (probeJson.recommendedBackend === 'unavailable' || !probeJson.available) {
  state = probeJson.warnings?.some(w => w.includes('No OpenCL platforms'))
    ? 'ADRENO_VENDOR_OPENCL_BLOCKED_BY_NAMESPACE'
    : 'ADRENO_VENDOR_OPENCL_UNAVAILABLE';
} else {
  state = 'ADRENO_VENDOR_OPENCL_UNAVAILABLE';
}

// CPU parity still counts
const cpuParityState = allPass ? 'CPU_REFERENCE_PARITY_PASS' : 'CPU_REFERENCE_PARITY_FAIL';

const report = {
  timestamp: new Date().toISOString(),
  state,
  cpu_parity_state: cpuParityState,
  all_pass: allPass,
  claim_safe: allPass && customStackReady,
  stack_class: customStackReady ? 'custom_rusticl_stack' : (vendorStackReady ? 'vendor_opencl' : 'unavailable'),
  opencl_probe: probeJson,
  device: deviceInfo,
  memory: memInfo,
  parity_tests: results,
  tests_run: testsRun,
  tests_passed: testsPassed,
};

writeReport(report);

if (!allPass && !allowUnavailable) {
  for (const r of results) {
    if (r.status === 'fail') console.error(`  FAIL: ${r.test}`);
  }
  process.exit(1);
}
