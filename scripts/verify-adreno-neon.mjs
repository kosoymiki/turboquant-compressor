#!/usr/bin/env node
/**
 * Gate: verify NEON backend builds and passes parity tests.
 * Writes forensics/adreno/neon-parity.json.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { spawnSync } from 'child_process';
import { existsSync, writeFileSync, mkdirSync, readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const buildDir = join(rootDir, 'native', 'adreno', 'build');
const forensicsDir = join(rootDir, 'forensics', 'adreno');

mkdirSync(forensicsDir, { recursive: true });
mkdirSync(buildDir, { recursive: true });

const outPath = join(forensicsDir, 'neon-parity.json');

// Build
console.log('[verify-adreno-neon] Building native/adreno...');
const buildResult = spawnSync('cmake', ['--build', '.', '-j4'], {
  cwd: buildDir, encoding: 'utf-8', timeout: 60000,
});

if (buildResult.status !== 0) {
  // Try configure first
  const confResult = spawnSync('cmake', ['..', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=clang', '-DCMAKE_CXX_COMPILER=clang++'], {
    cwd: buildDir, encoding: 'utf-8', timeout: 30000,
  });
  if (confResult.status !== 0) {
    const report = { state: 'BUILD_FAILED', error: confResult.stderr?.slice(0, 500), all_pass: false };
    writeFileSync(outPath, JSON.stringify(report, null, 2));
    console.error('FAIL [verify-adreno-neon]: cmake configure failed');
    process.exit(1);
  }
  const b2 = spawnSync('cmake', ['--build', '.', '-j4'], { cwd: buildDir, encoding: 'utf-8', timeout: 60000 });
  if (b2.status !== 0) {
    const report = { state: 'BUILD_FAILED', error: b2.stderr?.slice(0, 500), all_pass: false };
    writeFileSync(outPath, JSON.stringify(report, null, 2));
    console.error('FAIL [verify-adreno-neon]: cmake build failed');
    process.exit(1);
  }
}

const cliBin = join(buildDir, 'tq_adreno_cli');
if (!existsSync(cliBin)) {
  const report = { state: 'BINARY_MISSING', all_pass: false };
  writeFileSync(outPath, JSON.stringify(report, null, 2));
  console.error('FAIL [verify-adreno-neon]: tq_adreno_cli not found after build');
  process.exit(1);
}

// Probe
const probe = spawnSync(cliBin, ['probe'], { encoding: 'utf-8', timeout: 5000 });
let probeJson;
try { probeJson = JSON.parse(probe.stdout); } catch { probeJson = { hasNeon: false }; }

// Self-test (value dequant parity)
const selfTest = spawnSync(cliBin, ['self-test'], { encoding: 'utf-8', timeout: 10000 });
let selfTestJson;
try { selfTestJson = JSON.parse(selfTest.stdout); } catch { selfTestJson = { pass: 0, total: 1 }; }

const parityPass = selfTestJson.pass === selfTestJson.total && selfTestJson.total > 0;

// Memory/device info
let memInfo = {};
try {
  const mem = readFileSync('/proc/meminfo', 'utf-8');
  const total = mem.match(/MemTotal:\s+(\d+)/);
  if (total) memInfo.totalKb = parseInt(total[1]);
} catch {}

// Determine state based on actual NEON availability
let state;
if (!parityPass) {
  state = 'ADRENO_CPU_NEON_PARITY_FAIL';
} else if (probeJson.hasNeon === true || probeJson.backend === 'cpu_neon') {
  state = 'ADRENO_CPU_NEON_READY';
} else {
  state = 'ADRENO_CPU_SCALAR_READY';
}

const report = {
  timestamp: new Date().toISOString(),
  state,
  backend: probeJson.backend || 'unknown',
  hasNeon: probeJson.hasNeon || false,
  all_pass: parityPass,
  self_test: selfTestJson,
  memory: memInfo,
};

writeFileSync(outPath, JSON.stringify(report, null, 2));
console.log(`[verify-adreno-neon] Backend: ${report.backend}, NEON: ${report.hasNeon}`);
console.log(`[verify-adreno-neon] Parity: ${selfTestJson.pass}/${selfTestJson.total}`);
console.log(`[verify-adreno-neon] State: ${report.state}`);

if (!parityPass) {
  console.error('FAIL [verify-adreno-neon]: parity test failed');
  process.exit(1);
}
console.log(`[OK] verify-adreno-neon: ${report.state}`);
