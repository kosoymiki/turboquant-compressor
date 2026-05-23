#!/usr/bin/env node
/**
 * Gate: write forensics/adreno/loader-report.json with namespace state.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { existsSync, writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics', 'adreno');
const outPath = join(forensicsDir, 'loader-report.json');

mkdirSync(forensicsDir, { recursive: true });

const allowUnavailable = process.argv.includes('--allow-unavailable');

const distProbe = join(rootDir, 'dist', 'native', 'opencl_probe.js');
if (!existsSync(distProbe)) {
  console.error('FAIL [verify-adreno-loader]: dist/native/opencl_probe.js not found — run npm run build');
  process.exit(1);
}

let probeResult;
try {
  const { probeOpenCl } = await import(distProbe);
  probeResult = probeOpenCl({ deep: true, timeoutMs: 3000 });
} catch (e) {
  probeResult = { loaderState: 'NO_LIBRARY', available: false, libraryExists: false, loadable: false, warnings: [e.message] };
}

const report = {
  timestamp: new Date().toISOString(),
  state: probeResult.loaderState || 'UNKNOWN',
  libraryExists: probeResult.libraryExists || false,
  loadable: probeResult.loadable,
  available: probeResult.available || false,
  adrenoDetected: probeResult.adrenoDetected || false,
  recommendedBackend: probeResult.recommendedBackend || 'typescript_cpu',
  warnings: probeResult.warnings || [],
  probeTimeMs: probeResult.probeTimeMs || 0,
  production: probeResult.production,
};

writeFileSync(outPath, JSON.stringify(report, null, 2));
console.log(`[verify-adreno-loader] State: ${report.state}`);
console.log(`[verify-adreno-loader] Library exists: ${report.libraryExists}, Loadable: ${report.loadable}`);
console.log(`[verify-adreno-loader] Report: ${outPath}`);

if (!report.available && !allowUnavailable) {
  console.error('FAIL: OpenCL not available. Use --allow-unavailable to accept.');
  process.exit(1);
}

console.log(`[OK] verify-adreno-loader: ${report.state}`);
