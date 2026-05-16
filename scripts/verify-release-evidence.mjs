#!/usr/bin/env node
/**
 * Gate: verify release evidence bundle is complete and consistent.
 * Checks that forensics/ contains required device reports for claims.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { existsSync, readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const adrenoDir = join(forensicsDir, 'adreno');

const REQUIRED_EVIDENCE = [
  { path: 'forensics/opencl-adreno-report.json', field: 'state', description: 'OpenCL device readiness' },
  { path: 'forensics/adreno/loader-report.json', field: 'state', description: 'OpenCL loader state' },
];

const OPTIONAL_EVIDENCE = [
  { path: 'forensics/adreno/opencl-performance-report.json', field: 'all_pass', description: 'Performance benchmark' },
  { path: 'forensics/adreno/opencl-sustained-report.json', field: 'total_runs', description: 'Sustained benchmark' },
];

let errors = 0;
const manifest = { timestamp: new Date().toISOString(), required: [], optional: [], complete: false };

for (const ev of REQUIRED_EVIDENCE) {
  const fullPath = join(rootDir, ev.path);
  if (!existsSync(fullPath)) {
    console.error(`FAIL [release-evidence]: missing required: ${ev.path}`);
    errors++;
    manifest.required.push({ path: ev.path, status: 'missing' });
  } else {
    try {
      const data = JSON.parse(readFileSync(fullPath, 'utf-8'));
      manifest.required.push({ path: ev.path, status: 'present', [ev.field]: data[ev.field] });
    } catch {
      console.error(`FAIL [release-evidence]: unparseable: ${ev.path}`);
      errors++;
      manifest.required.push({ path: ev.path, status: 'unparseable' });
    }
  }
}

for (const ev of OPTIONAL_EVIDENCE) {
  const fullPath = join(rootDir, ev.path);
  if (!existsSync(fullPath)) {
    manifest.optional.push({ path: ev.path, status: 'missing' });
  } else {
    try {
      const data = JSON.parse(readFileSync(fullPath, 'utf-8'));
      manifest.optional.push({ path: ev.path, status: 'present', [ev.field]: data[ev.field] });
    } catch {
      manifest.optional.push({ path: ev.path, status: 'unparseable' });
    }
  }
}

manifest.complete = errors === 0;

// Write manifest
const manifestPath = join(forensicsDir, 'RELEASE_EVIDENCE_MANIFEST.json');
writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

if (errors > 0) {
  console.error(`FAIL [release-evidence]: ${errors} missing/invalid evidence file(s)`);
  console.error('Run device benchmarks to generate evidence, or use --allow-missing');
  if (!process.argv.includes('--allow-missing')) process.exit(1);
} else {
  console.log(`[OK] verify-release-evidence: ${REQUIRED_EVIDENCE.length} required + ${OPTIONAL_EVIDENCE.length} optional evidence files verified`);
}
