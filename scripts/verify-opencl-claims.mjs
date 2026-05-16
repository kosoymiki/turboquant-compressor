#!/usr/bin/env node
/**
 * Gate: verify OpenCL claims in docs/README are backed by benchmark evidence.
 * Rejects claims like "Adreno optimized" without forensics/opencl-adreno-report.json.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { readFileSync, existsSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

function fail(msg) {
  console.error(`FAIL [verify-opencl-claims]: ${msg}`);
  process.exit(1);
}

const CLAIMS_REQUIRING_EVIDENCE = [
  { pattern: /OpenCL\s+accelerat/i, evidence: 'forensics/opencl-adreno-report.json', label: 'OpenCL accelerated' },
  { pattern: /Adreno\s+optimiz/i, evidence: 'forensics/opencl-adreno-report.json', label: 'Adreno optimized' },
  { pattern: /Snapdragon.*ready/i, evidence: 'forensics/opencl-adreno-report.json', label: 'Snapdragon ready' },
  { pattern: /mobile\s+high.?end\s+backend/i, evidence: 'forensics/opencl-adreno-report.json', label: 'mobile high-end backend' },
];

// Policy files describe forbidden phrases — not product claims
const CLAIM_POLICY_FILES = new Set([
  'docs/OPENCL_BENCHMARK_POLICY.md',
]);

// Scan docs and README for claims
const filesToScan = [];
if (existsSync(join(rootDir, 'README.md'))) filesToScan.push('README.md');
const docsDir = join(rootDir, 'docs');
if (existsSync(docsDir)) {
  for (const f of readdirSync(docsDir)) {
    const rel = join('docs', f);
    if (f.endsWith('.md') && !CLAIM_POLICY_FILES.has(rel)) filesToScan.push(rel);
  }
}

let violations = 0;

for (const relPath of filesToScan) {
  const content = readFileSync(join(rootDir, relPath), 'utf-8');
  for (const claim of CLAIMS_REQUIRING_EVIDENCE) {
    if (claim.pattern.test(content)) {
      const evidencePath = join(rootDir, claim.evidence);
      if (!existsSync(evidencePath)) {
        console.error(`  CLAIM "${claim.label}" in ${relPath} — missing evidence: ${claim.evidence}`);
        violations++;
      } else {
        // Verify evidence shows success
        try {
          const report = JSON.parse(readFileSync(evidencePath, 'utf-8'));
          if (!report.all_pass) {
            console.error(`  CLAIM "${claim.label}" in ${relPath} — evidence exists but tests failed`);
            violations++;
          }
        } catch {
          console.error(`  CLAIM "${claim.label}" in ${relPath} — evidence file unparseable`);
          violations++;
        }
      }
    }
  }
}

if (violations > 0) {
  fail(`${violations} unsupported claim(s) found. Run benchmark-opencl-adreno.mjs first or remove claims.`);
}

console.log(`[OK] verify-opencl-claims: ${filesToScan.length} files scanned, no unsupported claims`);
