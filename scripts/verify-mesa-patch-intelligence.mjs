#!/usr/bin/env node
/**
 * Gate: verify Mesa patch intelligence structure is complete and consistent.
 * Does NOT claim Mesa readiness — only validates the intelligence layer.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { existsSync, readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const mesaDir = join(rootDir, 'third_party', 'mesa-adreno');

let errors = 0;

function check(path, desc) {
  const full = join(mesaDir, path);
  if (!existsSync(full)) {
    console.error(`FAIL: missing ${path} — ${desc}`);
    errors++;
    return null;
  }
  try {
    return JSON.parse(readFileSync(full, 'utf-8'));
  } catch {
    console.error(`FAIL: unparseable ${path}`);
    errors++;
    return null;
  }
}

// Required structure files
const upstream = check('upstream.lock.json', 'Mesa upstream pin');
const manifest = check('manifest.lock.json', 'Mesa manifest');
const donors = check('donor-index.json', 'Donor registry');
const patches = check('patch-index.json', 'Patch inventory');
const proof = check('proof-matrix.json', 'Proof requirements');
const build = check('build-matrix.json', 'Build modes');
const donorLicense = check('donor-license-report.json', 'Donor license verification');
const applicability = check('applicability-report.json', 'Patch applicability dry-run');

// Validate upstream has required fields
if (upstream && !upstream.mesa_upstream?.url) {
  console.error('FAIL: upstream.lock.json missing mesa_upstream.url');
  errors++;
}

// Validate upstream pin is real (40-char hex SHA)
if (upstream) {
  const sha = upstream.mesa_upstream?.pinned_commit;
  if (!sha || !/^[0-9a-f]{40}$/.test(sha)) {
    console.error('FAIL: upstream.lock.json pinned_commit must be a real 40-char SHA');
    errors++;
  }
  if (!upstream.mesa_upstream?.required_paths_verified) {
    console.error('FAIL: upstream.lock.json required_paths_verified must be true');
    errors++;
  }
}

// Validate donor-license-report
if (donorLicense) {
  if (!donorLicense.mesa_upstream_commit || !/^[0-9a-f]{40}$/.test(donorLicense.mesa_upstream_commit)) {
    console.error('FAIL: donor-license-report.json missing valid mesa_upstream_commit');
    errors++;
  }
  if (!Array.isArray(donorLicense.donors) || donorLicense.donors.length === 0) {
    console.error('FAIL: donor-license-report.json must have at least one donor');
    errors++;
  }
  for (const d of (donorLicense.donors || [])) {
    if (!d.id || !d.repo || typeof d.license_file_present !== 'boolean') {
      console.error(`FAIL: donor-license-report entry ${d.id || 'unknown'} missing required fields`);
      errors++;
    }
  }
}

// Validate applicability-report
if (applicability) {
  if (!Array.isArray(applicability.patches) || applicability.patches.length === 0) {
    console.error('FAIL: applicability-report.json must have at least one patch entry');
    errors++;
  }
  if (!applicability.summary?.all_research_only) {
    console.error('FAIL: applicability-report.json summary.all_research_only must be true');
    errors++;
  }
  for (const p of (applicability.patches || [])) {
    if (!p.patch_id || !p.claim_state) {
      console.error(`FAIL: applicability-report patch ${p.patch_id || 'unknown'} missing required fields`);
      errors++;
    }
    if (p.claim_state !== 'research_only') {
      console.error(`FAIL: applicability-report patch ${p.patch_id} claims '${p.claim_state}' — must be research_only`);
      errors++;
    }
  }
}

// Validate donors have required fields
if (donors && Array.isArray(donors)) {
  for (const d of donors) {
    if (!d.id || !d.url || !d.role || !d.copy_policy) {
      console.error(`FAIL: donor ${d.id || 'unknown'} missing required fields`);
      errors++;
    }
  }
}

// Validate patches have required fields
if (patches && Array.isArray(patches)) {
  for (const p of patches) {
    if (!p.patch_id || !p.source || !p.claim_state) {
      console.error(`FAIL: patch ${p.patch_id || 'unknown'} missing required fields`);
      errors++;
    }
    if (p.claim_state !== 'research_only' && p.claim_state !== 'not_attempted') {
      console.error(`FAIL: patch ${p.patch_id} has claim_state '${p.claim_state}' without proof`);
      errors++;
    }
  }
}

// Validate no patch claims READY without proof
if (patches && Array.isArray(patches)) {
  const readyClaims = patches.filter(p => p.claim_state === 'ready');
  if (readyClaims.length > 0) {
    console.error(`FAIL: ${readyClaims.length} patch(es) claim ready without proof gate`);
    errors++;
  }
}

if (errors > 0) {
  console.error(`FAIL [verify-mesa-patch-intelligence]: ${errors} error(s)`);
  process.exit(1);
}

const patchCount = patches ? patches.length : 0;
const donorCount = donors ? donors.length : 0;
const pinSha = upstream?.mesa_upstream?.pinned_commit?.slice(0, 12) || 'unknown';
console.log(`[OK] verify-mesa-patch-intelligence: ${donorCount} donors, ${patchCount} patches, pin=${pinSha}, all research_only, structure valid`);
