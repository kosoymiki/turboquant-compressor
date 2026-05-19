/**
 * OpenCL probe for Termux/Android — detects OpenCL availability and Adreno GPU.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { spawnSync } from 'child_process';
import { existsSync } from 'fs';
import { type ProductionPolicyAssessment, assessProductionPolicy } from './production_policy.js';

export interface OpenClProbeResult {
  available: boolean;
  libraryExists: boolean;
  loadable: boolean | null; // null = not tested (quick mode)
  loaderState: 'NO_LIBRARY' | 'LIBRARY_EXISTS_NOT_PROVEN_LOADABLE' | 'BLOCKED_BY_ANDROID_LINKER_NAMESPACE' | 'LOADABLE_NO_PLATFORMS' | 'READY';
  platformCount: number;
  deviceCount: number;
  deviceNames: string[];
  vendorNames: string[];
  versionStrings: string[];
  hasFp16: boolean;
  hasSubgroups: boolean;
  adrenoDetected: boolean;
  libraryCandidates: Array<{ path: string; exists: boolean }>;
  recommendedBackend: 'typescript_cpu' | 'opencl_adreno' | 'opencl_generic' | 'cpu_neon';
  production: ProductionPolicyAssessment;
  warnings: string[];
  probeTimeMs: number;
}

const LIBRARY_CANDIDATES = [
  '/system/vendor/lib64/libOpenCL.so',
  '/vendor/lib64/libOpenCL.so',
  '/system/lib64/libOpenCL.so',
  '/vendor/lib64/libOpenCL_adreno.so',
  '/system/vendor/lib64/libOpenCL_adreno.so',
  `${process.env['PREFIX'] ?? '/data/data/com.termux/files/usr'}/lib/libOpenCL.so`,
  `${process.env['PREFIX'] ?? '/data/data/com.termux/files/usr'}/opt/vendor/lib/libOpenCL.so`,
];

export interface OpenClProbeOptions {
  /** Run clinfo for deep device info (may take >1s) */
  deep?: boolean;
  /** Timeout for clinfo subprocess in ms */
  timeoutMs?: number;
}

export function probeOpenCl(options: OpenClProbeOptions = {}): OpenClProbeResult {
  const start = Date.now();
  const deep = options.deep === true;
  const timeoutMs = Math.max(100, Math.min(options.timeoutMs ?? 2000, 5000));
  const warnings: string[] = [];

  // Check library candidates
  const libResults = LIBRARY_CANDIDATES.map(path => ({
    path,
    exists: existsSync(path),
  }));

  const anyLibFound = libResults.some(l => l.exists);

  if (!anyLibFound) {
    const production = assessProductionPolicy();
    return {
      available: false,
      libraryExists: false,
      loadable: false,
      loaderState: 'NO_LIBRARY',
      platformCount: 0,
      deviceCount: 0,
      deviceNames: [],
      vendorNames: [],
      versionStrings: [],
      hasFp16: false,
      hasSubgroups: false,
      adrenoDetected: false,
      libraryCandidates: libResults,
      recommendedBackend: 'typescript_cpu',
      production,
      warnings: [
        'No OpenCL library found on device',
        ...(production.productionReady ? [] : [`Production route blocked: ${production.blockers.join(', ')}`]),
      ],
      probeTimeMs: Date.now() - start,
    };
  }

  // Quick mode: library exists but loadability not proven
  if (!deep) {
    const adrenoHint = libResults.some(l => l.exists && l.path.includes('adreno'));
    const production = assessProductionPolicy();
    return {
      available: false,
      libraryExists: true,
      loadable: null,
      loaderState: 'LIBRARY_EXISTS_NOT_PROVEN_LOADABLE',
      platformCount: -1,
      deviceCount: -1,
      deviceNames: [],
      vendorNames: adrenoHint ? ['Qualcomm (inferred from library path)'] : [],
      versionStrings: [],
      hasFp16: false,
      hasSubgroups: false,
      adrenoDetected: adrenoHint,
      libraryCandidates: libResults,
      recommendedBackend: 'typescript_cpu',
      production,
      warnings: [
        'Library exists but loadability not proven; use deep=true to test',
        ...(production.productionReady ? [] : [`Production route blocked: ${production.blockers.join(', ')}`]),
      ],
      probeTimeMs: Date.now() - start,
    };
  }

  // Deep mode: run clinfo
  let deviceNames: string[] = [];
  let vendorNames: string[] = [];
  let versionStrings: string[] = [];
  let platformCount = 0;
  let deviceCount = 0;
  let hasFp16 = false;
  let hasSubgroups = false;
  let adrenoDetected = false;

  const clinfoAvailable = spawnSync('command', ['-v', 'clinfo'], {
    shell: true, timeout: 1000,
  }).status === 0;

  if (!clinfoAvailable) {
    warnings.push('clinfo not installed; install with: pkg install clinfo');
  } else {
    const libPaths = libResults.filter(l => l.exists).map(l => l.path);
    const ldPath = libPaths.map(p => p.replace(/\/[^/]+$/, '')).join(':');

    const result = spawnSync('clinfo', [], {
      timeout: timeoutMs,
      env: { ...process.env, LD_LIBRARY_PATH: `${ldPath}:${process.env['LD_LIBRARY_PATH'] ?? ''}` },
      encoding: 'utf-8',
    });

    if (result.status === 0 && result.stdout) {
      const out = result.stdout;

      const platMatch = out.match(/Number of platforms\s+(\d+)/);
      if (platMatch && platMatch[1]) platformCount = parseInt(platMatch[1], 10);

      const devMatch = out.match(/Number of devices\s+(\d+)/g);
      if (devMatch) deviceCount = devMatch.length;

      const nameMatches = out.matchAll(/Device Name\s+(.+)/g);
      for (const m of nameMatches) { const v = m[1]; if (v) deviceNames.push(v.trim()); }

      const vendorMatches = out.matchAll(/Device Vendor\s+(.+)/g);
      for (const m of vendorMatches) { const v = m[1]; if (v) vendorNames.push(v.trim()); }

      const verMatches = out.matchAll(/Device Version\s+(.+)/g);
      for (const m of verMatches) { const v = m[1]; if (v) versionStrings.push(v.trim()); }

      hasFp16 = /cl_khr_fp16/.test(out);
      hasSubgroups = /cl_khr_subgroups|cl_intel_subgroups/.test(out);
      adrenoDetected = /[Aa]dreno|QUALCOMM|Qualcomm/i.test(out);
    } else {
      warnings.push(`clinfo exited with status ${result.status ?? 'timeout'}`);
    }
  }

  if (!adrenoDetected) {
    adrenoDetected = libResults.some(l => l.exists && l.path.includes('adreno'));
  }

  // Determine loader state based on deep probe results
  const clInfoWorked = platformCount > 0;
  const blocked = !clInfoWorked && anyLibFound;
  const loaderState = clInfoWorked ? 'READY' as const
    : blocked ? 'BLOCKED_BY_ANDROID_LINKER_NAMESPACE' as const
    : 'LOADABLE_NO_PLATFORMS' as const;

  const production = assessProductionPolicy();
  if (!production.productionReady) {
    warnings.push(`Production route blocked: ${production.blockers.join(', ')}`);
  }

  return {
    available: clInfoWorked && platformCount > 0,
    libraryExists: true,
    loadable: clInfoWorked,
    loaderState,
    platformCount,
    deviceCount,
    deviceNames,
    vendorNames,
    versionStrings,
    hasFp16,
    hasSubgroups,
    adrenoDetected,
    libraryCandidates: libResults,
    recommendedBackend: clInfoWorked
      ? (adrenoDetected ? 'opencl_adreno' : 'opencl_generic')
      : 'typescript_cpu',
    production,
    warnings,
    probeTimeMs: Date.now() - start,
  };
}
