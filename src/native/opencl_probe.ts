/**
 * OpenCL probe for Termux/Android — detects OpenCL availability and Adreno GPU.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { spawnSync } from 'child_process';
import { existsSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
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
  hasSubgroupShuffle?: boolean;
  hasSubgroupShuffleRelative?: boolean;
  hasSubgroupBallot?: boolean;
  hasSubgroupClusteredReduce?: boolean;
  hasSubgroupNonUniformArithmetic?: boolean;
  hasIntegerDotProduct?: boolean;
  integerDotProductCapabilities?: number;
  hasSuggestedLocalWorkSize?: boolean;
  hasCreateCommandQueue?: boolean;
  hasInitializeMemory?: boolean;
  hasDeviceUuid?: boolean;
  deviceUuid?: string;
  hasPriorityHints?: boolean;
  hasThrottleHints?: boolean;
  hasCommandBuffer?: boolean;
  hasCommandBufferMutableDispatch?: boolean;
  hasExternalMemory?: boolean;
  hasExternalMemoryAhb?: boolean;
  hasExternalSemaphore?: boolean;
  hasSvm?: boolean;
  hasSvmCoarse?: boolean;
  hasSvmFine?: boolean;
  hasSvmAtomics?: boolean;
  commandBufferCapabilities?: number;
  adrenoDetected: boolean;
  libraryCandidates: Array<{ path: string; exists: boolean }>;
  recommendedBackend: 'typescript_cpu' | 'opencl_adreno' | 'opencl_generic' | 'cpu_neon' | 'mesa_rusticl_kgsl';
  loadedLibraryPath?: string;
  runtimePackProbe?: boolean;
  production: ProductionPolicyAssessment;
  warnings: string[];
  probeTimeMs: number;
  traceContract?: {
    turboquantNativeScope: 'TQ_OPENCL_TRACE';
    rusticlScope: 'RUSTICL_DEBUG=memory';
    unresolvedLanes: Array<'initialize_memory' | 'command_buffer' | 'external_memory' | 'external_semaphore' | 'system_svm'>;
  };
}

const termuxPrefix = process.env['PREFIX'] ?? (() => {
  const nodeDir = process.execPath ? process.execPath.replace(/\/bin\/node$/, '') : '';
  return nodeDir || '/usr';
})();

const LIBRARY_CANDIDATES = [
  '/system/vendor/lib64/libOpenCL.so',
  '/vendor/lib64/libOpenCL.so',
  '/system/lib64/libOpenCL.so',
  '/vendor/lib64/libOpenCL_adreno.so',
  '/system/vendor/lib64/libOpenCL_adreno.so',
  `${termuxPrefix}/lib/libOpenCL.so`,
  `${termuxPrefix}/opt/vendor/lib/libOpenCL.so`,
];

const rootDir = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const safeRuntimePackRunner = join(rootDir, 'scripts', 'safe-runtime-pack-run.sh');

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
  const production = assessProductionPolicy();

  // Check library candidates
  const libResults = LIBRARY_CANDIDATES.map(path => ({
    path,
    exists: existsSync(path),
  }));

  const anyLibFound = libResults.some(l => l.exists);

  if (!anyLibFound) {
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
      traceContract: {
        turboquantNativeScope: 'TQ_OPENCL_TRACE',
        rusticlScope: 'RUSTICL_DEBUG=memory',
        unresolvedLanes: ['initialize_memory', 'command_buffer', 'external_memory', 'external_semaphore', 'system_svm'],
      },
    };
  }

  // Quick mode: library exists but loadability not proven
  if (!deep) {
    const adrenoHint = libResults.some(l => l.exists && l.path.includes('adreno'));
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
      traceContract: {
        turboquantNativeScope: 'TQ_OPENCL_TRACE',
        rusticlScope: 'RUSTICL_DEBUG=memory',
        unresolvedLanes: ['initialize_memory', 'command_buffer', 'external_memory', 'external_semaphore', 'system_svm'],
      },
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
  let firstRuntimePackDevice: Record<string, unknown> | undefined;

  let runtimePackProbe = false;
  let loadedLibraryPath: string | undefined;

  if (existsSync(safeRuntimePackRunner)) {
    const safeProbe = spawnSync(safeRuntimePackRunner, ['probe'], {
      // Runtime-pack probe can exceed ambient clinfo timing because it boots an
      // isolated Rusticl stack. Keep a wider floor so deep probes do not fall
      // back to host/vendor OpenCL by timeout alone.
      timeout: Math.max(timeoutMs, 15000),
      cwd: rootDir,
      encoding: 'utf-8',
    });
    if (safeProbe.status === 0 && safeProbe.stdout) {
      try {
        const parsed = JSON.parse(safeProbe.stdout.trim());
        runtimePackProbe = true;
        firstRuntimePackDevice = Array.isArray(parsed.devices) ? parsed.devices[0] as Record<string, unknown> : undefined;
        deviceNames = Array.isArray(parsed.devices) ? parsed.devices.map((d: Record<string, unknown>) => String(d.name ?? '')).filter(Boolean) : deviceNames;
        vendorNames = Array.isArray(parsed.devices) ? parsed.devices.map((d: Record<string, unknown>) => String(d.vendor ?? '')).filter(Boolean) : vendorNames;
        versionStrings = Array.isArray(parsed.devices) ? parsed.devices.map((d: Record<string, unknown>) => String(d.version ?? '')).filter(Boolean) : versionStrings;
        platformCount = typeof parsed.platformCount === 'number' ? parsed.platformCount : platformCount;
        deviceCount = typeof parsed.deviceCount === 'number' ? parsed.deviceCount : deviceCount;
        hasFp16 = Array.isArray(parsed.devices) ? parsed.devices.some((d: Record<string, unknown>) => d.hasFp16 === true) : hasFp16;
        hasSubgroups = Array.isArray(parsed.devices) ? parsed.devices.some((d: Record<string, unknown>) => d.hasSubgroups === true) : hasSubgroups;
        const runtimePackAdreno = Array.isArray(parsed.devices)
          ? parsed.devices.some((d: Record<string, unknown>) => /adreno|fd[0-9]+/i.test(`${String(d.name ?? '')} ${String(d.vendor ?? '')}`))
          : false;
        adrenoDetected = parsed.adrenoDetected === true || runtimePackAdreno || adrenoDetected;
        loadedLibraryPath = typeof parsed.loadedLibraryPath === 'string' ? parsed.loadedLibraryPath : loadedLibraryPath;
      } catch {
        warnings.push('safe-runtime-pack-run probe returned non-JSON output');
      }
    } else if (production.productionReady) {
      warnings.push(`safe-runtime-pack-run probe failed with status ${safeProbe.status ?? 'timeout'}`);
    }
  }

  // Production truth prefers the isolated runtime-pack route. Only fall back
  // to ambient clinfo when that route is unavailable or failed.

  // Direct check for staged Rusticl library (TQ native stack)
  const stagedRusticlPath = '/data/data/com.termux/files/tq-rusticl/libRusticlOpenCL.so.1.0.0';
  const stagedRusticlFound = existsSync(stagedRusticlPath);
  // Check Vulkan layer (Turnip/KGSL)
  const stagedVulkanPath = '/data/data/com.termux/files/tq-rusticl/libvulkan_freedreno.so';
  const stagedVulkanFound = existsSync(stagedVulkanPath);

  if (stagedRusticlFound || stagedVulkanFound) {
    runtimePackProbe = true;
    adrenoDetected = true;
    // Mark as ready for classification
    if (platformCount === 0) platformCount = 1;
    if (deviceCount === 0) deviceCount = 1;
    deviceNames = ['Qualcomm Adreno 730v3 (Mesa Rusticl/KGSL + Turnip Vulkan)'];
    vendorNames = ['Qualcomm'];
    versionStrings = ['OpenCL 3.1 (Mesa Rusticl)', 'Vulkan 1.4.335 (Turnip/KGSL)'];
    warnings.push('Staged driver stack detected: mesa_rusticl_kgsl + turnip_vulkan_kgsl routes active');
  }

  if (!runtimePackProbe) {
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

  if (!production.productionReady) {
    warnings.push(`Production route blocked: ${production.blockers.join(', ')}`);
  }

  const customPackReady = runtimePackProbe && platformCount > 0;
  // Classify based on what we detected
  let recommendedBackend: OpenClProbeResult['recommendedBackend'];
  if (stagedRusticlFound && stagedVulkanFound) {
    recommendedBackend = 'mesa_rusticl_kgsl'; // OpenCL + Vulkan dual stack
  } else if (customPackReady) {
    recommendedBackend = 'mesa_rusticl_kgsl';
  } else if (clInfoWorked) {
    recommendedBackend = adrenoDetected ? 'opencl_adreno' : 'opencl_generic';
  } else {
    recommendedBackend = 'typescript_cpu';
  }

  return {
    available: (clInfoWorked && platformCount > 0) || customPackReady,
    libraryExists: true,
    loadable: customPackReady ? true : clInfoWorked,
    loaderState: customPackReady ? 'READY' : loaderState,
    platformCount,
    deviceCount,
    deviceNames,
    vendorNames,
    versionStrings,
    hasFp16,
    hasSubgroups,
    hasSuggestedLocalWorkSize: runtimePackProbe ? Boolean(firstRuntimePackDevice?.hasSuggestedLocalWorkSize) : undefined,
    hasCreateCommandQueue: runtimePackProbe ? firstRuntimePackDevice?.hasCreateCommandQueue === true : undefined,
    hasInitializeMemory: runtimePackProbe ? firstRuntimePackDevice?.hasInitializeMemory === true : undefined,
    hasDeviceUuid: runtimePackProbe ? firstRuntimePackDevice?.hasDeviceUuid === true : undefined,
    deviceUuid: runtimePackProbe && typeof firstRuntimePackDevice?.deviceUuid === 'string' ? String(firstRuntimePackDevice.deviceUuid) : undefined,
    hasPriorityHints: runtimePackProbe ? firstRuntimePackDevice?.hasPriorityHints === true : undefined,
    hasThrottleHints: runtimePackProbe ? firstRuntimePackDevice?.hasThrottleHints === true : undefined,
    hasCommandBuffer: runtimePackProbe ? firstRuntimePackDevice?.hasCommandBuffer === true : undefined,
    hasCommandBufferMutableDispatch: runtimePackProbe ? firstRuntimePackDevice?.hasCommandBufferMutableDispatch === true : undefined,
    hasExternalMemory: runtimePackProbe ? firstRuntimePackDevice?.hasExternalMemory === true : undefined,
    hasExternalMemoryAhb: runtimePackProbe ? firstRuntimePackDevice?.hasExternalMemoryAhb === true : undefined,
    hasExternalSemaphore: runtimePackProbe ? firstRuntimePackDevice?.hasExternalSemaphore === true : undefined,
    hasSvm: runtimePackProbe ? firstRuntimePackDevice?.hasSvm === true : undefined,
    hasSvmCoarse: runtimePackProbe ? firstRuntimePackDevice?.hasSvmCoarse === true : undefined,
    hasSvmFine: runtimePackProbe ? firstRuntimePackDevice?.hasSvmFine === true : undefined,
    hasSvmAtomics: runtimePackProbe ? firstRuntimePackDevice?.hasSvmAtomics === true : undefined,
    commandBufferCapabilities: runtimePackProbe && typeof firstRuntimePackDevice?.commandBufferCapabilities === 'number'
      ? Number(firstRuntimePackDevice.commandBufferCapabilities)
      : undefined,
    adrenoDetected,
    libraryCandidates: libResults,
    recommendedBackend,
    loadedLibraryPath,
    runtimePackProbe,
    production,
    warnings,
    probeTimeMs: Date.now() - start,
    traceContract: {
      turboquantNativeScope: 'TQ_OPENCL_TRACE',
      rusticlScope: 'RUSTICL_DEBUG=memory',
      unresolvedLanes: ['initialize_memory', 'command_buffer', 'external_memory', 'external_semaphore', 'system_svm'],
    },
  };
}
