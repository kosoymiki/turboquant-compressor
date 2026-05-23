/**
 * MCP tool: turboquant_adreno_loader_probe — Android linker namespace diagnosis.
 * Reports whether the active OpenCL route is vendor/system or the custom Rusticl stack.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { probeOpenCl } from '../native/opencl_probe.js';
import { spawnSync } from 'child_process';

export const turboquantAdrenoLoaderProbeSchema = {
  description: 'Diagnose Android linker namespace state for vendor OpenCL. Reports whether libOpenCL.so exists, is loadable, or is blocked by namespace isolation. Includes device/SoC info when available.',
  inputSchema: {
    type: 'object' as const,
    properties: {
      deep: { type: 'boolean', description: 'Run clinfo deep probe (slower)' },
    },
    required: [] as string[],
  },
};

export function turboquantAdrenoLoaderProbe(args: { deep?: boolean }): object {
  const probe = probeOpenCl({ deep: args.deep ?? true, timeoutMs: 3000 });

  // Collect device info
  const device: Record<string, string> = {};
  const props = ['ro.board.platform', 'ro.hardware', 'ro.soc.model', 'ro.product.model', 'ro.build.version.release'];
  for (const prop of props) {
    const r = spawnSync('getprop', [prop], { encoding: 'utf-8', timeout: 1000 });
    if (r.status === 0 && r.stdout.trim()) device[prop] = r.stdout.trim();
  }

  // Determine recommendation
  let recommendation: string;
  switch (probe.loaderState) {
    case 'NO_LIBRARY':
      recommendation = 'No vendor OpenCL. Use CPU/NEON backend or install Mesa/Rusticl.';
      break;
    case 'LIBRARY_EXISTS_NOT_PROVEN_LOADABLE':
      recommendation = 'Library found but not tested. Run with deep=true to diagnose.';
      break;
    case 'BLOCKED_BY_ANDROID_LINKER_NAMESPACE':
      recommendation = 'Vendor OpenCL blocked by Android linker namespace. Options: (1) CPU/NEON backend, (2) Vulkan compute, (3) Mesa/Rusticl with freedreno, (4) Android companion service.';
      break;
    case 'LOADABLE_NO_PLATFORMS':
      recommendation = 'OpenCL loads but no platforms found. Driver may be broken.';
      break;
    case 'READY':
      recommendation = probe.recommendedBackend === 'mesa_rusticl_kgsl'
        ? 'Custom Rusticl/KGSL stack ready. Use mesa_rusticl_kgsl.'
        : 'Vendor OpenCL ready. Use opencl_adreno or opencl_generic backend.';
      break;
  }

  return {
    loaderState: probe.loaderState,
    libraryExists: probe.libraryExists,
    loadable: probe.loadable,
    available: probe.available,
    adrenoDetected: probe.adrenoDetected,
    recommendedBackend: probe.recommendedBackend,
    recommendation,
    device,
    libraryCandidates: probe.libraryCandidates.filter(l => l.exists),
    warnings: probe.warnings,
    probeTimeMs: probe.probeTimeMs,
  };
}
