/**
 * TurboQuant production policy for the standalone repo and mirrored deployments.
 * Custom driver stack is the only allowed production route.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { resolveRuntimeContractPaths, runtimeEvidenceExists } from './runtime_contracts.js';

export type AllowedProductionRoute = 'mesa_rusticl_kgsl' | 'turnip_vulkan_kgsl';
export type ProductionRoute = AllowedProductionRoute | 'diagnostic_only';

export interface ProductionPolicyAssessment {
  productionPolicy: 'custom_driver_stack_only';
  productionReady: boolean;
  productionRoute: ProductionRoute;
  allowedProductionRoutes: AllowedProductionRoute[];
  forbiddenProductionBackends: string[];
  evidenceSources: string[];
  blockers: string[];
}

function firstEvidenceHit(candidates: string[]): string | null {
  for (const path of candidates) {
    if (runtimeEvidenceExists(path)) {
      return path;
    }
  }
  return null;
}

export function assessProductionPolicy(): ProductionPolicyAssessment {
  const paths = resolveRuntimeContractPaths();
  const rusticlCandidates = [paths.rusticlEvidencePath, paths.openclEvidencePath];
  const turnipCandidates = [paths.mesaEvidencePath];
  const rusticlEvidence = firstEvidenceHit(rusticlCandidates);
  const turnipEvidence = firstEvidenceHit(turnipCandidates);
  const rusticlReady = rusticlEvidence !== null;
  const turnipReady = turnipEvidence !== null;

  let productionRoute: ProductionRoute = 'diagnostic_only';
  const blockers: string[] = [];

  if (rusticlReady) {
    productionRoute = 'mesa_rusticl_kgsl';
  } else if (turnipReady) {
    productionRoute = 'turnip_vulkan_kgsl';
  } else {
    blockers.push('missing_custom_driver_stack_runtime_evidence');
    blockers.push('vendor_opencl_or_cpu_fallback_not_allowed_for_production');
  }

  return {
    productionPolicy: 'custom_driver_stack_only',
    productionReady: productionRoute !== 'diagnostic_only',
    productionRoute,
    allowedProductionRoutes: ['mesa_rusticl_kgsl', 'turnip_vulkan_kgsl'],
    forbiddenProductionBackends: [
      'typescript_cpu',
      'python_cpu',
      'python_gpu',
      'triton_gpu',
      'opencl_generic',
      'opencl_adreno',
      'vendor_opencl_qualcomm',
      'cpu_neon',
    ],
    evidenceSources: [
      'third_party/mesa-adreno/manifest.lock.json',
      'third_party/mesa-adreno/proof-matrix.json',
      ...(rusticlEvidence ? [rusticlEvidence] : rusticlCandidates),
      ...(turnipEvidence ? [turnipEvidence] : turnipCandidates.filter(path => path !== rusticlEvidence)),
    ],
    blockers,
  };
}
