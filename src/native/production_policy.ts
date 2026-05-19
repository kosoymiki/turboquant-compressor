/**
 * TurboQuant production policy for corpus-control-plane.
 * Custom driver stack is the only allowed production route.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { existsSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

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

const RUSTICL_EVIDENCE = 'forensics/mesa/rusticl-ready.json';
const TURNIP_EVIDENCE = 'forensics/mesa/turnip-ready.json';

function hasEvidence(relPath: string): boolean {
  return existsSync(join(ROOT, relPath));
}

export function assessProductionPolicy(): ProductionPolicyAssessment {
  const rusticlReady = hasEvidence(RUSTICL_EVIDENCE);
  const turnipReady = hasEvidence(TURNIP_EVIDENCE);

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
      RUSTICL_EVIDENCE,
      TURNIP_EVIDENCE,
    ],
    blockers,
  };
}
