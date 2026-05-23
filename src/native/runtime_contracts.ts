import { existsSync, readFileSync } from 'fs';
import { dirname, join, resolve } from 'path';
import { fileURLToPath } from 'url';

const MODULE_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

export interface RuntimeContractPaths {
  rootDir: string;
  inferenceRuntimeContractPath: string;
  pagedKvRuntimeContractPath: string;
  precisionCalibrationStatePath: string;
  rusticlEvidencePath: string;
  openclEvidencePath: string;
  mesaEvidencePath: string;
}

export interface InferenceRuntimeContract {
  classification?: string;
  route?: string;
}

export interface PrecisionCalibrationStateContract {
  activationStepSizeMean?: number;
  weightStepSizeMean?: number;
  kvGroupStepSizeMean?: number;
}

export interface RuntimeContractBundle {
  paths: RuntimeContractPaths;
  inferenceRuntimeContract: InferenceRuntimeContract | null;
  pagedKvRuntimeContract: InferenceRuntimeContract | null;
  precisionCalibrationState: PrecisionCalibrationStateContract | null;
}

function envPath(name: string): string | null {
  const value = process.env[name]?.trim();
  return value ? resolve(value) : null;
}

export function resolveRuntimeContractPaths(rootDir = MODULE_ROOT): RuntimeContractPaths {
  const resolvedRoot = resolve(rootDir);
  const inferenceRuntimeContractPath = envPath('TQ_INFERENCE_RUNTIME_CONTRACT')
    ?? join(resolvedRoot, 'forensics', 'adreno', 'opencl-inference-runtime-contract.json');
  const pagedKvRuntimeContractPath = envPath('TQ_PAGED_KV_RUNTIME_CONTRACT')
    ?? join(resolvedRoot, 'forensics', 'adreno', 'opencl-paged-kv-prefix-cache-contract.json');
  const precisionCalibrationStatePath = envPath('TQ_PRECISION_CALIBRATION_STATE')
    ?? join(resolvedRoot, 'forensics', 'precision-calibration-runtime-state.json');
  const rusticlEvidencePath = envPath('TQ_RUSTICL_RUNTIME_EVIDENCE')
    ?? join(resolvedRoot, 'forensics', 'mesa', 'driver-mesh-recovery-ready.json');
  const openclEvidencePath = envPath('TQ_OPENCL_RUNTIME_EVIDENCE')
    ?? join(resolvedRoot, 'forensics', 'opencl-adreno-report.json');
  const mesaEvidencePath = envPath('TQ_MESA_RUNTIME_EVIDENCE')
    ?? join(resolvedRoot, 'forensics', 'mesa', 'driver-mesh-recovery-ready.json');

  return {
    rootDir: resolvedRoot,
    inferenceRuntimeContractPath,
    pagedKvRuntimeContractPath,
    precisionCalibrationStatePath,
    rusticlEvidencePath,
    openclEvidencePath,
    mesaEvidencePath,
  };
}

function loadJson<T>(path: string): T | null {
  try {
    return JSON.parse(readFileSync(path, 'utf8')) as T;
  } catch {
    return null;
  }
}

export function loadRuntimeContractBundle(rootDir = MODULE_ROOT): RuntimeContractBundle {
  const paths = resolveRuntimeContractPaths(rootDir);
  return {
    paths,
    inferenceRuntimeContract: loadJson<InferenceRuntimeContract>(paths.inferenceRuntimeContractPath),
    pagedKvRuntimeContract: loadJson<InferenceRuntimeContract>(paths.pagedKvRuntimeContractPath),
    precisionCalibrationState: loadJson<PrecisionCalibrationStateContract>(paths.precisionCalibrationStatePath),
  };
}

export function runtimeEvidenceExists(path: string): boolean {
  return existsSync(path);
}
