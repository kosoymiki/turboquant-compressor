#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const precisionDir = join(forensicsDir, 'precision');
const runner = join(rootDir, 'scripts', 'run-hybrid-cpu-gpu-training-runtime.mjs');
mkdirSync(precisionDir, { recursive: true });

function loadJson(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    return null;
  }
}

const runtimeStatePath = join(forensicsDir, 'hybrid-cpu-gpu-training-runtime-state.json');
const refreshRequested = process.argv.includes('--refresh');

let run = { status: null, stdout: '' };
let stdoutJson = null;
const runStartedAtMs = Date.now();

if (refreshRequested) {
  run = spawnSync(process.execPath, [runner], { cwd: rootDir, encoding: 'utf-8' });
  try {
    stdoutJson = JSON.parse(run.stdout || '');
  } catch {}
}

const runtimeState = loadJson(runtimeStatePath);
const runtimeStatePresent = !!runtimeState;
const runtimeStateFresh = refreshRequested
  ? existsSync(runtimeStatePath) && statSync(runtimeStatePath).mtimeMs >= runStartedAtMs
  : runtimeStatePresent;

const readiness = {
  cpuControlPlaneReady: runtimeState?.cpuControlPlaneReady ?? stdoutJson?.cpuControlPlaneReady ?? false,
  wasmAssistReady: runtimeState?.wasmAssistReady ?? stdoutJson?.wasmAssistReady ?? false,
  gpuInferencePlaneReady: runtimeState?.gpuInferencePlaneReady ?? stdoutJson?.gpuInferencePlaneReady ?? false,
  hybridPartitionReady: runtimeState?.hybridPartitionReady ?? stdoutJson?.hybridPartitionReady ?? false,
  runtimeArtifactReady: runtimeState?.runtimeArtifactReady ?? stdoutJson?.runtimeArtifactReady ?? false,
};

const artifactReady = Object.values(readiness).every(Boolean);
const smokeExitClass = refreshRequested
  ? (run.status === 0 ? 'clean_exit' : runtimeStateFresh ? 'artifact_materialized_before_nonzero_exit' : 'run_failed_without_fresh_artifact')
  : (runtimeStatePresent ? 'artifact_only_no_execute' : 'no_artifact_no_execute');

const artifact = {
  timestamp: new Date().toISOString(),
  type: 'hybrid_cpu_gpu_training_runtime_contract',
  route: 'artifact_first_phase_partition',
  executionModel: 'deterministic_process_artifact',
  isolationRunner: runner,
  refreshRequested,
  executionAttempted: refreshRequested,
  statusCode: run.status,
  smokeExitClass,
  runtimeStatePresent,
  runtimeStateFresh,
  runtimeStatePath,
  classification: artifactReady && runtimeStatePresent
    ? 'module_ready'
    : runtimeStatePresent
      ? 'artifact_present_not_ready'
      : 'awaiting_refresh_artifact',
  refreshHint: refreshRequested ? null : 'Run with --refresh only when you intentionally want to rebuild the hybrid CPU+GPU runtime artifact.',
  readiness,
  runtimePlane: {
    runtimeProfile: runtimeState?.runtimeProfile ?? stdoutJson?.runtimeProfile ?? null,
    recommendedBackend: runtimeState?.recommendedBackend ?? stdoutJson?.recommendedBackend ?? null,
    productionBackendAllowed: runtimeState?.productionBackendAllowed ?? stdoutJson?.productionBackendAllowed ?? false,
    wasmInitialized: runtimeState?.wasmInitialized ?? stdoutJson?.wasmInitialized ?? false,
    openclRoute: runtimeState?.openclRoute ?? stdoutJson?.openclRoute ?? null,
    inferenceContractClassification: runtimeState?.inferenceContractClassification ?? stdoutJson?.inferenceContractClassification ?? null,
    pagedKvContractClassification: runtimeState?.pagedKvContractClassification ?? stdoutJson?.pagedKvContractClassification ?? null,
    activationStepSizeMean: runtimeState?.activationStepSizeMean ?? stdoutJson?.activationStepSizeMean ?? 0,
    weightStepSizeMean: runtimeState?.weightStepSizeMean ?? stdoutJson?.weightStepSizeMean ?? 0,
    kvGroupStepSizeMean: runtimeState?.kvGroupStepSizeMean ?? stdoutJson?.kvGroupStepSizeMean ?? 0,
    phaseRoutes: runtimeState?.phaseRoutes ?? stdoutJson?.phaseRoutes ?? [],
  },
  evidence: {
    anchors: runtimeState?.anchors ?? stdoutJson?.anchors ?? [],
    warnings: runtimeState?.warnings ?? stdoutJson?.warnings ?? [],
  },
  residualFrontier: runtimeState?.residualFrontier ?? stdoutJson?.residualFrontier ?? [],
};

const outPath = join(precisionDir, 'hybrid-cpu-gpu-training-runtime-contract.json');
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(JSON.stringify(artifact, null, 2));
