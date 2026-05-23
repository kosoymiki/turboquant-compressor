#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const precisionDir = join(forensicsDir, 'precision');
const runner = join(rootDir, 'scripts', 'run-precision-calibration-runtime.mjs');
mkdirSync(precisionDir, { recursive: true });

function loadJson(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    return null;
  }
}

const runtimeStatePath = join(forensicsDir, 'precision-calibration-runtime-state.json');
const refreshRequested = process.argv.includes('--refresh');

let run = {
  status: null,
  stdout: '',
};
let stdoutJson = null;
const runStartedAtMs = Date.now();

if (refreshRequested) {
  run = spawnSync(process.execPath, [runner], {
    cwd: rootDir,
    encoding: 'utf-8',
  });

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
  observerReady: runtimeState?.observerReady ?? stdoutJson?.observerReady ?? false,
  fakeQuantPolicyReady: runtimeState?.fakeQuantPolicyReady ?? stdoutJson?.fakeQuantPolicyReady ?? false,
  stepSizeContractReady: runtimeState?.stepSizeContractReady ?? stdoutJson?.stepSizeContractReady ?? false,
  calibrationArtifactReady: runtimeState?.calibrationArtifactReady ?? stdoutJson?.calibrationArtifactReady ?? false,
};

const surfaceCount = runtimeState?.surfaceCount ?? stdoutJson?.surfaceCount ?? 0;
const artifactReady = Object.values(readiness).every(Boolean) && surfaceCount >= 3;
const smokeExitClass = refreshRequested
  ? (run.status === 0
      ? 'clean_exit'
      : runtimeStateFresh
        ? 'artifact_materialized_before_nonzero_exit'
        : 'run_failed_without_fresh_artifact')
  : (runtimeStatePresent
      ? 'artifact_only_no_execute'
      : 'no_artifact_no_execute');

const artifact = {
  timestamp: new Date().toISOString(),
  type: 'precision_calibration_runtime_contract',
  route: 'typescript_quantization_surfaces',
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
  refreshHint: refreshRequested
    ? null
    : 'Run with --refresh only when you intentionally want to rebuild the bounded calibration artifact.',
  readiness,
  calibrationPlane: {
    surfaceCount,
    observerStrategy: runtimeState?.observerStrategy ?? stdoutJson?.observerStrategy ?? null,
    fakeQuantNoisePolicy: runtimeState?.fakeQuantNoisePolicy ?? stdoutJson?.fakeQuantNoisePolicy ?? null,
    activationStepSizeMean: runtimeState?.activationStepSizeMean ?? stdoutJson?.activationStepSizeMean ?? 0,
    weightStepSizeMean: runtimeState?.weightStepSizeMean ?? stdoutJson?.weightStepSizeMean ?? 0,
    kvGroupStepSizeMean: runtimeState?.kvGroupStepSizeMean ?? stdoutJson?.kvGroupStepSizeMean ?? 0,
    betaCodebookRange: runtimeState?.betaCodebookRange ?? stdoutJson?.betaCodebookRange ?? null,
  },
  evidence: {
    surfaces: runtimeState?.surfaces ?? stdoutJson?.surfaces ?? [],
    anchors: runtimeState?.anchors ?? stdoutJson?.anchors ?? [],
  },
  residualFrontier: runtimeState?.residualFrontier ?? stdoutJson?.residualFrontier ?? [],
};

const outPath = join(precisionDir, 'precision-calibration-runtime-contract.json');
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(JSON.stringify(artifact, null, 2));
