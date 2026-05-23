#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const precisionDir = join(forensicsDir, 'precision');
const runner = join(rootDir, 'scripts', 'run-gradient-verified-lsq-optimizer.mjs');
mkdirSync(precisionDir, { recursive: true });

function loadJson(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    return null;
  }
}

const runtimeStatePath = join(forensicsDir, 'gradient-verified-lsq-optimizer-state.json');
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
  gradientVerificationReady: runtimeState?.gradientVerificationReady ?? stdoutJson?.gradientVerificationReady ?? false,
  optimizerStepReady: runtimeState?.optimizerStepReady ?? stdoutJson?.optimizerStepReady ?? false,
  lossReductionReady: runtimeState?.lossReductionReady ?? stdoutJson?.lossReductionReady ?? false,
  optimizerArtifactReady: runtimeState?.optimizerArtifactReady ?? stdoutJson?.optimizerArtifactReady ?? false,
};

const artifactReady = Object.values(readiness).every(Boolean);
const smokeExitClass = refreshRequested
  ? (run.status === 0
      ? 'clean_exit'
      : runtimeStateFresh
        ? 'artifact_materialized_before_nonzero_exit'
        : 'run_failed_without_fresh_artifact')
  : (runtimeStatePresent ? 'artifact_only_no_execute' : 'no_artifact_no_execute');

const artifact = {
  timestamp: new Date().toISOString(),
  type: 'gradient_verified_lsq_optimizer_contract',
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
  refreshHint: refreshRequested ? null : 'Run with --refresh only when you intentionally want to rebuild the LSQ optimizer artifact.',
  readiness,
  optimizerPlane: {
    optimizerProfile: runtimeState?.optimizerProfile ?? stdoutJson?.optimizerProfile ?? null,
    quantBits: runtimeState?.quantBits ?? stdoutJson?.quantBits ?? 0,
    tensorLength: runtimeState?.tensorLength ?? stdoutJson?.tensorLength ?? 0,
    initialStepSize: runtimeState?.initialStepSize ?? stdoutJson?.initialStepSize ?? 0,
    updatedStepSize: runtimeState?.updatedStepSize ?? stdoutJson?.updatedStepSize ?? 0,
    learningRate: runtimeState?.learningRate ?? stdoutJson?.learningRate ?? 0,
    gradientScale: runtimeState?.gradientScale ?? stdoutJson?.gradientScale ?? 0,
    initialLoss: runtimeState?.initialLoss ?? stdoutJson?.initialLoss ?? 0,
    updatedLoss: runtimeState?.updatedLoss ?? stdoutJson?.updatedLoss ?? 0,
    analyticGradient: runtimeState?.analyticGradient ?? stdoutJson?.analyticGradient ?? 0,
    finiteDifferenceGradient: runtimeState?.finiteDifferenceGradient ?? stdoutJson?.finiteDifferenceGradient ?? 0,
    relativeGradientError: runtimeState?.relativeGradientError ?? stdoutJson?.relativeGradientError ?? 0,
    cosineSimilarityAfterUpdate: runtimeState?.cosineSimilarityAfterUpdate ?? stdoutJson?.cosineSimilarityAfterUpdate ?? 0,
    clippedFraction: runtimeState?.clippedFraction ?? stdoutJson?.clippedFraction ?? 0,
  },
  evidence: {
    anchors: runtimeState?.anchors ?? stdoutJson?.anchors ?? [],
  },
  residualFrontier: runtimeState?.residualFrontier ?? stdoutJson?.residualFrontier ?? [],
};

const outPath = join(precisionDir, 'gradient-verified-lsq-optimizer-contract.json');
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(JSON.stringify(artifact, null, 2));
