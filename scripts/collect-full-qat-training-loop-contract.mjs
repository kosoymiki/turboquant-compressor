#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const precisionDir = join(forensicsDir, 'precision');
const runner = join(rootDir, 'scripts', 'run-full-qat-training-loop.mjs');
mkdirSync(precisionDir, { recursive: true });

function loadJson(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    return null;
  }
}

const runtimeStatePath = join(forensicsDir, 'full-qat-training-loop-state.json');
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
  calibrationStageReady: runtimeState?.calibrationStageReady ?? stdoutJson?.calibrationStageReady ?? false,
  lsqOptimizerStageReady: runtimeState?.lsqOptimizerStageReady ?? stdoutJson?.lsqOptimizerStageReady ?? false,
  distillationStageReady: runtimeState?.distillationStageReady ?? stdoutJson?.distillationStageReady ?? false,
  hybridRuntimeStageReady: runtimeState?.hybridRuntimeStageReady ?? stdoutJson?.hybridRuntimeStageReady ?? false,
  epochLoopReady: runtimeState?.epochLoopReady ?? stdoutJson?.epochLoopReady ?? false,
  metricImprovementReady: runtimeState?.metricImprovementReady ?? stdoutJson?.metricImprovementReady ?? false,
  loopArtifactReady: runtimeState?.loopArtifactReady ?? stdoutJson?.loopArtifactReady ?? false,
};

const artifactReady = Object.values(readiness).every(Boolean);
const smokeExitClass = refreshRequested
  ? (run.status === 0 ? 'clean_exit' : runtimeStateFresh ? 'artifact_materialized_before_nonzero_exit' : 'run_failed_without_fresh_artifact')
  : (runtimeStatePresent ? 'artifact_only_no_execute' : 'no_artifact_no_execute');

const artifact = {
  timestamp: new Date().toISOString(),
  type: 'full_qat_training_loop_contract',
  route: 'artifact_first_qat_loop',
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
  refreshHint: refreshRequested ? null : 'Run with --refresh only when you intentionally want to rebuild the bounded QAT loop artifact.',
  readiness,
  loopPlane: {
    loopProfile: runtimeState?.loopProfile ?? stdoutJson?.loopProfile ?? null,
    epochCount: runtimeState?.epochCount ?? stdoutJson?.epochCount ?? 0,
    stageOrder: runtimeState?.stageOrder ?? stdoutJson?.stageOrder ?? [],
    initialQuantLoss: runtimeState?.initialQuantLoss ?? stdoutJson?.initialQuantLoss ?? 0,
    postLsqQuantLoss: runtimeState?.postLsqQuantLoss ?? stdoutJson?.postLsqQuantLoss ?? 0,
    initialDistillationLoss: runtimeState?.initialDistillationLoss ?? stdoutJson?.initialDistillationLoss ?? 0,
    postDistillationLoss: runtimeState?.postDistillationLoss ?? stdoutJson?.postDistillationLoss ?? 0,
    aggregateInitialLoss: runtimeState?.aggregateInitialLoss ?? stdoutJson?.aggregateInitialLoss ?? 0,
    aggregateFinalLoss: runtimeState?.aggregateFinalLoss ?? stdoutJson?.aggregateFinalLoss ?? 0,
    aggregateImprovement: runtimeState?.aggregateImprovement ?? stdoutJson?.aggregateImprovement ?? 0,
    routeAdmission: runtimeState?.routeAdmission ?? stdoutJson?.routeAdmission ?? null,
  },
  evidence: {
    anchors: runtimeState?.anchors ?? stdoutJson?.anchors ?? [],
  },
  residualFrontier: runtimeState?.residualFrontier ?? stdoutJson?.residualFrontier ?? [],
};

const outPath = join(precisionDir, 'full-qat-training-loop-contract.json');
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(JSON.stringify(artifact, null, 2));
