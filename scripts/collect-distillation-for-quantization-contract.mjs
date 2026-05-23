#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const precisionDir = join(forensicsDir, 'precision');
const runner = join(rootDir, 'scripts', 'run-distillation-for-quantization-contract.mjs');
mkdirSync(precisionDir, { recursive: true });

function loadJson(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    return null;
  }
}

const runtimeStatePath = join(forensicsDir, 'distillation-for-quantization-state.json');
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
  teacherStudentPairReady: runtimeState?.teacherStudentPairReady ?? stdoutJson?.teacherStudentPairReady ?? false,
  softenedTargetReady: runtimeState?.softenedTargetReady ?? stdoutJson?.softenedTargetReady ?? false,
  distillationLossReady: runtimeState?.distillationLossReady ?? stdoutJson?.distillationLossReady ?? false,
  studentUpdateReady: runtimeState?.studentUpdateReady ?? stdoutJson?.studentUpdateReady ?? false,
  contractArtifactReady: runtimeState?.contractArtifactReady ?? stdoutJson?.contractArtifactReady ?? false,
};

const artifactReady = Object.values(readiness).every(Boolean);
const smokeExitClass = refreshRequested
  ? (run.status === 0 ? 'clean_exit' : runtimeStateFresh ? 'artifact_materialized_before_nonzero_exit' : 'run_failed_without_fresh_artifact')
  : (runtimeStatePresent ? 'artifact_only_no_execute' : 'no_artifact_no_execute');

const artifact = {
  timestamp: new Date().toISOString(),
  type: 'distillation_for_quantization_contract',
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
  refreshHint: refreshRequested ? null : 'Run with --refresh only when you intentionally want to rebuild the distillation contract artifact.',
  readiness,
  distillationPlane: {
    distillationProfile: runtimeState?.distillationProfile ?? stdoutJson?.distillationProfile ?? null,
    batchSize: runtimeState?.batchSize ?? stdoutJson?.batchSize ?? 0,
    vocabSize: runtimeState?.vocabSize ?? stdoutJson?.vocabSize ?? 0,
    temperature: runtimeState?.temperature ?? stdoutJson?.temperature ?? 0,
    quantBits: runtimeState?.quantBits ?? stdoutJson?.quantBits ?? 0,
    quantStepSize: runtimeState?.quantStepSize ?? stdoutJson?.quantStepSize ?? 0,
    teacherEntropyMean: runtimeState?.teacherEntropyMean ?? stdoutJson?.teacherEntropyMean ?? 0,
    initialKlMean: runtimeState?.initialKlMean ?? stdoutJson?.initialKlMean ?? 0,
    updatedKlMean: runtimeState?.updatedKlMean ?? stdoutJson?.updatedKlMean ?? 0,
    initialAnchorMse: runtimeState?.initialAnchorMse ?? stdoutJson?.initialAnchorMse ?? 0,
    updatedAnchorMse: runtimeState?.updatedAnchorMse ?? stdoutJson?.updatedAnchorMse ?? 0,
    initialCompositeLoss: runtimeState?.initialCompositeLoss ?? stdoutJson?.initialCompositeLoss ?? 0,
    updatedCompositeLoss: runtimeState?.updatedCompositeLoss ?? stdoutJson?.updatedCompositeLoss ?? 0,
    updateNorm: runtimeState?.updateNorm ?? stdoutJson?.updateNorm ?? 0,
  },
  evidence: {
    anchors: runtimeState?.anchors ?? stdoutJson?.anchors ?? [],
  },
  residualFrontier: runtimeState?.residualFrontier ?? stdoutJson?.residualFrontier ?? [],
};

const outPath = join(precisionDir, 'distillation-for-quantization-contract.json');
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(JSON.stringify(artifact, null, 2));
