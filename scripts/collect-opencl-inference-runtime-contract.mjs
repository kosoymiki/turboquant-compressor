#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const adrenoDir = join(forensicsDir, 'adreno');
const safeRunner = join(rootDir, 'scripts', 'safe-runtime-pack-run.sh');
mkdirSync(adrenoDir, { recursive: true });

function loadJson(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    return null;
  }
}

const runtimeStatePath = join(forensicsDir, 'inference-runtime-state.json');
const refreshRequested = process.argv.includes('--refresh');
const timeoutSeconds = process.argv.includes('--timeout-seconds')
  ? parseInt(process.argv[process.argv.indexOf('--timeout-seconds') + 1], 10) || 20
  : 20;

let run = {
  status: null,
  stdout: '',
};
let stdoutJson = null;
const runStartedAtMs = Date.now();

if (refreshRequested) {
  const env = {
    ...process.env,
    TQ_OPENCL_TIMEOUT_SECONDS: String(timeoutSeconds),
    TQ_OPENCL_SILENCE_STDERR: '1',
  };

  run = spawnSync(safeRunner, ['inference-runtime-smoke'], {
    cwd: rootDir,
    encoding: 'utf-8',
    timeout: Math.max(30000, timeoutSeconds * 2000),
    env,
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
  prefillDecodeSplitReady: runtimeState?.prefillDecodeSplitReady ?? stdoutJson?.prefillDecodeSplitReady ?? false,
  chunkedPrefillReady: runtimeState?.chunkedPrefillReady ?? stdoutJson?.chunkedPrefillReady ?? false,
  continuousBatchingReady: runtimeState?.continuousBatchingReady ?? stdoutJson?.continuousBatchingReady ?? false,
  mixedPrefillDecodeReady: runtimeState?.mixedPrefillDecodeReady ?? stdoutJson?.mixedPrefillDecodeReady ?? false,
};

const allReady = Object.values(readiness).every(Boolean);
const requestCount = runtimeState?.requestCount ?? stdoutJson?.requestCount ?? 0;
const waveCount =
  (runtimeState?.prefillWaveCount ?? stdoutJson?.prefillWaveCount ?? 0) +
  (runtimeState?.decodeWaveCount ?? stdoutJson?.decodeWaveCount ?? 0) +
  (runtimeState?.mixedWaveCount ?? stdoutJson?.mixedWaveCount ?? 0);
const artifactReady = Object.values(readiness).every(Boolean) && requestCount >= 3 && waveCount >= 3;
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
  type: 'opencl_inference_runtime_contract',
  route: 'mesa_rusticl_kgsl',
  executionModel: 'process_scoped_isolated_smoke',
  isolationRunner: safeRunner,
  refreshRequested,
  executionAttempted: refreshRequested,
  statusCode: run.status,
  smokeExitClass,
  smokePresent: !!stdoutJson,
  smokeStatus: stdoutJson?.status || null,
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
    : 'Run with --refresh only when you intentionally want a live isolated smoke refresh.',
  readiness,
  runtimeShape: {
    requestCount,
    chunkSizeTokens: runtimeState?.chunkSizeTokens ?? stdoutJson?.chunkSizeTokens ?? 0,
    maxBatchTokens: runtimeState?.maxBatchTokens ?? stdoutJson?.maxBatchTokens ?? 0,
    totalPrefillTokens: runtimeState?.totalPrefillTokens ?? stdoutJson?.totalPrefillTokens ?? 0,
    totalDecodeTokens: runtimeState?.totalDecodeTokens ?? stdoutJson?.totalDecodeTokens ?? 0,
    totalPrefillChunks: runtimeState?.totalPrefillChunks ?? stdoutJson?.totalPrefillChunks ?? 0,
    prefillWaveCount: runtimeState?.prefillWaveCount ?? stdoutJson?.prefillWaveCount ?? 0,
    decodeWaveCount: runtimeState?.decodeWaveCount ?? stdoutJson?.decodeWaveCount ?? 0,
    mixedWaveCount: runtimeState?.mixedWaveCount ?? stdoutJson?.mixedWaveCount ?? 0,
    schedulerTurns: runtimeState?.schedulerTurns ?? stdoutJson?.schedulerTurns ?? 0,
    maxContinuousBatchSize: runtimeState?.maxContinuousBatchSize ?? stdoutJson?.maxContinuousBatchSize ?? 0,
    maxDecodeBatchSize: runtimeState?.maxDecodeBatchSize ?? stdoutJson?.maxDecodeBatchSize ?? 0,
    maxPrefillBatchSize: runtimeState?.maxPrefillBatchSize ?? stdoutJson?.maxPrefillBatchSize ?? 0,
    maxPrefillChunkTokens: runtimeState?.maxPrefillChunkTokens ?? stdoutJson?.maxPrefillChunkTokens ?? 0,
    avgPrefillChunkTokens: runtimeState?.avgPrefillChunkTokens ?? stdoutJson?.avgPrefillChunkTokens ?? 0,
    decodeQueuePeak: runtimeState?.decodeQueuePeak ?? stdoutJson?.decodeQueuePeak ?? 0,
    prefillQueuePeak: runtimeState?.prefillQueuePeak ?? stdoutJson?.prefillQueuePeak ?? 0,
    decodePriorityTurns: runtimeState?.decodePriorityTurns ?? stdoutJson?.decodePriorityTurns ?? 0,
  },
  backendPlane: {
    prefillBackend: runtimeState?.prefillBackend ?? stdoutJson?.prefillBackend ?? null,
    decodeBackend: runtimeState?.decodeBackend ?? stdoutJson?.decodeBackend ?? null,
    batchingPolicy: runtimeState?.batchingPolicy ?? stdoutJson?.batchingPolicy ?? null,
    chunkingPolicy: runtimeState?.chunkingPolicy ?? stdoutJson?.chunkingPolicy ?? null,
  },
  residualFrontier: [
    'speculative_decoding',
  ],
  liveKernelEvidence: {
    totalFusedAttentionNs: runtimeState?.totalFusedAttentionNs ?? stdoutJson?.totalFusedAttentionNs ?? 0,
    totalQjlNs: runtimeState?.totalQjlNs ?? stdoutJson?.totalQjlNs ?? 0,
    totalValueDequantNs: runtimeState?.totalValueDequantNs ?? stdoutJson?.totalValueDequantNs ?? 0,
    requestSummaries: runtimeState?.requests ?? stdoutJson?.requests ?? [],
    waveSummaries: runtimeState?.waves ?? stdoutJson?.waves ?? [],
  },
};

const outPath = join(adrenoDir, 'opencl-inference-runtime-contract.json');
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(JSON.stringify(artifact, null, 2));
