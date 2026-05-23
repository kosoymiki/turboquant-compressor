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

const runtimeStatePath = join(forensicsDir, 'paged-kv-prefix-cache-state.json');
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

  run = spawnSync(safeRunner, ['paged-kv-prefix-cache-smoke'], {
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
  pagedKvAllocatorReady: runtimeState?.pagedKvAllocatorReady ?? stdoutJson?.pagedKvAllocatorReady ?? false,
  blockTableReady: runtimeState?.blockTableReady ?? stdoutJson?.blockTableReady ?? false,
  prefixCacheSharingReady: runtimeState?.prefixCacheSharingReady ?? stdoutJson?.prefixCacheSharingReady ?? false,
  cacheEvictionReady: runtimeState?.cacheEvictionReady ?? stdoutJson?.cacheEvictionReady ?? false,
};

const allReady = Object.values(readiness).every(Boolean);
const requestCount = runtimeState?.requestCount ?? stdoutJson?.requestCount ?? 0;
const physicalBlockCount = runtimeState?.physicalBlockCount ?? stdoutJson?.physicalBlockCount ?? 0;
const sharedBlockCount = runtimeState?.sharedBlockCount ?? stdoutJson?.sharedBlockCount ?? 0;
const evictedBlockCount = runtimeState?.evictedBlockCount ?? stdoutJson?.evictedBlockCount ?? 0;
const artifactReady =
  allReady &&
  requestCount >= 3 &&
  physicalBlockCount > 0 &&
  sharedBlockCount > 0 &&
  evictedBlockCount > 0;
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
  type: 'opencl_paged_kv_prefix_cache_contract',
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
  allocatorPlane: {
    requestCount,
    blockSizeTokens: runtimeState?.blockSizeTokens ?? stdoutJson?.blockSizeTokens ?? 0,
    physicalBlockCount,
    logicalBlockRefCount: runtimeState?.logicalBlockRefCount ?? stdoutJson?.logicalBlockRefCount ?? 0,
    sharedBlockCount,
    uniqueBlockCount: runtimeState?.uniqueBlockCount ?? stdoutJson?.uniqueBlockCount ?? 0,
    sharedPrefixGroupCount: runtimeState?.sharedPrefixGroupCount ?? stdoutJson?.sharedPrefixGroupCount ?? 0,
    peakLiveBlockCount: runtimeState?.peakLiveBlockCount ?? stdoutJson?.peakLiveBlockCount ?? 0,
    residentBlockCount: runtimeState?.residentBlockCount ?? stdoutJson?.residentBlockCount ?? 0,
    evictableBlockCount: runtimeState?.evictableBlockCount ?? stdoutJson?.evictableBlockCount ?? 0,
    evictedBlockCount,
    maxBlockRefcount: runtimeState?.maxBlockRefcount ?? stdoutJson?.maxBlockRefcount ?? 0,
  },
  backendPlane: {
    prefillBackend: runtimeState?.prefillBackend ?? stdoutJson?.prefillBackend ?? null,
    decodeBackend: runtimeState?.decodeBackend ?? stdoutJson?.decodeBackend ?? null,
    sharingPolicy: runtimeState?.sharingPolicy ?? stdoutJson?.sharingPolicy ?? null,
    evictionPolicy: runtimeState?.evictionPolicy ?? stdoutJson?.evictionPolicy ?? null,
    evictionDecision: runtimeState?.evictionDecision ?? stdoutJson?.evictionDecision ?? null,
  },
  residualFrontier: [
    'speculative_decoding',
  ],
  liveKernelEvidence: {
    totalFusedAttentionNs: runtimeState?.totalFusedAttentionNs ?? stdoutJson?.totalFusedAttentionNs ?? 0,
    totalQjlNs: runtimeState?.totalQjlNs ?? stdoutJson?.totalQjlNs ?? 0,
    totalValueDequantNs: runtimeState?.totalValueDequantNs ?? stdoutJson?.totalValueDequantNs ?? 0,
    requestSummaries: runtimeState?.requests ?? stdoutJson?.requests ?? [],
    blockTable: runtimeState?.blocks ?? stdoutJson?.blocks ?? [],
  },
};

const outPath = join(adrenoDir, 'opencl-paged-kv-prefix-cache-contract.json');
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(JSON.stringify(artifact, null, 2));
