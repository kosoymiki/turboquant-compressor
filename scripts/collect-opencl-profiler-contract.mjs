#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const adrenoDir = join(forensicsDir, 'adreno');
mkdirSync(adrenoDir, { recursive: true });

function loadJson(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    return null;
  }
}

const eventProfile = loadJson(join(adrenoDir, 'opencl-event-profile.json'));
const perfReport = loadJson(join(adrenoDir, 'opencl-performance-report.json'));
const interceptManifest = loadJson(join(adrenoDir, 'opencl-intercept-manifest.json'));
const interceptSmoke = loadJson(join(adrenoDir, 'opencl-intercept-smoke.json'));
const controlledProfileLane = loadJson(join(adrenoDir, 'opencl-controlled-profile-lane.json'));
const controlledSustainedLane = loadJson(join(adrenoDir, 'opencl-controlled-sustained-lane.json'));
const runtimeInterceptCandidates = [
  join(process.env.PREFIX || '/data/data/com.termux/files/usr', 'lib', 'libOpenCLIntercept.so'),
  join(rootDir, 'third_party', 'OpenCL-Intercept-Layer', 'libOpenCLIntercept.so'),
];
const contractInterceptCandidates = [
  join(rootDir, 'forensics', 'adreno', 'opencl-intercept.config'),
  join(rootDir, 'third_party', 'OpenCL-Intercept-Layer', 'intercept.conf'),
];

const contract = {
  timestamp: new Date().toISOString(),
  type: 'opencl_profiler_contract',
  route: 'mesa_rusticl_kgsl',
  eventProfiling: {
    artifactPresent: !!eventProfile,
    allPass: eventProfile?.all_pass ?? false,
    kernels: eventProfile?.kernels?.map((k) => ({
      kernel: k.kernel,
      mean_ms: k.mean_ms,
      host_mean_ms: k.host_mean_ms,
      delta_ms: k.profiling_signal?.event_overhead_delta_ms ?? null,
    })) ?? [],
  },
  interceptLayer: {
    candidatePresent: contractInterceptCandidates.some((p) => existsSync(p)),
    runtimeCandidatePresent: runtimeInterceptCandidates.some((p) => existsSync(p)),
    searched: {
      runtime: runtimeInterceptCandidates,
      contract: contractInterceptCandidates,
    },
    integrationMode: interceptManifest?.integrationMode || 'optional_external_tool',
    manifestPresent: !!interceptManifest,
    recommendedEnv: interceptManifest?.recommendedEnv || null,
    selectedLibrary: interceptManifest?.selectedLibrary || null,
    selectedConfig: interceptManifest?.selectedConfig || null,
    selectedBy: interceptManifest?.selectedBy || null,
    ready: interceptManifest?.ready ?? false,
    runtimeReady: interceptManifest?.runtimeReady ?? false,
    smokePresent: !!interceptSmoke,
    smokeCommand: interceptSmoke?.command ?? null,
    smokeCrashClass: interceptSmoke?.crashClass ?? null,
    smokeStatusCode: interceptSmoke?.statusCode ?? null,
  },
  controlledLane: {
    profileLanePresent: !!controlledProfileLane,
    sustainedLanePresent: !!controlledSustainedLane,
    latestMode: controlledSustainedLane ? 'sustained' : (controlledProfileLane ? 'profile' : 'none'),
    execute: controlledSustainedLane?.execute ?? controlledProfileLane?.execute ?? false,
    profile: controlledSustainedLane?.benchmark?.profile ?? controlledProfileLane?.benchmark?.profile ?? null,
    timeoutSeconds: controlledSustainedLane?.benchmark?.timeoutSeconds ?? controlledProfileLane?.benchmark?.timeoutSeconds ?? null,
    interceptStatus: controlledSustainedLane?.intercept?.status ?? controlledProfileLane?.intercept?.status ?? null,
  },
  mesaProfiler: {
    envContract: {
      FD_MESA_DEBUG: 'flush',
      FD_RD_DUMP: 'enable,full',
      FD_RD_DUMP_PATH: '/data/local/tmp',
      TU_DEBUG: 'flushall,rd',
      TU_PERFETTO: '1',
      TQ_OPENCL_TRACE: '1',
    },
    linkedArtifacts: [
      'forensics/mesa-stack-forensics-profile.json',
      'forensics/renderdoc-gpu-debug-corpus-ingest.json',
      'forensics/adreno/opencl-performance-report.json',
      'forensics/adreno/opencl-event-profile.json',
      'forensics/adreno/opencl-controlled-profile-lane.json',
      'forensics/adreno/opencl-controlled-sustained-lane.json',
      'forensics/adreno/opencl-intercept-manifest.json',
      'forensics/adreno/opencl-intercept-smoke.json',
    ],
  },
  performanceBaselineReady: !!perfReport || !!eventProfile,
};

const outPath = join(adrenoDir, 'opencl-profiler-contract.json');
writeFileSync(outPath, JSON.stringify(contract, null, 2));
console.log(JSON.stringify({ status: 'ok', outPath, performanceBaselineReady: contract.performanceBaselineReady }, null, 2));
