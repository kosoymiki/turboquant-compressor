#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const adrenoDir = join(rootDir, 'forensics', 'adreno');
mkdirSync(adrenoDir, { recursive: true });

const sustained = process.argv.includes('--sustained');
const execute = process.argv.includes('--execute');
const minutes = process.argv.includes('--minutes') ? parseInt(process.argv[process.argv.indexOf('--minutes') + 1], 10) || 1 : 1;
const warmup = process.argv.includes('--warmup') ? parseInt(process.argv[process.argv.indexOf('--warmup') + 1], 10) || 1 : 1;
const iters = process.argv.includes('--iters') ? parseInt(process.argv[process.argv.indexOf('--iters') + 1], 10) || 2 : 2;
const maxRuns = process.argv.includes('--max-runs') ? parseInt(process.argv[process.argv.indexOf('--max-runs') + 1], 10) || 2 : 2;
const timeoutSeconds = process.argv.includes('--timeout-seconds')
  ? parseInt(process.argv[process.argv.indexOf('--timeout-seconds') + 1], 10) || 30
  : 30;
const profile = process.argv.includes('--profile') ? String(process.argv[process.argv.indexOf('--profile') + 1] || 'smoke') : 'smoke';
const externalIntercept = process.argv.includes('--external-intercept');
const requireReadyIntercept = process.argv.includes('--require-ready-intercept');

function runNodeScript(script, args = [], env = process.env) {
  const result = spawnSync(process.execPath, [script, ...args], {
    cwd: rootDir,
    encoding: 'utf-8',
    timeout: 120000,
    env,
  });
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || `${script} failed`);
  }
  return result.stdout.trim();
}

runNodeScript('scripts/collect-opencl-intercept-manifest.mjs');

const interceptManifestPath = join(adrenoDir, 'opencl-intercept-manifest.json');
const interceptManifest = JSON.parse(readFileSync(interceptManifestPath, 'utf-8'));
const interceptRequestedButUnavailable = externalIntercept && !interceptManifest.runtimeReady;

if (requireReadyIntercept && interceptRequestedButUnavailable) {
  throw new Error(`external intercept requested but no ready intercept library found: ${interceptManifestPath}`);
}

const benchmarkArgs = [
  'scripts/benchmark-opencl-perf.mjs',
  '--profile', profile,
  '--warmup', String(warmup),
  '--iters', String(iters),
  '--timeout-seconds', String(timeoutSeconds),
  '--max-runs', String(maxRuns),
];

if (sustained) {
  benchmarkArgs.push('--sustained', '--minutes', String(minutes));
}
if (!execute) {
  benchmarkArgs.push('--dry-run-policy');
}
if (externalIntercept) {
  benchmarkArgs.push('--external-intercept');
  if (interceptManifest.selectedLibrary) {
    benchmarkArgs.push('--intercept-lib', interceptManifest.selectedLibrary);
  }
  if (interceptManifest.selectedConfig) {
    benchmarkArgs.push('--intercept-config', interceptManifest.selectedConfig);
  }
}

runNodeScript(benchmarkArgs[0], benchmarkArgs.slice(1));
runNodeScript('scripts/collect-opencl-profiler-contract.mjs');

const laneManifest = {
  timestamp: new Date().toISOString(),
  type: sustained ? 'opencl_controlled_sustained_lane' : 'opencl_controlled_profile_lane',
  route: 'mesa_rusticl_kgsl',
  execute,
  sustained,
  benchmark: {
    profile,
    warmup,
    iters,
    maxRuns,
    minutes: sustained ? minutes : 0,
    timeoutSeconds,
  },
  intercept: {
    requested: externalIntercept,
    requireReady: requireReadyIntercept,
    manifestPath: interceptManifestPath,
    ready: interceptManifest.ready,
    runtimeReady: interceptManifest.runtimeReady,
    selectedLibrary: interceptManifest.selectedLibrary,
    selectedConfig: interceptManifest.selectedConfig,
    status: interceptRequestedButUnavailable ? 'requested_unavailable' : (externalIntercept ? 'requested_ready' : 'not_requested'),
  },
  policyArtifact: join(
    adrenoDir,
    sustained ? 'opencl-sustained-policy.json' : 'opencl-performance-policy.json',
  ),
  profilerContractArtifact: join(adrenoDir, 'opencl-profiler-contract.json'),
};

const outPath = join(adrenoDir, sustained ? 'opencl-controlled-sustained-lane.json' : 'opencl-controlled-profile-lane.json');
writeFileSync(outPath, JSON.stringify(laneManifest, null, 2));
console.log(JSON.stringify({ status: 'ok', outPath, execute, sustained, externalIntercept, interceptReady: interceptManifest.ready }, null, 2));
