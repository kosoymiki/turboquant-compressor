#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const adrenoDir = join(rootDir, 'forensics', 'adreno');
const safeRunner = join(rootDir, 'scripts', 'safe-runtime-pack-run.sh');
mkdirSync(adrenoDir, { recursive: true });

function loadJson(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    return null;
  }
}

function classifyFailure(status, stderr, parsed) {
  const text = `${stderr || ''}\n${parsed?.error || ''}`;
  if (status === 124 || /timed out/i.test(text)) return 'timeout';
  if (/segmentation fault|sigsegv|fatal signal 11/i.test(text)) return 'sigsegv';
  if (/bus error|sigbus|fatal signal 7/i.test(text)) return 'sigbus';
  if (/abort|sigabrt|fatal signal 6/i.test(text)) return 'sigabrt';
  if (/cannot open shared object file|dlopen|LD_PRELOAD|OpenCLFileName/i.test(text)) return 'loader_or_preload_failure';
  if (/No OpenCL platforms|CL_PLATFORM_NOT_FOUND_KHR/i.test(text)) return 'opencl_platform_missing';
  return status === 0 ? 'ok' : 'probe_failed';
}

const command = process.argv.includes('--command')
  ? String(process.argv[process.argv.indexOf('--command') + 1] || 'probe')
  : 'probe';
const timeoutSeconds = process.argv.includes('--timeout-seconds')
  ? parseInt(process.argv[process.argv.indexOf('--timeout-seconds') + 1], 10) || 15
  : 15;

if (!['probe', 'frontier-smoke'].includes(command)) {
  throw new Error(`unsupported intercept smoke command: ${command}`);
}
if (!existsSync(safeRunner)) {
  throw new Error(`missing safe runner: ${safeRunner}`);
}

spawnSync(process.execPath, ['scripts/collect-opencl-intercept-manifest.mjs'], {
  cwd: rootDir,
  encoding: 'utf-8',
  timeout: 120000,
});

const manifestPath = join(adrenoDir, 'opencl-intercept-manifest.json');
const manifest = loadJson(manifestPath);
if (!manifest?.runtimeReady || !manifest?.selectedLibrary) {
  const outPath = join(adrenoDir, 'opencl-intercept-smoke.json');
  const artifact = {
    timestamp: new Date().toISOString(),
    type: 'opencl_intercept_smoke',
    route: 'mesa_rusticl_kgsl',
    command,
    timeoutSeconds,
    manifestPath,
    runtimeReady: manifest?.runtimeReady ?? false,
    selectedLibrary: manifest?.selectedLibrary ?? null,
    selectedConfig: manifest?.selectedConfig ?? null,
    status: 'runtime_not_ready',
  };
  writeFileSync(outPath, JSON.stringify(artifact, null, 2));
  console.log(JSON.stringify({ status: 'ok', outPath, runtimeReady: false }, null, 2));
  process.exit(0);
}

const env = {
  ...process.env,
  TQ_OPENCL_TIMEOUT_SECONDS: String(timeoutSeconds),
  TQ_OPENCL_SILENCE_STDERR: '0',
  TQ_OPENCL_INTERCEPT_LIB: manifest.selectedLibrary,
  TQ_OPENCL_INTERCEPT_CONFIG: manifest.selectedConfig || '',
  OCL_INTERCEPT_CONFIG_FILE: manifest.selectedConfig || '',
  CLI_OpenCLFileName: '/data/data/com.termux/files/usr/lib/libOpenCL.so',
  LD_PRELOAD: manifest.selectedLibrary,
};

const run = spawnSync(safeRunner, [command], {
  cwd: rootDir,
  encoding: 'utf-8',
  timeout: Math.max(30000, timeoutSeconds * 2000),
  env,
});

let parsedStdout = null;
try {
  parsedStdout = JSON.parse(run.stdout || '');
} catch {}

const artifact = {
  timestamp: new Date().toISOString(),
  type: 'opencl_intercept_smoke',
  route: 'mesa_rusticl_kgsl',
  command,
  timeoutSeconds,
  manifestPath,
  runtimeReady: true,
  selectedLibrary: manifest.selectedLibrary,
  selectedConfig: manifest.selectedConfig,
  statusCode: run.status,
  crashClass: classifyFailure(run.status, run.stderr || '', parsedStdout),
  stdoutJson: parsedStdout,
  stdout: (run.stdout || '').slice(0, 16000),
  stderr: (run.stderr || '').slice(0, 16000),
};

const outPath = join(adrenoDir, 'opencl-intercept-smoke.json');
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(JSON.stringify({
  status: 'ok',
  outPath,
  crashClass: artifact.crashClass,
  statusCode: artifact.statusCode,
}, null, 2));
