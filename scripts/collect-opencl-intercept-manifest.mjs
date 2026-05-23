#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const adrenoDir = join(rootDir, 'forensics', 'adreno');
mkdirSync(adrenoDir, { recursive: true });

function loadJson(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    return null;
  }
}

const profilerContract = loadJson(join(adrenoDir, 'opencl-profiler-contract.json'));
const explicitLib = process.env.TQ_OPENCL_INTERCEPT_LIB || '';
const explicitConfig = process.env.TQ_OPENCL_INTERCEPT_CONFIG || '';
const candidateLibs = [
  explicitLib,
  join(process.env.PREFIX || '/data/data/com.termux/files/usr', 'lib', 'libOpenCLIntercept.so'),
  join(rootDir, 'third_party', 'OpenCL-Intercept-Layer', 'libOpenCLIntercept.so'),
].filter(Boolean);
const candidateConfigs = [
  explicitConfig,
  join(rootDir, 'forensics', 'adreno', 'opencl-intercept.config'),
  join(rootDir, 'third_party', 'OpenCL-Intercept-Layer', 'intercept.conf'),
].filter(Boolean);

const selectedLibrary = candidateLibs.find((p) => existsSync(p)) || null;
const selectedConfig = candidateConfigs.find((p) => existsSync(p)) || null;
const runtimeReady = Boolean(selectedLibrary);
const contractReady = Boolean(selectedLibrary || selectedConfig);
const selectedBy = selectedLibrary === explicitLib && explicitLib
  ? 'explicit_env'
  : (selectedLibrary || selectedConfig ? 'auto_discovery' : 'none');

const manifest = {
  timestamp: new Date().toISOString(),
  type: 'opencl_external_intercept_manifest',
  route: 'mesa_rusticl_kgsl',
  integrationMode: 'formalized_external_tool',
  ready: contractReady,
  runtimeReady,
  selectedLibrary,
  selectedConfig,
  selectedBy,
  candidates: {
    libraries: candidateLibs,
    configs: candidateConfigs,
  },
  recommendedEnv: {
    TQ_OPENCL_INTERCEPT_LIB: selectedLibrary,
    TQ_OPENCL_INTERCEPT_CONFIG: selectedConfig,
    OCL_INTERCEPT_CONFIG_FILE: selectedConfig,
    LD_PRELOAD: selectedLibrary,
  },
  profilerContractSeen: Boolean(profilerContract),
};

const outPath = join(adrenoDir, 'opencl-intercept-manifest.json');
writeFileSync(outPath, JSON.stringify(manifest, null, 2));
console.log(JSON.stringify({ status: 'ok', outPath, ready: contractReady, runtimeReady, selectedLibrary, selectedConfig }, null, 2));
