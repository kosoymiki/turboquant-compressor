#!/usr/bin/env node
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const adrenoDir = join(rootDir, 'forensics', 'adreno');
mkdirSync(adrenoDir, { recursive: true });

const currentPath = join(adrenoDir, 'opencl-event-profile.json');
const baselinePath = join(adrenoDir, 'opencl-performance-baseline.json');
const reportPath = join(adrenoDir, 'opencl-performance-regression.json');
const updateBaseline = process.argv.includes('--update-baseline');
const maxRegression = Number(process.env.TQ_PERF_REGRESSION_MAX || 0.03);

function loadJson(path) {
  return JSON.parse(readFileSync(path, 'utf-8'));
}

if (!existsSync(currentPath)) {
  console.error(`FAIL: missing current event profile: ${currentPath}`);
  process.exit(1);
}

if (updateBaseline || !existsSync(baselinePath)) {
  copyFileSync(currentPath, baselinePath);
  const bootstrap = {
    timestamp: new Date().toISOString(),
    status: 'baseline_updated',
    baselinePath,
    source: currentPath,
  };
  writeFileSync(reportPath, JSON.stringify(bootstrap, null, 2));
  console.log(JSON.stringify(bootstrap, null, 2));
  process.exit(0);
}

const current = loadJson(currentPath);
const baseline = loadJson(baselinePath);

const baselineMap = new Map((baseline.kernels || []).map((k) => [`${k.kernel}:${JSON.stringify(k.shape)}`, k]));
const checks = [];
let fail = false;

for (const currentKernel of current.kernels || []) {
  const key = `${currentKernel.kernel}:${JSON.stringify(currentKernel.shape)}`;
  const base = baselineMap.get(key);
  if (!base) continue;
  const delta = base.mean_ms > 0 ? (currentKernel.mean_ms - base.mean_ms) / base.mean_ms : 0;
  const passed = delta <= maxRegression;
  if (!passed) fail = true;
  checks.push({
    key,
    kernel: currentKernel.kernel,
    shape: currentKernel.shape,
    baseline_mean_ms: base.mean_ms,
    current_mean_ms: currentKernel.mean_ms,
    regression_ratio: Number(delta.toFixed(6)),
    threshold: maxRegression,
    passed,
  });
}

const report = {
  timestamp: new Date().toISOString(),
  status: fail ? 'regression_detected' : 'ok',
  threshold: maxRegression,
  currentPath,
  baselinePath,
  checks,
};

writeFileSync(reportPath, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
process.exit(fail ? 1 : 0);
