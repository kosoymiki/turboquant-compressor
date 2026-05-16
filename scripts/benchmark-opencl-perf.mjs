#!/usr/bin/env node
/**
 * OpenCL performance benchmark orchestrator.
 * Runs tq_opencl_cli benchmark, adds device/thermal metadata, writes forensics.
 * Supports --sustained --minutes N for thermal stability testing.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { spawnSync } from 'child_process';
import { existsSync, writeFileSync, readFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const buildDir = join(rootDir, 'native', 'opencl', 'build');
const forensicsDir = join(rootDir, 'forensics', 'adreno');
const cliBin = join(buildDir, 'tq_opencl_cli');
const kernelDir = join(rootDir, 'native', 'opencl', 'kernels');

mkdirSync(forensicsDir, { recursive: true });

const openclLdPath = [
  '/system/vendor/lib64',
  '/vendor/lib64',
  `${process.env.PREFIX ?? '/data/data/com.termux/files/usr'}/lib`,
  process.env.LD_LIBRARY_PATH || '',
].filter(Boolean).join(':');

const sustained = process.argv.includes('--sustained');
const minutesIdx = process.argv.indexOf('--minutes');
const minutes = minutesIdx >= 0 ? parseInt(process.argv[minutesIdx + 1]) || 5 : 5;
const warmup = process.argv.includes('--warmup') ? parseInt(process.argv[process.argv.indexOf('--warmup') + 1]) || 10 : 10;
const iters = process.argv.includes('--iters') ? parseInt(process.argv[process.argv.indexOf('--iters') + 1]) || 100 : 100;

function getDeviceInfo() {
  const info = {};
  const props = ['ro.product.model', 'ro.soc.model', 'ro.board.platform', 'ro.hardware.chipname'];
  for (const p of props) {
    try {
      const r = spawnSync('getprop', [p], { encoding: 'utf-8', timeout: 1000 });
      if (r.status === 0 && r.stdout.trim()) info[p] = r.stdout.trim();
    } catch {}
  }
  return info;
}

function getMemInfo() {
  try {
    const mem = readFileSync('/proc/meminfo', 'utf-8');
    const total = mem.match(/MemTotal:\s+(\d+)/);
    const avail = mem.match(/MemAvailable:\s+(\d+)/);
    return { totalKb: total ? parseInt(total[1]) : 0, availableKb: avail ? parseInt(avail[1]) : 0 };
  } catch { return {}; }
}

function getThermal() {
  const zones = [];
  for (let i = 0; i < 30; i++) {
    const path = `/sys/class/thermal/thermal_zone${i}/temp`;
    try {
      if (existsSync(path)) {
        const temp = parseInt(readFileSync(path, 'utf-8').trim());
        const typePath = `/sys/class/thermal/thermal_zone${i}/type`;
        const type = existsSync(typePath) ? readFileSync(typePath, 'utf-8').trim() : `zone${i}`;
        zones.push({ zone: i, type, temp_mc: temp });
      }
    } catch {}
  }
  return zones.length > 0 ? zones : null;
}

function runBenchmark(extraArgs = []) {
  if (!existsSync(cliBin)) {
    return { error: 'tq_opencl_cli not built', state: 'ADRENO_NATIVE_NOT_BUILT' };
  }

  const args = ['benchmark', '--warmup', String(warmup), '--iters', String(iters), '--kernel-dir', kernelDir, ...extraArgs];
  const r = spawnSync(cliBin, args, {
    encoding: 'utf-8',
    timeout: 600000,
    env: { ...process.env, LD_LIBRARY_PATH: openclLdPath },
  });

  try {
    return JSON.parse(r.stdout);
  } catch {
    return { error: 'parse_failed', stdout: (r.stdout || '').slice(0, 500), stderr: (r.stderr || '').slice(0, 500) };
  }
}

if (!sustained) {
  // Single benchmark run
  const deviceInfo = getDeviceInfo();
  const memBefore = getMemInfo();
  const thermalBefore = getThermal();

  const result = runBenchmark();

  const memAfter = getMemInfo();
  const thermalAfter = getThermal();

  const report = {
    timestamp: new Date().toISOString(),
    type: 'performance',
    device: deviceInfo,
    memory: { before: memBefore, after: memAfter },
    thermal: { before: thermalBefore, after: thermalAfter },
    ...result,
  };

  const outPath = join(forensicsDir, 'opencl-performance-report.json');
  writeFileSync(outPath, JSON.stringify(report, null, 2));
  console.log(`[benchmark-opencl-perf] Report: ${outPath}`);
  console.log(`[benchmark-opencl-perf] all_pass: ${report.all_pass}, results: ${(report.results || []).length}`);
  process.exit(report.all_pass ? 0 : 1);

} else {
  // Sustained benchmark
  console.log(`[sustained] Running ${minutes}-minute sustained benchmark...`);
  const runs = [];
  const startTime = Date.now();
  const endTime = startTime + minutes * 60 * 1000;
  let runIndex = 0;

  while (Date.now() < endTime) {
    const thermalNow = getThermal();
    const memNow = getMemInfo();
    const t0 = Date.now();

    const result = runBenchmark();

    const elapsed = Date.now() - t0;
    runs.push({
      run_index: runIndex++,
      elapsed_ms: elapsed,
      timestamp: new Date().toISOString(),
      thermal: thermalNow,
      mem_available_kb: memNow.availableKb,
      all_pass: result.all_pass,
      results_summary: (result.results || []).map(r => ({
        kernel: r.kernel,
        shape: r.shape,
        p50_ms: r.p50_ms,
        p95_ms: r.p95_ms,
        parity: r.parity,
      })),
    });

    const minutesDone = ((Date.now() - startTime) / 60000).toFixed(1);
    console.log(`[sustained] Run ${runIndex}: ${minutesDone}/${minutes} min, all_pass=${result.all_pass}`);
  }

  // Compute slowdown ratio
  const firstP50s = {};
  const lastP50s = {};
  if (runs.length >= 2) {
    for (const r of runs[0].results_summary) {
      const key = `${r.kernel}_${r.shape.tokens}_${r.shape.bits}`;
      firstP50s[key] = r.p50_ms;
    }
    for (const r of runs[runs.length - 1].results_summary) {
      const key = `${r.kernel}_${r.shape.tokens}_${r.shape.bits}`;
      lastP50s[key] = r.p50_ms;
    }
  }

  const slowdownRatios = {};
  for (const [key, first] of Object.entries(firstP50s)) {
    if (lastP50s[key] && first > 0) {
      slowdownRatios[key] = lastP50s[key] / first;
    }
  }

  const report = {
    timestamp: new Date().toISOString(),
    type: 'sustained',
    device: getDeviceInfo(),
    duration_minutes: minutes,
    total_runs: runs.length,
    slowdown_ratios: slowdownRatios,
    thermal_start: runs[0]?.thermal,
    thermal_end: runs[runs.length - 1]?.thermal,
    mem_start_kb: runs[0]?.mem_available_kb,
    mem_end_kb: runs[runs.length - 1]?.mem_available_kb,
    runs,
  };

  const outPath = join(forensicsDir, 'opencl-sustained-report.json');
  writeFileSync(outPath, JSON.stringify(report, null, 2));
  console.log(`[sustained] Report: ${outPath}`);
  console.log(`[sustained] ${runs.length} runs over ${minutes} minutes`);
  process.exit(0);
}
