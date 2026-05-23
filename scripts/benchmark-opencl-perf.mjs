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
const safeRuntimePackRunner = join(rootDir, 'scripts', 'safe-runtime-pack-run.sh');
const kernelDir = join(rootDir, 'native', 'opencl', 'kernels');
const termuxPrefix = process.env.PREFIX || dirname(dirname(process.execPath));

mkdirSync(forensicsDir, { recursive: true });

const openclLdPath = [
  '/system/vendor/lib64',
  '/vendor/lib64',
  `${termuxPrefix}/lib`,
  process.env.LD_LIBRARY_PATH || '',
].filter(Boolean).join(':');

const sustained = process.argv.includes('--sustained');
const minutesIdx = process.argv.indexOf('--minutes');
const minutes = minutesIdx >= 0 ? parseInt(process.argv[minutesIdx + 1]) || 5 : 5;
const warmup = process.argv.includes('--warmup') ? parseInt(process.argv[process.argv.indexOf('--warmup') + 1]) || 10 : 10;
const iters = process.argv.includes('--iters') ? parseInt(process.argv[process.argv.indexOf('--iters') + 1]) || 100 : 100;
const profile = process.argv.includes('--profile') ? String(process.argv[process.argv.indexOf('--profile') + 1] || 'default') : 'default';
const maxRuns = process.argv.includes('--max-runs') ? parseInt(process.argv[process.argv.indexOf('--max-runs') + 1]) || 0 : 0;
const timeoutSeconds = process.argv.includes('--timeout-seconds')
  ? parseInt(process.argv[process.argv.indexOf('--timeout-seconds') + 1]) || 45
  : 45;
const dryRunPolicy = process.argv.includes('--dry-run-policy');
const externalIntercept = process.argv.includes('--external-intercept');
const interceptLib = process.argv.includes('--intercept-lib')
  ? String(process.argv[process.argv.indexOf('--intercept-lib') + 1] || '')
  : (process.env.TQ_OPENCL_INTERCEPT_LIB || '');
const interceptConfig = process.argv.includes('--intercept-config')
  ? String(process.argv[process.argv.indexOf('--intercept-config') + 1] || '')
  : (process.env.TQ_OPENCL_INTERCEPT_CONFIG || '');
const thermalSoftLimitMc = process.argv.includes('--thermal-soft-limit-mc')
  ? parseInt(process.argv[process.argv.indexOf('--thermal-soft-limit-mc') + 1]) || 65000
  : 65000;
const thermalHardLimitMc = process.argv.includes('--thermal-hard-limit-mc')
  ? parseInt(process.argv[process.argv.indexOf('--thermal-hard-limit-mc') + 1]) || 70000
  : 70000;
const cooldownSoftMs = process.argv.includes('--cooldown-soft-ms')
  ? parseInt(process.argv[process.argv.indexOf('--cooldown-soft-ms') + 1]) || 2000
  : 2000;
const cooldownHardMs = process.argv.includes('--cooldown-hard-ms')
  ? parseInt(process.argv[process.argv.indexOf('--cooldown-hard-ms') + 1]) || 5000
  : 5000;

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

function sleepMs(ms) {
  const delay = Math.max(0, Number(ms) || 0);
  if (delay === 0) return;
  const start = Date.now();
  while (Date.now() - start < delay) {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, Math.min(250, delay));
  }
}

function getThermalPeak(thermal) {
  if (!Array.isArray(thermal) || thermal.length === 0) return null;
  const valid = thermal
    .map((zone) => Number(zone?.temp_mc))
    .filter((temp) => Number.isFinite(temp) && temp > 0);
  return valid.length > 0 ? Math.max(...valid) : null;
}

function computeThermalThrottleAction(thermal) {
  const peakMc = getThermalPeak(thermal);
  if (!Number.isFinite(peakMc)) {
    return { peakMc: null, level: 'unknown', cooldownMs: 0 };
  }
  if (peakMc >= thermalHardLimitMc) {
    return { peakMc, level: 'hard', cooldownMs: cooldownHardMs };
  }
  if (peakMc >= thermalSoftLimitMc) {
    return { peakMc, level: 'soft', cooldownMs: cooldownSoftMs };
  }
  return { peakMc, level: 'none', cooldownMs: 0 };
}

function runBenchmark(extraArgs = []) {
  if (!existsSync(cliBin) && !existsSync(safeRuntimePackRunner)) {
    return { error: 'tq_opencl_cli not built', state: 'ADRENO_NATIVE_NOT_BUILT' };
  }

  const args = ['benchmark', '--warmup', String(warmup), '--iters', String(iters), '--kernel-dir', kernelDir, ...extraArgs];
  const runtimeEnv = {
    ...process.env,
    TQ_OPENCL_SILENCE_STDERR: '1',
    TQ_OPENCL_TIMEOUT_SECONDS: String(timeoutSeconds),
    TQ_OPENCL_BENCH_PROFILE: profile,
  };
  if (externalIntercept) {
    if (interceptLib) {
      runtimeEnv.TQ_OPENCL_INTERCEPT_LIB = interceptLib;
      runtimeEnv.LD_PRELOAD = runtimeEnv.LD_PRELOAD ? `${interceptLib}:${runtimeEnv.LD_PRELOAD}` : interceptLib;
    }
    if (interceptConfig) {
      runtimeEnv.TQ_OPENCL_INTERCEPT_CONFIG = interceptConfig;
      runtimeEnv.OCL_INTERCEPT_CONFIG_FILE = interceptConfig;
    }
  }
  const useSafeRuntimePack = existsSync(safeRuntimePackRunner);
  const r = useSafeRuntimePack
    ? spawnSync(safeRuntimePackRunner, args, {
        encoding: 'utf-8',
        timeout: 600000,
        env: runtimeEnv,
      })
    : spawnSync(cliBin, args, {
        encoding: 'utf-8',
        timeout: 600000,
        env: { ...runtimeEnv, LD_LIBRARY_PATH: openclLdPath },
      });

  try {
    const parsed = JSON.parse(r.stdout);
    if (parsed?.device) {
      const gpu = String(parsed.device.gpu || '');
      parsed.device.is_adreno = parsed.device.is_adreno || /adreno|fd7/i.test(gpu);
    }
    return parsed;
  } catch {
    return { error: 'parse_failed', stdout: (r.stdout || '').slice(0, 500), stderr: (r.stderr || '').slice(0, 500) };
  }
}

const controlledLane = {
  profile,
  timeout_seconds: timeoutSeconds,
  external_intercept: externalIntercept,
  intercept_lib: interceptLib || null,
  intercept_config: interceptConfig || null,
  max_runs: maxRuns > 0 ? maxRuns : null,
};

if (dryRunPolicy) {
  const policy = {
    timestamp: new Date().toISOString(),
    type: sustained ? 'controlled_sustained_policy' : 'controlled_performance_policy',
    route: 'mesa_rusticl_kgsl',
    controlled_lane: controlledLane,
    thermal_policy: {
      soft_limit_mc: thermalSoftLimitMc,
      hard_limit_mc: thermalHardLimitMc,
      cooldown_soft_ms: cooldownSoftMs,
      cooldown_hard_ms: cooldownHardMs,
    },
    benchmark_policy: {
      warmup,
      iters,
      minutes: sustained ? minutes : 0,
    },
  };
  const outPath = join(forensicsDir, sustained ? 'opencl-sustained-policy.json' : 'opencl-performance-policy.json');
  writeFileSync(outPath, JSON.stringify(policy, null, 2));
  console.log(JSON.stringify({ status: 'ok', outPath, controlled_lane: controlledLane }, null, 2));
  process.exit(0);
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
    route: 'mesa_rusticl_kgsl',
    controlled_lane: controlledLane,
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
  const throttleActions = [];
  const startTime = Date.now();
  const endTime = startTime + minutes * 60 * 1000;
  let runIndex = 0;

  while (Date.now() < endTime) {
    if (maxRuns > 0 && runIndex >= maxRuns) {
      break;
    }
    const thermalNow = getThermal();
    const memNow = getMemInfo();
    const throttleAction = computeThermalThrottleAction(thermalNow);
    if (throttleAction.cooldownMs > 0) {
      throttleActions.push({
        run_index: runIndex,
        timestamp: new Date().toISOString(),
        ...throttleAction,
      });
      console.log(
        `[sustained] thermal ${throttleAction.level} throttle at ${throttleAction.peakMc} mc, cooldown ${throttleAction.cooldownMs} ms`,
      );
      sleepMs(throttleAction.cooldownMs);
    }
    const t0 = Date.now();

    const result = runBenchmark();

    const elapsed = Date.now() - t0;
    runs.push({
      run_index: runIndex++,
      elapsed_ms: elapsed,
      timestamp: new Date().toISOString(),
      thermal: thermalNow,
      thermal_peak_mc: throttleAction.peakMc,
      throttle_level: throttleAction.level,
      cooldown_ms: throttleAction.cooldownMs,
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
    route: 'mesa_rusticl_kgsl',
    controlled_lane: controlledLane,
    device: getDeviceInfo(),
    duration_minutes: minutes,
    total_runs: runs.length,
    thermal_policy: {
      soft_limit_mc: thermalSoftLimitMc,
      hard_limit_mc: thermalHardLimitMc,
      cooldown_soft_ms: cooldownSoftMs,
      cooldown_hard_ms: cooldownHardMs,
    },
    throttle_actions: throttleActions,
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
