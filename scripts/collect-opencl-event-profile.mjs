#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { existsSync, mkdirSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics', 'adreno');
const safeRunner = join(rootDir, 'scripts', 'safe-runtime-pack-run.sh');
const cliBin = process.env.TQ_OPENCL_CLI || join(process.env.HOME || '', '.cache', 'turboquant', 'native-opencl-build', 'tq_opencl_cli');
const kernelDir = join(rootDir, 'native', 'opencl', 'kernels');
mkdirSync(forensicsDir, { recursive: true });

const warmup = Number(process.env.TQ_EVENT_PROFILE_WARMUP || 1);
const iters = Number(process.env.TQ_EVENT_PROFILE_ITERS || 2);
const profile = process.env.TQ_EVENT_PROFILE_MODE || 'smoke';
const timeoutMs = Number(process.env.TQ_EVENT_PROFILE_TIMEOUT_MS || 120000);

function runBenchmark() {
  const args = ['benchmark', '--profile', profile, '--warmup', String(warmup), '--iters', String(iters), '--kernel-dir', kernelDir];
  const proc = existsSync(safeRunner)
    ? spawnSync(safeRunner, args, {
        encoding: 'utf-8',
        timeout: timeoutMs,
        env: {
          ...process.env,
          TQ_OPENCL_SILENCE_STDERR: '1',
          TQ_OPENCL_TIMEOUT_SECONDS: String(Math.max(30, Math.ceil(timeoutMs / 1000))),
          TQ_OPENCL_BENCH_PROFILE: profile,
        },
      })
    : spawnSync(cliBin, args, {
        encoding: 'utf-8',
        timeout: timeoutMs,
        env: { ...process.env, TQ_OPENCL_BENCH_PROFILE: profile },
      });

  return {
    status: proc.status ?? 1,
    stdout: proc.stdout || '',
    stderr: proc.stderr || '',
  };
}

const run = runBenchmark();
let parsed = null;
let parseError = null;
try {
  parsed = JSON.parse(run.stdout);
} catch (err) {
  parseError = String(err);
}

const eventProfile = {
  timestamp: new Date().toISOString(),
  type: 'opencl_event_profile',
  mode: profile,
  warmup,
  iters,
  timeout_ms: timeoutMs,
  command_ok: run.status === 0,
  parsed: !!parsed,
  parse_error: parseError,
  device: parsed?.device ?? null,
  all_pass: parsed?.all_pass ?? false,
  kernels: (parsed?.results || []).map((r) => ({
    kernel: r.kernel,
    shape: r.shape,
    parity: r.parity,
    mean_ms: r.mean_ms,
    p50_ms: r.p50_ms,
    p95_ms: r.p95_ms,
    host_mean_ms: r.host_mean_ms,
    host_p95_ms: r.host_p95_ms,
    cpu_mean_ms: r.cpu_mean_ms,
    speedup_vs_cpu: r.speedup_vs_cpu,
    profiling_signal: {
      event_overhead_delta_ms: Number((r.host_mean_ms - r.mean_ms).toFixed(4)),
      event_to_host_ratio: r.host_mean_ms > 0 ? Number((r.mean_ms / r.host_mean_ms).toFixed(4)) : null,
    },
  })),
  stderr_preview: run.stderr.slice(0, 4000),
};

const outPath = join(forensicsDir, 'opencl-event-profile.json');
writeFileSync(outPath, JSON.stringify(eventProfile, null, 2));
console.log(JSON.stringify({ status: parsed?.all_pass ? 'ok' : 'error', outPath, kernels: eventProfile.kernels.length }, null, 2));
process.exit(parsed?.all_pass ? 0 : 1);
