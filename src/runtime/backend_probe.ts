/**
 * Backend probe — detect available runtimes without crashing.
 * Quick mode (default): only checks `command -v python3`, no heavy imports.
 * Deep mode: attempts bounded torch/triton/vllm imports.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { spawnSync } from 'child_process';

export interface BackendProbeOptions {
  deep?: boolean;
  timeoutMs?: number;
}

export interface BackendProbeResult {
  termux: boolean;
  node: string;
  python: 'available' | 'missing';
  torch: 'available' | 'missing' | 'not_checked';
  triton: 'available' | 'missing' | 'not_checked';
  cuda: 'available' | 'missing' | 'not_checked';
  vllm: 'available' | 'missing' | 'not_checked';
  recommendedBackend: 'typescript_cpu' | 'python_cpu' | 'python_gpu' | 'triton_gpu';
  probeMode: 'quick' | 'deep';
  elapsedMs: number;
  warnings: string[];
}

function tryRun(cmd: string, args: string[], timeoutMs: number): { ok: boolean; stdout: string } {
  const res = spawnSync(cmd, args, {
    encoding: 'utf8',
    timeout: timeoutMs,
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: false,
  });
  if (res.error || res.status !== 0 || res.signal) return { ok: false, stdout: '' };
  return { ok: true, stdout: String(res.stdout ?? '').trim() };
}

function commandExists(command: string, timeoutMs: number): boolean {
  const res = spawnSync('sh', ['-c', `command -v ${command}`], {
    encoding: 'utf8',
    timeout: timeoutMs,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  return !res.error && res.status === 0 && String(res.stdout ?? '').trim().length > 0;
}

export function probeBackends(options: BackendProbeOptions = {}): BackendProbeResult {
  const started = Date.now();
  const deep = options.deep === true;
  const timeoutMs = Math.max(100, Math.min(options.timeoutMs ?? 500, 3000));

  const result: BackendProbeResult = {
    termux: process.env['PREFIX']?.includes('com.termux') ?? false,
    node: process.version,
    python: 'missing',
    torch: deep ? 'missing' : 'not_checked',
    triton: deep ? 'missing' : 'not_checked',
    cuda: deep ? 'missing' : 'not_checked',
    vllm: deep ? 'missing' : 'not_checked',
    recommendedBackend: 'typescript_cpu',
    probeMode: deep ? 'deep' : 'quick',
    elapsedMs: 0,
    warnings: [],
  };

  result.python = commandExists('python3', timeoutMs) ? 'available' : 'missing';

  if (!deep) {
    if (result.python === 'missing') {
      result.warnings.push('Python not found; Python reference validation unavailable');
    }
    result.warnings.push('Quick probe: pass deep=true to check torch/triton/vllm');
    result.elapsedMs = Date.now() - started;
    return result;
  }

  // Deep mode — bounded imports
  if (result.python === 'available') {
    result.torch = tryRun('python3', ['-c', 'import torch; print("ok")'], timeoutMs).ok ? 'available' : 'missing';

    if (result.torch === 'available') {
      const cuda = tryRun('python3', ['-c', 'import torch; print("yes" if torch.cuda.is_available() else "no")'], timeoutMs);
      result.cuda = cuda.ok && cuda.stdout === 'yes' ? 'available' : 'missing';
      result.vllm = tryRun('python3', ['-c', 'import vllm; print("ok")'], timeoutMs).ok ? 'available' : 'missing';
    }

    if (result.cuda === 'available') {
      result.triton = tryRun('python3', ['-c', 'import triton; print("ok")'], timeoutMs).ok ? 'available' : 'missing';
    }
  }

  if (result.triton === 'available') {
    result.recommendedBackend = 'triton_gpu';
  } else if (result.cuda === 'available') {
    result.recommendedBackend = 'python_gpu';
  } else if (result.torch === 'available') {
    result.recommendedBackend = 'python_cpu';
  }

  if (result.termux && result.cuda !== 'available') {
    result.warnings.push('Triton/vLLM unavailable on standard Termux; using TypeScript CPU path');
  }
  if (result.python === 'missing') {
    result.warnings.push('Python not found; Python reference validation unavailable');
  }

  result.elapsedMs = Date.now() - started;
  return result;
}
