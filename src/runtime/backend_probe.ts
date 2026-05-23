/**
 * Backend probe — detect available runtimes without crashing.
 * Quick mode (default): only checks `command -v python3`, no heavy imports.
 * Deep mode: attempts bounded torch/triton/vllm imports.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { spawnSync } from 'child_process';
import { type OpenClProbeResult, probeOpenCl } from '../native/opencl_probe.js';
import { type ProductionPolicyAssessment, assessProductionPolicy } from '../native/production_policy.js';

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
  nativeBackend: OpenClProbeResult['recommendedBackend'] | 'diagnostic_only';
  recommendedBackend: OpenClProbeResult['recommendedBackend'] | 'diagnostic_only' | 'python_cpu' | 'python_gpu' | 'triton_gpu';
  executionOwner: 'native_cli_contract' | 'diagnostic_runtime';
  admissionSource: 'native_probe' | 'production_policy_only';
  productionBackendAllowed: boolean;
  production: ProductionPolicyAssessment;
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
  const nativeProbe = probeOpenCl({ deep, timeoutMs });

  const result: BackendProbeResult = {
    termux: process.env['PREFIX']?.includes('com.termux') ?? false,
    node: process.version,
    python: 'missing',
    torch: deep ? 'missing' : 'not_checked',
    triton: deep ? 'missing' : 'not_checked',
    cuda: deep ? 'missing' : 'not_checked',
    vllm: deep ? 'missing' : 'not_checked',
    nativeBackend: nativeProbe.production.productionReady
      ? nativeProbe.recommendedBackend
      : 'diagnostic_only',
    recommendedBackend: 'diagnostic_only',
    executionOwner: nativeProbe.production.productionReady ? 'native_cli_contract' : 'diagnostic_runtime',
    admissionSource: nativeProbe.runtimePackProbe || nativeProbe.loadable !== null
      ? 'native_probe'
      : 'production_policy_only',
    productionBackendAllowed: false,
    production: nativeProbe.production ?? assessProductionPolicy(),
    probeMode: deep ? 'deep' : 'quick',
    elapsedMs: 0,
    warnings: [...nativeProbe.warnings],
  };

  result.python = commandExists('python3', timeoutMs) ? 'available' : 'missing';

  if (!deep) {
    if (result.python === 'missing') {
      result.warnings.push('Python not found; Python reference validation unavailable');
    }
    result.recommendedBackend = result.nativeBackend;
    result.productionBackendAllowed = result.production.productionReady
      && result.production.allowedProductionRoutes.includes(result.production.productionRoute as typeof result.production.allowedProductionRoutes[number]);
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

  if (result.nativeBackend !== 'diagnostic_only') {
    result.recommendedBackend = result.nativeBackend;
  } else if (result.triton === 'available') {
    result.recommendedBackend = 'triton_gpu';
  } else if (result.cuda === 'available') {
    result.recommendedBackend = 'python_gpu';
  } else if (result.torch === 'available') {
    result.recommendedBackend = 'python_cpu';
  } else {
    result.recommendedBackend = 'typescript_cpu';
  }

  result.productionBackendAllowed = result.production.productionReady
    && result.production.allowedProductionRoutes.includes(result.production.productionRoute as typeof result.production.allowedProductionRoutes[number]);
  if (!result.productionBackendAllowed) {
    result.warnings.push('Detected backends are diagnostic-only until custom driver stack runtime evidence is present');
  }

  if (result.termux && result.cuda !== 'available') {
    result.warnings.push('Triton/vLLM unavailable on standard Termux; CPU probes stay diagnostic-only under custom driver policy');
  }
  if (result.python === 'missing') {
    result.warnings.push('Python not found; Python reference validation unavailable');
  }
  if (!result.production.productionReady) {
    result.warnings.push(`Production route blocked: ${result.production.blockers.join(', ')}`);
  }

  result.elapsedMs = Date.now() - started;
  return result;
}
