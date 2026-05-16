/**
 * Backend probe — detect available runtimes without crashing.
 * Quick mode (default): only checks `command -v python3`, no heavy imports.
 * Deep mode: attempts bounded torch/triton/vllm imports.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
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
export declare function probeBackends(options?: BackendProbeOptions): BackendProbeResult;
