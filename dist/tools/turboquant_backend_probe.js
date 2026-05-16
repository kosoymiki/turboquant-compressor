/**
 * MCP tool: turboquant_backend_probe — detect available runtimes.
 * Quick mode (default) never imports torch/triton/vLLM. Termux-safe.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
import { probeBackends } from '../runtime/backend_probe.js';
export function turboquantBackendProbe(input = {}) {
    return probeBackends(input);
}
export const turboquantBackendProbeSchema = {
    name: 'turboquant_backend_probe',
    description: 'Detect available TurboQuant backends. Quick mode (default) is Termux-safe and never imports torch/triton/vLLM. Pass deep=true for full check.',
    inputSchema: {
        type: 'object',
        properties: {
            deep: { type: 'boolean', description: 'If true, attempt bounded Python imports for torch/triton/vLLM. Default false.' },
            timeoutMs: { type: 'number', description: 'Per-check timeout ms (100-3000). Default 500.' },
        },
        required: [],
    },
};
