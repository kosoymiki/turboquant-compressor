/**
 * MCP tool: turboquant_backend_probe — detect available runtimes.
 * Quick mode (default) never imports torch/triton/vLLM. Termux-safe.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
import { BackendProbeOptions, BackendProbeResult } from '../runtime/backend_probe.js';
export declare function turboquantBackendProbe(input?: BackendProbeOptions): BackendProbeResult;
export declare const turboquantBackendProbeSchema: {
    name: string;
    description: string;
    inputSchema: {
        type: "object";
        properties: {
            deep: {
                type: "boolean";
                description: string;
            };
            timeoutMs: {
                type: "number";
                description: string;
            };
        };
        required: string[];
    };
};
