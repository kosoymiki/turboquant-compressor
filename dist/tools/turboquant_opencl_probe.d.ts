/**
 * MCP tool: turboquant_opencl_probe — OpenCL/Adreno availability detection.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
export declare const turboquantOpenclProbeSchema: {
    description: string;
    inputSchema: {
        type: "object";
        properties: {
            deep: {
                type: string;
                description: string;
            };
            timeoutMs: {
                type: string;
                description: string;
                minimum: number;
                maximum: number;
            };
        };
        required: string[];
    };
};
export declare function turboquantOpenclProbe(args: {
    deep?: boolean;
    timeoutMs?: number;
}): object;
