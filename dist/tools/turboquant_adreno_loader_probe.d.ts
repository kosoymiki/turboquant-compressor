/**
 * MCP tool: turboquant_adreno_loader_probe — Android linker namespace diagnosis.
 * Reports exact loader state for vendor OpenCL on this device.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
export declare const turboquantAdrenoLoaderProbeSchema: {
    description: string;
    inputSchema: {
        type: "object";
        properties: {
            deep: {
                type: string;
                description: string;
            };
        };
        required: string[];
    };
};
export declare function turboquantAdrenoLoaderProbe(args: {
    deep?: boolean;
}): object;
