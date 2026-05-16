/**
 * MCP tool: turboquant_quantize — expose TurboQuant codebook + quantization.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
export interface TurboQuantQuantizeInput {
    /** Action: 'compute_codebook' | 'quantize_vector' | 'benchmark_mse' */
    action: 'compute_codebook' | 'quantize_vector' | 'benchmark_mse';
    /** Dimension (required for all actions) */
    dimension: number;
    /** Bits per coordinate (1-6, default 4) */
    bits?: number;
    /** Vector to quantize (for quantize_vector) */
    vector?: number[];
    /** Rotation seed (default 42) */
    seed?: number;
}
export interface TurboQuantQuantizeOutput {
    action: string;
    dimension: number;
    bits: number;
    codebook?: {
        centroids: number[];
        boundaries: number[];
        msePerCoord: number;
        mseTotal: number;
    };
    quantized?: {
        indices: number[];
        norm: number;
        reconstructed: number[];
        mse: number;
        cosineSimilarity: number;
    };
    benchmark?: {
        msePerCoord: number;
        mseTotal: number;
        compressionRatio: number;
        bitsPerDimension: number;
    };
}
export declare function turboquantQuantize(input: TurboQuantQuantizeInput): TurboQuantQuantizeOutput;
/** MCP tool schema for turboquant_quantize */
export declare const turboquantQuantizeSchema: {
    name: string;
    description: string;
    inputSchema: {
        type: "object";
        properties: {
            action: {
                type: "string";
                enum: string[];
                description: string;
            };
            dimension: {
                type: "number";
                description: string;
            };
            bits: {
                type: "number";
                description: string;
            };
            vector: {
                type: "array";
                items: {
                    type: "number";
                };
                description: string;
            };
            seed: {
                type: "number";
                description: string;
            };
        };
        required: string[];
    };
};
