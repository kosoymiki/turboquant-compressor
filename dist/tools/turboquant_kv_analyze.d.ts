/**
 * MCP tool: turboquant_kv_analyze — KV cache compression analysis.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
export interface KVAnalyzeInput {
    /** Action: 'estimate_savings' | 'recommend_config' */
    action: 'estimate_savings' | 'recommend_config';
    /** Model head dimension */
    headDim: number;
    /** Number of KV heads */
    numKvHeads?: number;
    /** Number of layers */
    numLayers?: number;
    /** Sequence length */
    seqLen?: number;
    /** Key bits (1-6) */
    keyBits?: number;
    /** Value bits (2 or 4) */
    valueBits?: number;
    /** Buffer size (recent unquantized tokens) */
    bufferSize?: number;
    /** Target memory budget in MB */
    memoryBudgetMb?: number;
}
export interface KVAnalyzeOutput {
    action: string;
    estimate?: {
        baselineMemoryMb: number;
        compressedMemoryMb: number;
        savingsPercent: number;
        compressionRatio: number;
        keyBitsUsed: number;
        valueBitsUsed: number;
        bufferSize: number;
    };
    recommendation?: {
        keyBits: number;
        valueBits: number;
        bufferSize: number;
        estimatedMsePerCoord: number;
        maxSeqLen: number;
        memoryMb: number;
    };
}
export declare function turboquantKvAnalyze(input: KVAnalyzeInput): KVAnalyzeOutput;
export declare const turboquantKvAnalyzeSchema: {
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
            headDim: {
                type: "number";
                description: string;
            };
            numKvHeads: {
                type: "number";
                description: string;
            };
            numLayers: {
                type: "number";
                description: string;
            };
            seqLen: {
                type: "number";
                description: string;
            };
            keyBits: {
                type: "number";
                description: string;
            };
            valueBits: {
                type: "number";
                description: string;
            };
            bufferSize: {
                type: "number";
                description: string;
            };
            memoryBudgetMb: {
                type: "number";
                description: string;
            };
        };
        required: string[];
    };
};
