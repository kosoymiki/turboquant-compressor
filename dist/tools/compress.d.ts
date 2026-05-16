/**
 * TurboQuant compression tool.
 * Implements TurboQuant-inspired vector compression with rotation and quantization.
 */
import type { CompressResult } from './types.js';
export declare function compressVectors(input: {
    vectors: number[][];
    dimensions?: number;
    seed?: number;
    bitsPerValue?: number;
    includeQJL?: boolean;
    rotationSeed?: number;
}): CompressResult;
