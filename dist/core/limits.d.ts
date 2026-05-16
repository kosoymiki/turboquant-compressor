/**
 * Mobile constraints and Termux compatibility limits.
 */
export declare const MAX_DIMENSIONS = 8192;
export declare const MAX_VECTORS_DEFAULT = 4096;
export declare const MAX_BITS_PER_VALUE = 8;
export declare const MIN_BITS_PER_VALUE = 2;
export declare const TERMUX_DEFAULT_DIMENSIONS = 1024;
export declare const TERMUX_DEFAULT_VECTORS = 128;
export declare const TERMUX_MAX_MEMORY_MB = 256;
export declare function estimateCompressionMemory(dimensions: number, vectorCount: number, bitsPerValue: number): number;
export declare function validateCompressionParams(dimensions: number, vectorCount: number, bitsPerValue: number): void;
export declare function validateSearchParams(dimensions: number, k: number, maxVectors: number): void;
