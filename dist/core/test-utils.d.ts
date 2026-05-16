/**
 * Deterministic test utilities using seeded PRNG.
 */
export declare function deterministicFloatArray(length: number, seed?: number, min?: number, max?: number): Float32Array;
export declare function deterministicIntArray(length: number, maxExclusive: number, seed?: number): number[];
export declare function deterministicNumber(seed?: number, min?: number, max?: number): number;
