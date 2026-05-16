/**
 * Fast Walsh-Hadamard Transform implementation.
 * Mobile-friendly O(n log n) rotation without floating-point operations.
 */
export declare function isPowerOfTwo(n: number): boolean;
export declare function nextPowerOfTwo(n: number): number;
export declare function fwhtInPlace(data: Float32Array): void;
export declare function normalizedFwhtInPlace(data: Float32Array): void;
export declare function padVector(vector: Float32Array, targetLength: number): Float32Array;
export declare function l2Norm(vector: Float32Array): number;
export declare function normalizeVector(vector: Float32Array): Float32Array;
