/**
 * Deterministic PRNG for rotation and quantization.
 * Uses Mulberry32 algorithm for reproducible results.
 */
export declare class Mulberry32 {
    private state;
    constructor(seed: number);
    /** Returns float in [0, 1) */
    nextFloat(): number;
    /** @deprecated Use nextFloat(). Name is misleading — returns float, not uint32. */
    nextUint32(): number;
    randomFloat(): number;
    randomSign(): number;
    gaussian(): number;
}
export declare function createMulberry32(seed: number): Mulberry32;
