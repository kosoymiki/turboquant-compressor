/**
 * Deterministic test utilities using seeded PRNG.
 */
import { Mulberry32 } from './prng.js';
export function deterministicFloatArray(length, seed = 123, min = -1, max = 1) {
    const rng = new Mulberry32(seed);
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) {
        out[i] = min + (max - min) * rng.randomFloat();
    }
    return out;
}
export function deterministicIntArray(length, maxExclusive, seed = 123) {
    const rng = new Mulberry32(seed);
    return Array.from({ length }, () => Math.floor(rng.randomFloat() * maxExclusive));
}
export function deterministicNumber(seed = 123, min = 0, max = 1) {
    const rng = new Mulberry32(seed);
    return min + (max - min) * rng.randomFloat();
}
