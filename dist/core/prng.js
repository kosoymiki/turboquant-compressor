/**
 * Deterministic PRNG for rotation and quantization.
 * Uses Mulberry32 algorithm for reproducible results.
 */
export class Mulberry32 {
    state;
    constructor(seed) {
        this.state = seed >>> 0;
    }
    /** Returns float in [0, 1) */
    nextFloat() {
        let t = this.state += 0x6D2B79F5;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    /** @deprecated Use nextFloat(). Name is misleading — returns float, not uint32. */
    nextUint32() {
        return this.nextFloat();
    }
    randomFloat() {
        return this.nextFloat();
    }
    randomSign() {
        return this.randomFloat() < 0.5 ? -1 : 1;
    }
    gaussian() {
        const u = 1 - this.randomFloat();
        const v = this.randomFloat();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
}
export function createMulberry32(seed) {
    return new Mulberry32(seed);
}
