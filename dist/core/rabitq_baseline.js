/**
 * RaBitQ baseline — 1-bit sign/hypercube quantization for symmetric comparison.
 * Not a claim that TurboQuant > RaBitQ; this is a fair baseline.
 *
 * Algorithm: normalize → rotate → sign(x) → popcount scoring.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
import { RotationEngine, FWHT_SIGN } from './rotation.js';
/**
 * Encode a vector using RaBitQ: normalize → rotate → sign quantize.
 */
export function rabitqEncode(vector, seed = 0) {
    const d = vector.length;
    // Compute norm
    let norm = 0;
    for (let i = 0; i < d; i++)
        norm += vector[i] * vector[i];
    norm = Math.sqrt(norm);
    // Normalize
    const normalized = new Float32Array(d);
    const invNorm = norm > 1e-10 ? 1 / norm : 0;
    for (let i = 0; i < d; i++)
        normalized[i] = vector[i] * invNorm;
    // Rotate
    const rotation = new RotationEngine(d, seed, FWHT_SIGN);
    const rotated = rotation.rotate(normalized);
    // Sign quantize: pack into bits
    const numBytes = Math.ceil(d / 8);
    const signs = new Uint8Array(numBytes);
    for (let i = 0; i < d; i++) {
        if (rotated[i] >= 0) {
            signs[Math.floor(i / 8)] |= (1 << (i % 8));
        }
    }
    return { signs, norm, dimension: d };
}
/**
 * Popcount-based inner product estimation between query and encoded vector.
 * score ≈ norm * (2 * popcount(xor(q_signs, v_signs)) / d - 1) * correction
 */
export function rabitqScore(query, encoded, seed = 0) {
    const d = encoded.dimension;
    // Encode query the same way
    const qEncoded = rabitqEncode(query, seed);
    // XOR + popcount
    const numBytes = qEncoded.signs.length;
    let matchBits = 0;
    for (let i = 0; i < numBytes; i++) {
        // Count matching sign bits (XNOR popcount)
        const xnor = ~(qEncoded.signs[i] ^ encoded.signs[i]) & 0xFF;
        matchBits += popcount8(xnor);
    }
    // Estimate: cos(angle) ≈ (2 * matchBits / d - 1) * correction_factor
    // correction_factor = sqrt(pi/2) for 1-bit quantization
    const cosEstimate = (2 * matchBits / d - 1) * Math.sqrt(Math.PI / 2);
    return qEncoded.norm * encoded.norm * cosEstimate;
}
/**
 * Batch score: query against multiple encoded vectors.
 */
export function rabitqBatchScore(query, encoded, seed = 0) {
    return encoded.map(e => rabitqScore(query, e, seed));
}
/**
 * Popcount for a single byte.
 */
function popcount8(x) {
    x = x - ((x >> 1) & 0x55);
    x = (x & 0x33) + ((x >> 2) & 0x33);
    return (x + (x >> 4)) & 0x0F;
}
/**
 * Brute-force exact inner product for baseline comparison.
 */
export function exactInnerProduct(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++)
        sum += a[i] * b[i];
    return sum;
}
/**
 * Recall@k: fraction of true top-k that appear in approximate top-k.
 */
export function recallAtK(query, database, encoded, k, seed = 0) {
    // Exact scores
    const exactScores = database.map((v, i) => ({ i, s: exactInnerProduct(query, v) }));
    exactScores.sort((a, b) => b.s - a.s);
    const trueTopK = new Set(exactScores.slice(0, k).map(x => x.i));
    // Approximate scores
    const approxScores = encoded.map((e, i) => ({ i, s: rabitqScore(query, e, seed) }));
    approxScores.sort((a, b) => b.s - a.s);
    const approxTopK = approxScores.slice(0, k).map(x => x.i);
    let hits = 0;
    for (const idx of approxTopK) {
        if (trueTopK.has(idx))
            hits++;
    }
    return hits / k;
}
