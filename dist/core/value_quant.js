/**
 * Value quantization — symmetric group quantization for value vectors.
 * Port of donor turboquant/kv_cache.py quantize_values/dequantize_values.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
/**
 * Quantize value vectors using symmetric group quantization.
 * @param values - flat Float32Array of shape (numTokens * headDim)
 * @param numTokens - number of tokens
 * @param headDim - dimension per token
 * @param bits - 2 or 4
 * @param groupSize - elements per quantization group
 */
export function quantizeValues(values, numTokens, headDim, bits = 2, groupSize = 32) {
    if (headDim % groupSize !== 0) {
        throw new Error(`headDim ${headDim} must be divisible by groupSize ${groupSize}`);
    }
    const nGroups = headDim / groupSize;
    const nLevels = (1 << bits) - 1;
    const valsPerByte = bits === 2 ? 4 : 2; // 3-bit and 4-bit both pack 2 per byte
    const packedDim = Math.ceil(headDim / valsPerByte);
    const scales = new Float32Array(numTokens * nGroups);
    const zeros = new Float32Array(numTokens * nGroups);
    const data = new Uint8Array(numTokens * packedDim);
    for (let t = 0; t < numTokens; t++) {
        const tokenOffset = t * headDim;
        for (let g = 0; g < nGroups; g++) {
            const groupStart = tokenOffset + g * groupSize;
            let vMin = Infinity;
            let vMax = -Infinity;
            for (let i = 0; i < groupSize; i++) {
                const v = values[groupStart + i];
                if (v < vMin)
                    vMin = v;
                if (v > vMax)
                    vMax = v;
            }
            const scale = Math.max((vMax - vMin) / nLevels, 1e-10);
            scales[t * nGroups + g] = scale;
            zeros[t * nGroups + g] = vMin;
            // Quantize and pack
            for (let i = 0; i < groupSize; i++) {
                const v = values[groupStart + i];
                const qi = Math.round(Math.max(0, Math.min(nLevels, (v - vMin) / scale)));
                const flatIdx = g * groupSize + i;
                const byteIdx = t * packedDim + Math.floor(flatIdx / valsPerByte);
                const shift = (flatIdx % valsPerByte) * bits;
                data[byteIdx] |= (qi << shift);
            }
        }
    }
    return { data, scales, zeros, numTokens, headDim, bits, groupSize };
}
/**
 * Dequantize value vectors from bit-packed format.
 */
export function dequantizeValues(vq) {
    const { data, scales, zeros, numTokens, headDim, bits, groupSize } = vq;
    const nGroups = Math.ceil(headDim / groupSize);
    const valsPerByte = bits === 2 ? 4 : 2; // 3-bit and 4-bit both pack 2 per byte
    const mask = (1 << bits) - 1;
    const packedDim = Math.ceil(headDim / valsPerByte);
    const result = new Float32Array(numTokens * headDim);
    for (let t = 0; t < numTokens; t++) {
        for (let g = 0; g < nGroups; g++) {
            const scale = scales[t * nGroups + g];
            const zero = zeros[t * nGroups + g];
            for (let i = 0; i < groupSize; i++) {
                const flatIdx = g * groupSize + i;
                const byteIdx = t * packedDim + Math.floor(flatIdx / valsPerByte);
                const shift = (flatIdx % valsPerByte) * bits;
                const qi = (data[byteIdx] >> shift) & mask;
                result[t * headDim + g * groupSize + i] = qi * scale + zero;
            }
        }
    }
    return result;
}
/**
 * Concatenate two ValueQuantized along the token dimension.
 */
export function concatValueQuantized(a, b) {
    if (a.headDim !== b.headDim || a.bits !== b.bits || a.groupSize !== b.groupSize) {
        throw new Error('Incompatible ValueQuantized for concatenation');
    }
    const numTokens = a.numTokens + b.numTokens;
    const data = new Uint8Array(a.data.length + b.data.length);
    data.set(a.data, 0);
    data.set(b.data, a.data.length);
    const scales = new Float32Array(a.scales.length + b.scales.length);
    scales.set(a.scales, 0);
    scales.set(b.scales, a.scales.length);
    const zeros = new Float32Array(a.zeros.length + b.zeros.length);
    zeros.set(a.zeros, 0);
    zeros.set(b.zeros, a.zeros.length);
    return { data, scales, zeros, numTokens, headDim: a.headDim, bits: a.bits, groupSize: a.groupSize };
}
