/**
 * Uniform symmetric scalar quantization.
 * Fixed codebook over [-1, 1] range for consistent encode/decode.
 * Supports 2, 3, 4, and 8 bits per value.
 */
import { packValues, unpackValues } from './pack.js';
export class UniformSymmetricCodebook {
    bitsPerValue;
    numLevels;
    halfLevels;
    constructor(bitsPerValue = 4) {
        if (![2, 3, 4, 8].includes(bitsPerValue)) {
            throw new Error(`Unsupported bits per value: ${bitsPerValue}`);
        }
        this.bitsPerValue = bitsPerValue;
        this.numLevels = 2 ** bitsPerValue;
        this.halfLevels = this.numLevels / 2;
    }
    /**
     * Encode values using fixed symmetric codebook over [-1, 1].
     * Values are clipped to [-1, 1] range.
     */
    encode(values) {
        const indices = [];
        for (let i = 0; i < values.length; i++) {
            // Clip to [-1, 1] range
            const clipped = Math.max(-1, Math.min(1, values[i]));
            // Map to index: idx = round((value + 1) * (levels - 1) / 2)
            const idx = Math.round((clipped + 1) * (this.numLevels - 1) / 2);
            indices.push(idx);
        }
        return packValues(indices, this.bitsPerValue);
    }
    /**
     * Decode indices using fixed symmetric codebook over [-1, 1].
     */
    decode(packed, count) {
        const unpackCount = count ?? this.getPackedCount(packed.length);
        const indices = unpackValues(packed, unpackCount, this.bitsPerValue);
        // Map from index: value = (idx / (levels - 1)) * 2 - 1
        const result = new Float32Array(unpackCount);
        for (let i = 0; i < unpackCount; i++) {
            result[i] = (indices[i] / (this.numLevels - 1)) * 2 - 1;
        }
        return result;
    }
    /**
     * Quantize with explicit scale (for backward compatibility).
     * Uses fixed codebook, scale parameter is ignored.
     */
    quantize(values) {
        const indices = [];
        for (let i = 0; i < values.length; i++) {
            const clipped = Math.max(-1, Math.min(1, values[i]));
            const idx = Math.round((clipped + 1) * (this.numLevels - 1) / 2);
            indices.push(idx);
        }
        return { indices: packValues(indices, this.bitsPerValue), scale: 1.0 };
    }
    /**
     * Dequantize with explicit scale (for backward compatibility).
     * Uses fixed codebook, scale parameter is ignored.
     */
    dequantize(indices, _scale, count) {
        const unpackCount = count ?? this.getPackedCount(indices.length);
        const unpacked = unpackValues(indices, unpackCount, this.bitsPerValue);
        const result = new Float32Array(unpackCount);
        for (let i = 0; i < unpackCount; i++) {
            result[i] = (unpacked[i] / (this.numLevels - 1)) * 2 - 1;
        }
        return result;
    }
    getPackedCount(packedLength) {
        const bitsPerByte = 8;
        const totalBits = packedLength * bitsPerByte;
        return Math.floor(totalBits / this.bitsPerValue);
    }
}
