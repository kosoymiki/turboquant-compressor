/**
 * Uniform symmetric scalar quantization.
 * Fixed codebook over [-1, 1] range for consistent encode/decode.
 * Supports 2, 3, 4, and 8 bits per value.
 */
export declare class UniformSymmetricCodebook {
    private bitsPerValue;
    private numLevels;
    private halfLevels;
    constructor(bitsPerValue?: number);
    /**
     * Encode values using fixed symmetric codebook over [-1, 1].
     * Values are clipped to [-1, 1] range.
     */
    encode(values: Float32Array): Uint8Array;
    /**
     * Decode indices using fixed symmetric codebook over [-1, 1].
     */
    decode(packed: Uint8Array, count?: number): Float32Array;
    /**
     * Quantize with explicit scale (for backward compatibility).
     * Uses fixed codebook, scale parameter is ignored.
     */
    quantize(values: Float32Array): {
        indices: Uint8Array;
        scale: number;
    };
    /**
     * Dequantize with explicit scale (for backward compatibility).
     * Uses fixed codebook, scale parameter is ignored.
     */
    dequantize(indices: Uint8Array, _scale: number, count?: number): Float32Array;
    private getPackedCount;
}
