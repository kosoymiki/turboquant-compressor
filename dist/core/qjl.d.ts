/**
 * Experimental QJL-like residual sketch utilities.
 *
 * IMPORTANT: This is NOT paper-faithful QJL (Zandieh et al., 2024, arXiv:2406.03482).
 * Paper QJL uses:
 *   - Sign-bit quantization (1-bit) with zero overhead (no scale/zero stored)
 *   - Asymmetric estimator: QJL on keys, standard JL on queries → unbiased dot product
 *   - Structured JL via randomized Hadamard (O(d log d), not O(d²))
 *
 * This module uses dense random projection + multi-bit uniform quantization.
 * It is NOT wired into the public MCP compression/search tools.
 * Research-only (LEVEL_1). Do not claim unbiased-estimator guarantees.
 */
export interface QJLConfig {
    targetDimensions: number;
    bitsPerValue: number;
    seed: number;
    useHadamard: boolean;
}
export interface QJLCompressed {
    sketch: Uint8Array;
    dimensions: number;
    originalDimensions: number;
    scale: number;
}
export interface QJLStats {
    compressionRatio: number;
    distortion: number;
    sketchSize: number;
}
/**
 * Experimental QJL-like residual sketch research utility.
 *
 * Not paper-exact TurboQuant QJL.
 * Not wired into public MCP compression/search.
 * Does not make unbiased-estimator guarantees.
 */
export declare class QJLResidualEstimator {
    private config;
    private projectionMatrix;
    private codebook;
    private isInitialized;
    constructor(config?: Partial<QJLConfig>);
    /**
     * Initialize projection matrix for QJL.
     */
    initialize(originalDimensions: number): void;
    /**
     * Compress residual vector using QJL.
     */
    compress(residual: Float32Array): QJLCompressed;
    /**
     * Reconstruct residual vector from compressed QJL sketch.
     */
    decompress(compressed: QJLCompressed): Float32Array;
    /**
     * Estimate residual and get compression statistics.
     */
    estimateAndCompress(residual: Float32Array): {
        compressed: QJLCompressed;
        stats: QJLStats;
    };
    /**
     * Get configuration.
     */
    getConfig(): QJLConfig;
    /**
     * Check if initialized.
     */
    getIsInitialized(): boolean;
}
/**
 * Optimal QJL dimensions calculator.
 * Based on Johnson-Lindenstrauss lemma for given distortion tolerance.
 */
export declare function optimalQJLDimensions(nPoints: number, distortion?: number, delta?: number): number;
/**
 * Estimate QJL compression ratio.
 */
export declare function estimateQJLCompressionRatio(originalDimensions: number, targetDimensions: number, bitsPerValue: number): number;
