/**
 * Lloyd-Max and Beta codebooks for optimal quantization.
 * Lloyd-Max minimizes quantization error for a given distribution.
 * Beta codebook provides asymmetric quantization for skewed data.
 *
 * This module integrates with beta_sphere.ts for paper-faithful
 * Beta-distributed coordinate PDF and Lloyd-Max computation.
 */
import { TurboQuantCodebook } from './beta_sphere.js';
export interface CodebookEntry {
    index: number;
    value: number;
    reconstruction: number;
    count: number;
}
export interface Codebook {
    bitsPerValue: number;
    entries: CodebookEntry[];
    mse: number;
    iterations: number;
}
export interface BetaParameters {
    alpha: number;
    beta: number;
    offset: number;
    scale: number;
}
/**
 * Lloyd-Max quantization algorithm.
 * Iteratively finds optimal quantization levels for a given distribution.
 */
export declare class LloydMaxCodebook {
    private codebook;
    private bitsPerValue;
    constructor(bitsPerValue?: number);
    /**
     * Train codebook on data using Lloyd-Max algorithm.
     */
    train(data: Float32Array, maxIterations?: number, tolerance?: number): Codebook;
    /**
     * Encode values using trained codebook.
     */
    encode(values: Float32Array): Uint8Array;
    /**
     * Decode indices using trained codebook.
     */
    decode(packed: Uint8Array, count?: number): Float32Array;
    /**
     * Get trained codebook.
     */
    getCodebook(): Codebook | null;
}
/**
 * Experimental Beta distribution-based codebook for asymmetric quantization.
 * Useful for skewed data distributions.
 *
 * WARNING:
 * - Not validated against scipy.stats.beta.ppf.
 * - Not used by public MCP compression/search.
 * - Do not claim optimality.
 * - Research-only until cross-language test vectors are added.
 */
export declare class BetaCodebook {
    private codebook;
    private params;
    private bitsPerValue;
    constructor(bitsPerValue?: number);
    /**
     * Estimate Beta distribution parameters from data.
     */
    private estimateBetaParameters;
    /**
     * Generate quantizer levels from Beta distribution.
     */
    private generateBetaLevels;
    /**
     * Approximate Beta distribution quantile (inverse CDF).
     */
    private betaQuantile;
    /**
     * Beta distribution CDF.
     */
    private betaCDF;
    /**
     * Beta distribution PDF.
     */
    private betaPDF;
    /**
     * Log gamma function using Lanczos approximation.
     */
    private logGamma;
    /**
     * Incomplete beta function using continued fraction.
     */
    private incompleteBeta;
    /**
     * Continued fraction for incomplete beta.
     */
    private betaCF;
    /**
     * Train codebook using Beta distribution.
     */
    train(data: Float32Array): Codebook;
    /**
     * Encode values using trained Beta codebook.
     */
    encode(values: Float32Array): Uint8Array;
    /**
     * Decode indices using trained Beta codebook.
     */
    decode(packed: Uint8Array, count?: number): Float32Array;
    /**
     * Get trained codebook.
     */
    getCodebook(): Codebook | null;
    /**
     * Get Beta parameters.
     */
    getParameters(): BetaParameters | null;
}
/**
 * Quantizer wrapping a validated TurboQuantCodebook.
 * All indexed access via checkedAt — no TS2532.
 */
export declare class TurboQuantCodebookQuantizer {
    private readonly centroids;
    private readonly boundaries;
    private readonly nClusters;
    constructor(codebook: TurboQuantCodebook);
    quantizeIndex(value: number): number;
    quantize(value: number): number;
    dequantize(index: number): number;
}
/**
 * Compute MSE of a codebook against the Beta PDF.
 */
export declare function computeCodebookMse(codebook: TurboQuantCodebook): number;
/**
 * Paper-faithful TurboQuant Beta Lloyd-Max codebook wrapper.
 */
export declare class TurboQuantBetaCodebook {
    private readonly dimension;
    private readonly bits;
    private codebook;
    private quantizer;
    constructor(dimension: number, bits: 1 | 2 | 3 | 4 | 5 | 6);
    compute(options?: {
        maxIter?: number;
        tol?: number;
    }): TurboQuantCodebook;
    getCodebook(): TurboQuantCodebook | null;
    getQuantizer(): TurboQuantCodebookQuantizer;
    quantize(value: number): number;
    quantizeIndex(value: number): number;
    dequantize(index: number): number;
}
