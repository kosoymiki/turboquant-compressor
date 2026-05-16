/**
 * Beta PDF and Lloyd-Max codebook computation for TurboQuant.
 *
 * After random rotation, each coordinate of a unit-norm vector follows:
 * f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
 * which is a scaled Beta distribution on [-1, 1].
 */
/**
 * Log Gamma function approximation (Lanczos approximation).
 */
export declare function logGamma(x: number): number;
/**
 * PDF of a single coordinate of a uniform random point on S^{d-1}.
 * f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
 */
export declare function sphereCoordinateBetaPdf(x: number, dimension: number): number;
/**
 * Adaptive Simpson integration.
 */
export declare function integrateAdaptiveSimpson(f: (x: number) => number, lo: number, hi: number, options?: {
    eps?: number;
    maxDepth?: number;
}): number;
/**
 * E[X | lo < X < hi] under the Beta PDF on [-1, 1].
 */
export declare function conditionalMeanBetaSphere(lo: number, hi: number, dimension: number): number;
/**
 * Lloyd-Max codebook computation.
 */
export interface TurboQuantCodebook {
    centroids: number[];
    boundaries: number[];
    msePerCoord: number;
    mseTotal: number;
    dimension: number;
    bits: number;
    algorithm: 'lloyd-max-beta';
    source: 'computed' | 'preloaded';
}
export interface LloydMaxOptions {
    maxIter?: number;
    tol?: number;
}
export declare function computeLloydMaxBetaCodebook(dimension: number, bits: 1 | 2 | 3 | 4 | 5 | 6, options?: LloydMaxOptions): TurboQuantCodebook;
