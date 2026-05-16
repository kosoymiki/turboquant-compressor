/**
 * Python oracle parity module.
 * Provides optimal quantization parameters based on data characteristics.
 * Matches Python implementation's quantization behavior and thresholds.
 */
export interface OracleRecommendation {
    bitsPerValue: number;
    compressionRatio: number;
    estimatedMSE: number;
    confidence: number;
}
export interface DataCharacteristics {
    min: number;
    max: number;
    mean: number;
    variance: number;
    stdDev: number;
    skewness: number;
    kurtosis: number;
    entropy: number;
    dynamicRange: number;
    data?: Float32Array;
}
/**
 * Analyze data characteristics for quantization oracle.
 */
export declare function analyzeCharacteristics(data: Float32Array): DataCharacteristics;
/**
 * Estimate quantization MSE for given bits per value.
 */
export declare function estimateMSE(characteristics: DataCharacteristics, bitsPerValue: number): number;
/**
 * Estimate compression ratio for given bits per value.
 */
export declare function estimateCompressionRatio(characteristics: DataCharacteristics, bitsPerValue: number, originalBits?: number): number;
/**
 * Oracle recommendation based on data characteristics.
 */
export declare function recommend(data: Float32Array, targetMSE?: number, targetCompression?: number): OracleRecommendation;
/**
 * Python-compatible quantization oracle.
 * Matches Python's TurboQuant oracle behavior.
 */
export declare class QuantizationOracle {
    private characteristics;
    /**
     * Fit oracle on training data.
     */
    fit(data: Float32Array): this;
    /**
     * Get optimal bits per value for given tolerance.
     */
    optimalBits(tolerance?: number): number;
    /**
     * Get compression ratio for given bits.
     */
    compressionRatio(bits: number): number;
    /**
     * Predict quantization error.
     */
    predictError(bits: number): number;
    /**
     * Get full recommendation.
     */
    recommend(targetMSE?: number): OracleRecommendation;
}
