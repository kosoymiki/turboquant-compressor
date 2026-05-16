/**
 * Python oracle parity module.
 * Provides optimal quantization parameters based on data characteristics.
 * Matches Python implementation's quantization behavior and thresholds.
 */
/**
 * Analyze data characteristics for quantization oracle.
 */
export function analyzeCharacteristics(data) {
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let sumSq = 0;
    let sumCube = 0;
    let sumQuad = 0;
    for (let i = 0; i < data.length; i++) {
        const val = data[i];
        min = Math.min(min, val);
        max = Math.max(max, val);
        sum += val;
        sumSq += val * val;
        sumCube += val * val * val;
        sumQuad += val * val * val * val;
    }
    const n = data.length;
    const mean = sum / n;
    const variance = (sumSq / n) - (mean * mean);
    const stdDev = Math.sqrt(Math.max(0, variance));
    // Skewness and kurtosis
    const centered = data.map(v => (v - mean) / (stdDev || 1));
    let skewness = 0;
    let kurtosis = 0;
    for (let i = 0; i < n; i++) {
        skewness += Math.pow(centered[i], 3);
        kurtosis += Math.pow(centered[i], 4);
    }
    skewness /= n;
    kurtosis /= n;
    // Entropy estimation
    const histogram = new Map();
    for (let i = 0; i < data.length; i++) {
        const key = Math.round(data[i] * 1000) / 1000;
        histogram.set(key, (histogram.get(key) || 0) + 1);
    }
    let entropy = 0;
    for (const count of histogram.values()) {
        const p = count / n;
        entropy -= p * Math.log2(p || 1);
    }
    return {
        min,
        max,
        mean,
        variance,
        stdDev,
        skewness,
        kurtosis,
        entropy,
        dynamicRange: max - min,
        data
    };
}
/**
 * Estimate quantization MSE for given bits per value.
 */
export function estimateMSE(characteristics, bitsPerValue) {
    const levels = 2 ** bitsPerValue;
    const quantizationStep = 2 / (levels - 1);
    const theoreticalMSE = (quantizationStep * quantizationStep) / 12;
    // Adjust for data distribution
    const distributionFactor = 1 + Math.abs(characteristics.skewness) * 0.1;
    const varianceFactor = Math.max(1, characteristics.variance * 2);
    return theoreticalMSE * distributionFactor * varianceFactor;
}
/**
 * Estimate compression ratio for given bits per value.
 */
export function estimateCompressionRatio(characteristics, bitsPerValue, originalBits = 32) {
    const efficiency = Math.min(1, characteristics.entropy / 8);
    return (originalBits / bitsPerValue) * (0.9 + efficiency * 0.1);
}
/**
 * Oracle recommendation based on data characteristics.
 */
export function recommend(data, targetMSE, targetCompression) {
    const characteristics = analyzeCharacteristics(data);
    const candidates = [2, 3, 4, 8].map(bits => ({
        bitsPerValue: bits,
        compressionRatio: estimateCompressionRatio(characteristics, bits),
        estimatedMSE: estimateMSE(characteristics, bits),
        confidence: calculateConfidence(characteristics, bits)
    }));
    // Filter by constraints
    let filtered = candidates;
    if (targetMSE !== undefined) {
        filtered = filtered.filter(c => c.estimatedMSE <= targetMSE);
    }
    if (targetCompression !== undefined) {
        filtered = filtered.filter(c => c.compressionRatio >= targetCompression);
    }
    // Select best candidate
    const best = filtered.length > 0
        ? filtered.reduce((a, b) => (a.confidence > b.confidence ? a : b))
        : candidates[0];
    return {
        bitsPerValue: best.bitsPerValue,
        compressionRatio: best.compressionRatio,
        estimatedMSE: best.estimatedMSE,
        confidence: best.confidence
    };
}
/**
 * Calculate confidence score for quantization recommendation.
 */
function calculateConfidence(characteristics, bitsPerValue) {
    let confidence = 1.0;
    // Penalize if data range exceeds [-1, 1]
    if (characteristics.dynamicRange > 2) {
        confidence -= 0.2;
    }
    // Penalize high kurtosis (heavy tails)
    if (characteristics.kurtosis > 3) {
        confidence -= 0.1;
    }
    // Penalize extreme skewness
    if (Math.abs(characteristics.skewness) > 1) {
        confidence -= 0.1;
    }
    // Boost confidence for good entropy match
    const expectedEntropy = Math.log2(2 ** bitsPerValue);
    const entropyMatch = 1 - Math.abs(characteristics.entropy - expectedEntropy) / expectedEntropy;
    confidence += entropyMatch * 0.2;
    return Math.max(0, Math.min(1, confidence));
}
/**
 * Python-compatible quantization oracle.
 * Matches Python's TurboQuant oracle behavior.
 */
export class QuantizationOracle {
    characteristics = null;
    /**
     * Fit oracle on training data.
     */
    fit(data) {
        this.characteristics = analyzeCharacteristics(data);
        return this;
    }
    /**
     * Get optimal bits per value for given tolerance.
     */
    optimalBits(tolerance = 0.01) {
        if (!this.characteristics) {
            throw new Error('Oracle not fitted. Call fit() first.');
        }
        for (const bits of [2, 3, 4, 8]) {
            const mse = estimateMSE(this.characteristics, bits);
            if (mse <= tolerance) {
                return bits;
            }
        }
        return 8;
    }
    /**
     * Get compression ratio for given bits.
     */
    compressionRatio(bits) {
        if (!this.characteristics) {
            throw new Error('Oracle not fitted. Call fit() first.');
        }
        return estimateCompressionRatio(this.characteristics, bits);
    }
    /**
     * Predict quantization error.
     */
    predictError(bits) {
        if (!this.characteristics) {
            throw new Error('Oracle not fitted. Call fit() first.');
        }
        return estimateMSE(this.characteristics, bits);
    }
    /**
     * Get full recommendation.
     */
    recommend(targetMSE) {
        if (!this.characteristics) {
            throw new Error('Oracle not fitted. Call fit() first.');
        }
        return recommend(this.characteristics.data, targetMSE);
    }
}
