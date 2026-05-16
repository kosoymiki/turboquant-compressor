/**
 * Lloyd-Max and Beta codebooks for optimal quantization.
 * Lloyd-Max minimizes quantization error for a given distribution.
 * Beta codebook provides asymmetric quantization for skewed data.
 *
 * This module integrates with beta_sphere.ts for paper-faithful
 * Beta-distributed coordinate PDF and Lloyd-Max computation.
 */
import { packValues, unpackValues } from './pack.js';
import { sphereCoordinateBetaPdf, integrateAdaptiveSimpson, computeLloydMaxBetaCodebook } from './beta_sphere.js';
/**
 * Lloyd-Max quantization algorithm.
 * Iteratively finds optimal quantization levels for a given distribution.
 */
export class LloydMaxCodebook {
    codebook = null;
    bitsPerValue;
    constructor(bitsPerValue = 4) {
        if (![2, 3, 4, 8].includes(bitsPerValue)) {
            throw new Error(`Unsupported bits per value: ${bitsPerValue}`);
        }
        this.bitsPerValue = bitsPerValue;
    }
    /**
     * Train codebook on data using Lloyd-Max algorithm.
     */
    train(data, maxIterations = 50, tolerance = 1e-6) {
        const numLevels = 2 ** this.bitsPerValue;
        const sorted = Array.from(data).sort((a, b) => a - b);
        const n = sorted.length;
        // Initialize centroids using uniform spacing
        const entries = [];
        const step = (sorted[n - 1] - sorted[0]) / (numLevels - 1 || 1);
        for (let i = 0; i < numLevels; i++) {
            entries.push({
                index: i,
                value: sorted[0] + i * step,
                reconstruction: sorted[0] + i * step,
                count: 0
            });
        }
        let prevMSE = Infinity;
        let iterations = 0;
        for (iterations = 0; iterations < maxIterations; iterations++) {
            // Use explicit sums and counts to avoid accumulator bias
            const sums = new Float64Array(numLevels);
            const counts = new Uint32Array(numLevels);
            // Assign each point to nearest centroid
            for (const val of sorted) {
                let minDist = Infinity;
                let nearest = 0;
                for (let i = 0; i < entries.length; i++) {
                    const dist = Math.abs(val - entries[i].reconstruction);
                    if (dist < minDist) {
                        minDist = dist;
                        nearest = i;
                    }
                }
                sums[nearest] += val;
                counts[nearest]++;
            }
            // Update centroids
            let maxShift = 0;
            for (let i = 0; i < entries.length; i++) {
                const entry = entries[i];
                const old = entry.reconstruction;
                if (counts[i] > 0) {
                    entry.reconstruction = sums[i] / counts[i];
                }
                else {
                    // Empty cell: use midpoint between neighbors
                    entry.reconstruction = entry.value;
                }
                entry.value = entry.reconstruction;
                entry.count = counts[i];
                maxShift = Math.max(maxShift, Math.abs(entry.reconstruction - old));
            }
            if (maxShift < tolerance) {
                break;
            }
        }
        // Calculate final MSE
        let finalMSE = 0;
        for (const val of sorted) {
            let minDist = Infinity;
            for (const entry of entries) {
                const dist = Math.pow(val - entry.reconstruction, 2);
                if (dist < minDist) {
                    minDist = dist;
                }
            }
            finalMSE += minDist;
        }
        finalMSE /= n;
        this.codebook = {
            bitsPerValue: this.bitsPerValue,
            entries,
            mse: finalMSE,
            iterations
        };
        return this.codebook;
    }
    /**
     * Encode values using trained codebook.
     */
    encode(values) {
        if (!this.codebook) {
            throw new Error('Codebook not trained. Call train() first.');
        }
        const indices = [];
        for (const val of values) {
            let minDist = Infinity;
            let nearest = 0;
            for (let i = 0; i < this.codebook.entries.length; i++) {
                const dist = Math.abs(val - this.codebook.entries[i].reconstruction);
                if (dist < minDist) {
                    minDist = dist;
                    nearest = i;
                }
            }
            indices.push(nearest);
        }
        return packValues(indices, this.bitsPerValue);
    }
    /**
     * Decode indices using trained codebook.
     */
    decode(packed, count) {
        if (!this.codebook) {
            throw new Error('Codebook not trained. Call train() first.');
        }
        const unpackCount = count ?? Math.floor((packed.length * 8) / this.bitsPerValue);
        const indices = unpackValues(packed, unpackCount, this.bitsPerValue);
        const result = new Float32Array(unpackCount);
        for (let i = 0; i < unpackCount; i++) {
            const idx = indices[i];
            if (idx >= 0 && idx < this.codebook.entries.length) {
                result[i] = this.codebook.entries[idx].reconstruction;
            }
            else {
                result[i] = 0;
            }
        }
        return result;
    }
    /**
     * Get trained codebook.
     */
    getCodebook() {
        return this.codebook;
    }
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
export class BetaCodebook {
    codebook = null;
    params = null;
    bitsPerValue;
    constructor(bitsPerValue = 4) {
        if (![2, 3, 4, 8].includes(bitsPerValue)) {
            throw new Error(`Unsupported bits per value: ${bitsPerValue}`);
        }
        this.bitsPerValue = bitsPerValue;
    }
    /**
     * Estimate Beta distribution parameters from data.
     */
    estimateBetaParameters(data) {
        const min = Math.min(...Array.from(data));
        const max = Math.max(...Array.from(data));
        const range = max - min || 1;
        // Shift data to [0, 1]
        const shifted = Array.from(data).map(v => (v - min) / range);
        // Method of moments for Beta distribution
        const n = shifted.length;
        const mean = shifted.reduce((a, b) => a + b, 0) / n;
        const variance = shifted.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
        // Handle edge cases
        if (variance === 0 || mean === 0 || mean === 1) {
            return { alpha: 2, beta: 2, offset: min, scale: range };
        }
        // Method of moments estimation
        const temp = (mean * (1 - mean)) / variance - 1;
        const alpha = Math.max(0.1, mean * temp);
        const beta = Math.max(0.1, (1 - mean) * temp);
        return { alpha, beta, offset: min, scale: range };
    }
    /**
     * Generate quantizer levels from Beta distribution.
     */
    generateBetaLevels(alpha, beta, numLevels) {
        const levels = [];
        // Use quantiles of Beta distribution
        for (let i = 0; i < numLevels; i++) {
            // Approximate quantile using inverse CDF
            const p = (i + 0.5) / numLevels;
            levels.push(this.betaQuantile(p, alpha, beta));
        }
        return levels;
    }
    /**
     * Approximate Beta distribution quantile (inverse CDF).
     */
    betaQuantile(p, alpha, beta) {
        if (p <= 0)
            return 0;
        if (p >= 1)
            return 1;
        // Use Newton's method to find x such that BetaCDF(x) = p
        let x = p; // Initial guess
        for (let i = 0; i < 20; i++) {
            const cdf = this.betaCDF(x, alpha, beta);
            const pdf = this.betaPDF(x, alpha, beta);
            if (pdf === 0)
                break;
            const delta = (cdf - p) / pdf;
            x -= delta;
            if (Math.abs(delta) < 1e-10)
                break;
        }
        return Math.max(0, Math.min(1, x));
    }
    /**
     * Beta distribution CDF.
     */
    betaCDF(x, alpha, beta) {
        if (x <= 0)
            return 0;
        if (x >= 1)
            return 1;
        // Regularized incomplete beta function approximation
        const bt = Math.exp(this.logGamma(alpha + beta) -
            this.logGamma(alpha) -
            this.logGamma(beta) +
            alpha * Math.log(x) +
            beta * Math.log(1 - x));
        // Use continued fraction approximation
        return this.incompleteBeta(x, alpha, beta) / bt;
    }
    /**
     * Beta distribution PDF.
     */
    betaPDF(x, alpha, beta) {
        if (x <= 0 || x >= 1)
            return 0;
        return Math.exp((alpha - 1) * Math.log(x) +
            (beta - 1) * Math.log(1 - x) -
            this.logGamma(alpha) -
            this.logGamma(beta) +
            this.logGamma(alpha + beta));
    }
    /**
     * Log gamma function using Lanczos approximation.
     */
    logGamma(x) {
        const g = 7;
        const c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        ];
        if (x < 0.5) {
            return Math.log(Math.PI / Math.sin(Math.PI * x)) - this.logGamma(1 - x);
        }
        x -= 1;
        let a = c[0];
        for (let i = 1; i < g + 2; i++) {
            a += c[i] / (x + i);
        }
        const t = x + g + 0.5;
        return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
    }
    /**
     * Incomplete beta function using continued fraction.
     */
    incompleteBeta(x, alpha, beta) {
        const maxIterations = 200;
        const epsilon = 1e-10;
        if (x === 0)
            return 0;
        if (x === 1)
            return 1;
        // Use continued fraction representation
        const bt = Math.exp(this.logGamma(alpha + beta) -
            this.logGamma(alpha) -
            this.logGamma(beta) +
            alpha * Math.log(x) +
            beta * Math.log(1 - x));
        if (x < (alpha + 1) / (alpha + beta + 2)) {
            return bt * this.betaCF(x, alpha, beta) / alpha;
        }
        else {
            return 1 - bt * this.betaCF(1 - x, beta, alpha) / beta;
        }
    }
    /**
     * Continued fraction for incomplete beta.
     */
    betaCF(x, alpha, beta) {
        const maxIterations = 200;
        const epsilon = 1e-10;
        let qab = alpha + beta;
        let qap = alpha + 1;
        let qam = alpha - 1;
        let c = 1;
        let d = 1 - qab * x / qap;
        if (Math.abs(d) < epsilon)
            d = epsilon;
        d = 1 / d;
        let h = d;
        for (let m = 1; m <= maxIterations; m++) {
            const m2 = 2 * m;
            let aa = m * (beta - m) * x / ((qam + m2) * (alpha + m2));
            d = 1 + aa * d;
            if (Math.abs(d) < epsilon)
                d = epsilon;
            c = 1 + aa / c;
            if (Math.abs(c) < epsilon)
                c = epsilon;
            d = 1 / d;
            h *= d * c;
            aa = -(alpha + m) * (qab + m) * x / ((alpha + m2) * (qap + m2));
            d = 1 + aa * d;
            if (Math.abs(d) < epsilon)
                d = epsilon;
            c = 1 + aa / c;
            if (Math.abs(c) < epsilon)
                c = epsilon;
            d = 1 / d;
            const delta = d * c;
            h *= delta;
            if (Math.abs(delta - 1) < epsilon)
                break;
        }
        return h;
    }
    /**
     * Train codebook using Beta distribution.
     */
    train(data) {
        const numLevels = 2 ** this.bitsPerValue;
        // Estimate Beta parameters
        this.params = this.estimateBetaParameters(data);
        // Generate quantizer levels
        const levels = this.generateBetaLevels(this.params.alpha, this.params.beta, numLevels);
        // Create codebook entries
        const entries = levels.map((level, i) => ({
            index: i,
            value: level,
            reconstruction: level,
            count: 0
        }));
        // Calculate MSE
        let totalMSE = 0;
        for (const val of data) {
            const normalized = (val - this.params.offset) / this.params.scale;
            let minDist = Infinity;
            for (const entry of entries) {
                const dist = Math.pow(normalized - entry.reconstruction, 2);
                if (dist < minDist) {
                    minDist = dist;
                }
            }
            totalMSE += minDist;
        }
        const mse = totalMSE / data.length;
        this.codebook = {
            bitsPerValue: this.bitsPerValue,
            entries,
            mse,
            iterations: 0
        };
        return this.codebook;
    }
    /**
     * Encode values using trained Beta codebook.
     */
    encode(values) {
        if (!this.codebook || !this.params) {
            throw new Error('Codebook not trained. Call train() first.');
        }
        const indices = [];
        for (const val of values) {
            const normalized = (val - this.params.offset) / this.params.scale;
            let minDist = Infinity;
            let nearest = 0;
            for (let i = 0; i < this.codebook.entries.length; i++) {
                const dist = Math.abs(normalized - this.codebook.entries[i].reconstruction);
                if (dist < minDist) {
                    minDist = dist;
                    nearest = i;
                }
            }
            indices.push(nearest);
        }
        return packValues(indices, this.bitsPerValue);
    }
    /**
     * Decode indices using trained Beta codebook.
     */
    decode(packed, count) {
        if (!this.codebook || !this.params) {
            throw new Error('Codebook not trained. Call train() first.');
        }
        const unpackCount = count ?? Math.floor((packed.length * 8) / this.bitsPerValue);
        const indices = unpackValues(packed, unpackCount, this.bitsPerValue);
        const result = new Float32Array(unpackCount);
        for (let i = 0; i < unpackCount; i++) {
            const idx = indices[i];
            if (idx >= 0 && idx < this.codebook.entries.length) {
                const normalized = this.codebook.entries[idx].reconstruction;
                result[i] = normalized * this.params.scale + this.params.offset;
            }
            else {
                result[i] = this.params.offset;
            }
        }
        return result;
    }
    /**
     * Get trained codebook.
     */
    getCodebook() {
        return this.codebook;
    }
    /**
     * Get Beta parameters.
     */
    getParameters() {
        return this.params;
    }
}
// --- TurboQuant Beta Lloyd-Max invariant helpers ---
function invariant(condition, msg) {
    if (!condition)
        throw new Error(`Invariant violation: ${msg}`);
}
function checkedAt(arr, i, label) {
    invariant(i >= 0 && i < arr.length, `${label} index ${i} out of bounds [0, ${arr.length})`);
    const value = arr[i];
    invariant(typeof value === 'number' && Number.isFinite(value), `${label}[${i}] is not a finite number`);
    return value;
}
function assertCodebookShape(cb) {
    const nClusters = 2 ** cb.bits;
    invariant(cb.centroids.length === nClusters, `centroids.length=${cb.centroids.length}, expected ${nClusters}`);
    invariant(cb.boundaries.length === nClusters + 1, `boundaries.length=${cb.boundaries.length}, expected ${nClusters + 1}`);
    for (let i = 1; i < cb.boundaries.length; i++) {
        invariant(checkedAt(cb.boundaries, i, 'boundaries') > checkedAt(cb.boundaries, i - 1, 'boundaries'), `boundaries not strictly increasing at ${i}`);
    }
}
/**
 * Quantizer wrapping a validated TurboQuantCodebook.
 * All indexed access via checkedAt — no TS2532.
 */
export class TurboQuantCodebookQuantizer {
    centroids;
    boundaries;
    nClusters;
    constructor(codebook) {
        assertCodebookShape(codebook);
        this.centroids = codebook.centroids;
        this.boundaries = codebook.boundaries;
        this.nClusters = 2 ** codebook.bits;
    }
    quantizeIndex(value) {
        if (value <= checkedAt(this.boundaries, 0, 'boundaries'))
            return 0;
        if (value >= checkedAt(this.boundaries, this.boundaries.length - 1, 'boundaries'))
            return this.nClusters - 1;
        for (let i = 0; i < this.nClusters; i++) {
            const lo = checkedAt(this.boundaries, i, 'boundaries');
            const hi = checkedAt(this.boundaries, i + 1, 'boundaries');
            if (i === this.nClusters - 1) {
                if (value >= lo && value <= hi)
                    return i;
            }
            else if (value >= lo && value < hi) {
                return i;
            }
        }
        return this.nClusters - 1;
    }
    quantize(value) {
        return this.dequantize(this.quantizeIndex(value));
    }
    dequantize(index) {
        invariant(index >= 0 && index < this.nClusters, `Quantization index out of range: ${index}`);
        return checkedAt(this.centroids, index, 'centroids');
    }
}
/**
 * Compute MSE of a codebook against the Beta PDF.
 */
export function computeCodebookMse(codebook) {
    assertCodebookShape(codebook);
    let mse = 0;
    const nClusters = 2 ** codebook.bits;
    for (let i = 0; i < nClusters; i++) {
        const lo = checkedAt(codebook.boundaries, i, 'boundaries');
        const hi = checkedAt(codebook.boundaries, i + 1, 'boundaries');
        const c = checkedAt(codebook.centroids, i, 'centroids');
        const integrand = (x) => (x - c) ** 2 * sphereCoordinateBetaPdf(x, codebook.dimension);
        mse += integrateAdaptiveSimpson(integrand, lo, hi);
    }
    return mse;
}
/**
 * Paper-faithful TurboQuant Beta Lloyd-Max codebook wrapper.
 */
export class TurboQuantBetaCodebook {
    dimension;
    bits;
    codebook = null;
    quantizer = null;
    constructor(dimension, bits) {
        this.dimension = dimension;
        this.bits = bits;
    }
    compute(options) {
        this.codebook = computeLloydMaxBetaCodebook(this.dimension, this.bits, options);
        this.quantizer = new TurboQuantCodebookQuantizer(this.codebook);
        return this.codebook;
    }
    getCodebook() {
        return this.codebook;
    }
    getQuantizer() {
        if (!this.quantizer)
            throw new Error('Codebook not computed. Call compute() first.');
        return this.quantizer;
    }
    quantize(value) {
        return this.getQuantizer().quantize(value);
    }
    quantizeIndex(value) {
        return this.getQuantizer().quantizeIndex(value);
    }
    dequantize(index) {
        return this.getQuantizer().dequantize(index);
    }
}
