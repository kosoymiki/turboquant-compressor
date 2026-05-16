/**
 * Termux Benchmark Matrix
 * Measures compression performance, memory usage, and throughput
 * across different configurations and data sizes.
 */
import { encodeCompressedDatabase, decodeCompressedDatabase, encodeBase64 } from './format.js';
import { UniformSymmetricCodebook } from './quantizer.js';
import { RotationEngine, FWHT_SIGN, NONE } from './rotation.js';
import { LloydMaxCodebook, BetaCodebook } from './codebook.js';
import { QJLResidualEstimator } from './qjl.js';
const DEFAULT_CONFIG = {
    dimensions: [64, 128, 256, 512],
    vectorCounts: [100, 500, 1000],
    bitsPerValue: [2, 3, 4, 8],
    rotationModes: [FWHT_SIGN, NONE],
    codebookTypes: ['uniform', 'lloydmax', 'beta'],
    useQJL: [false, true]
};
/**
 * Generate test data with specified characteristics.
 */
function generateTestData(dimensions, count, seed = 42) {
    const vectors = [];
    let state = seed;
    for (let i = 0; i < count; i++) {
        const vector = new Float32Array(dimensions);
        for (let j = 0; j < dimensions; j++) {
            state = (state * 1103515245 + 12345) & 0x7fffffff;
            vector[j] = (state / 0x7fffffff) * 2 - 1;
        }
        vectors.push(vector);
    }
    return vectors;
}
/**
 * Calculate MSE between original and reconstructed data.
 */
function calculateMSE(original, reconstructed) {
    let totalError = 0;
    let count = 0;
    for (let i = 0; i < Math.min(original.length, reconstructed.length); i++) {
        const origVec = original[i];
        const reconVec = reconstructed[i];
        if (!origVec || !reconVec)
            continue;
        for (let j = 0; j < Math.min(origVec.length, reconVec.length); j++) {
            const origVal = origVec[j];
            const reconVal = reconVec[j];
            if (origVal === undefined || reconVal === undefined)
                continue;
            if (Number.isNaN(origVal) || Number.isNaN(reconVal))
                continue;
            const diff = origVal - reconVal;
            totalError += diff * diff;
            count++;
        }
    }
    return count > 0 ? totalError / count : 0;
}
/**
 * Calculate max absolute error.
 */
function calculateMaxError(original, reconstructed) {
    let maxError = 0;
    for (let i = 0; i < Math.min(original.length, reconstructed.length); i++) {
        const origVec = original[i];
        const reconVec = reconstructed[i];
        if (!origVec || !reconVec)
            continue;
        for (let j = 0; j < Math.min(origVec.length, reconVec.length); j++) {
            const origVal = origVec[j];
            const reconVal = reconVec[j];
            if (origVal === undefined || reconVal === undefined)
                continue;
            const diff = Math.abs(origVal - reconVal);
            if (diff > maxError) {
                maxError = diff;
            }
        }
    }
    return maxError;
}
/**
 * Get memory usage (approximate).
 */
function getMemoryUsage() {
    if (typeof process !== 'undefined' && process.memoryUsage) {
        return process.memoryUsage().heapUsed;
    }
    return 0;
}
/**
 * Run a single benchmark test.
 */
function runBenchmark(dimensions, vectorCount, bitsPerValue, rotationMode, codebookType, useQJL) {
    // Generate test data
    const originalVectors = generateTestData(dimensions, vectorCount);
    // Calculate norms
    const norms = new Float32Array(vectorCount);
    for (let i = 0; i < vectorCount; i++) {
        let sum = 0;
        for (let j = 0; j < dimensions; j++) {
            sum += originalVectors[i][j] * originalVectors[i][j];
        }
        norms[i] = Math.sqrt(sum);
    }
    // Initialize components
    const rotation = new RotationEngine(dimensions, 42, rotationMode);
    const codebook = codebookType === 'lloydmax'
        ? new LloydMaxCodebook(bitsPerValue)
        : codebookType === 'beta'
            ? new BetaCodebook(bitsPerValue)
            : new UniformSymmetricCodebook(bitsPerValue);
    // Train codebook if needed
    if (codebookType === 'lloydmax' || codebookType === 'beta') {
        const trainingData = new Float32Array(vectorCount * dimensions);
        for (let i = 0; i < vectorCount; i++) {
            trainingData.set(originalVectors[i], i * dimensions);
        }
        codebook.train(trainingData);
    }
    // Initialize QJL if needed
    const qjl = useQJL ? new QJLResidualEstimator() : null;
    // Measure encoding
    const encodeStart = performance.now();
    const quantizedVectors = [];
    const residuals = [];
    for (let i = 0; i < vectorCount; i++) {
        // Rotate
        const rotated = rotation.rotate(originalVectors[i]);
        // Quantize
        const quantized = codebook.encode(rotated);
        quantizedVectors.push(quantized);
        // Calculate residual
        const decoded = codebook.decode(quantized);
        const residual = new Float32Array(dimensions);
        for (let j = 0; j < dimensions; j++) {
            residual[j] = rotated[j] - decoded[j];
        }
        residuals.push(residual);
    }
    // Compress residuals with QJL if enabled
    let qjlSketch;
    if (qjl) {
        const allResiduals = new Float32Array(vectorCount * dimensions);
        for (let i = 0; i < vectorCount; i++) {
            allResiduals.set(residuals[i], i * dimensions);
        }
        const qjlCompressed = qjl.compress(allResiduals);
        qjlSketch = qjlCompressed.sketch;
    }
    // Encode database
    const binary = encodeCompressedDatabase(quantizedVectors, dimensions, bitsPerValue, 42, norms, qjlSketch);
    const encodeEnd = performance.now();
    const encodeTime = encodeEnd - encodeStart;
    // Measure decoding
    const decodeStart = performance.now();
    const base64 = encodeBase64(binary);
    const decoded = decodeCompressedDatabase(base64);
    // Reconstruct vectors
    const reconstructedVectors = [];
    for (let i = 0; i < vectorCount; i++) {
        const decodedVector = codebook.decode(decoded.vectors[i]);
        const rotated = rotation.inverseRotate(decodedVector);
        reconstructedVectors.push(rotated);
    }
    const decodeEnd = performance.now();
    const decodeTime = decodeEnd - decodeStart;
    // Calculate metrics
    const originalSize = vectorCount * dimensions * 4; // Float32Array
    const compressedSize = binary.length;
    const compressionRatio = originalSize / compressedSize;
    const encodeThroughput = (vectorCount / encodeTime) * 1000;
    const decodeThroughput = (vectorCount / decodeTime) * 1000;
    const throughputMB = (originalSize / 1024 / 1024) / (encodeTime / 1000);
    const mse = calculateMSE(originalVectors, reconstructedVectors);
    const maxError = calculateMaxError(originalVectors, reconstructedVectors);
    return {
        config: {
            dimensions,
            vectorCount,
            bitsPerValue,
            rotationMode,
            codebookType,
            useQJL
        },
        metrics: {
            encodeTimeMs: encodeTime,
            decodeTimeMs: decodeTime,
            compressionRatio,
            throughputVectorsPerSec: encodeThroughput,
            throughputMBPerSec: throughputMB,
            memoryUsageBytes: getMemoryUsage(),
            mse,
            maxError
        }
    };
}
/**
 * Run full benchmark matrix.
 */
export function runBenchmarkMatrix(config = {}) {
    const cfg = { ...DEFAULT_CONFIG, ...config };
    const results = [];
    for (const dimensions of cfg.dimensions) {
        for (const vectorCount of cfg.vectorCounts) {
            for (const bitsPerValue of cfg.bitsPerValue) {
                for (const rotationMode of cfg.rotationModes) {
                    for (const codebookType of cfg.codebookTypes) {
                        for (const useQJL of cfg.useQJL) {
                            try {
                                const result = runBenchmark(dimensions, vectorCount, bitsPerValue, rotationMode, codebookType, useQJL);
                                results.push(result);
                            }
                            catch (error) {
                                console.error(`Benchmark failed for config:`, {
                                    dimensions,
                                    vectorCount,
                                    bitsPerValue,
                                    rotationMode,
                                    codebookType,
                                    useQJL
                                });
                                console.error(error);
                            }
                        }
                    }
                }
            }
        }
    }
    return results;
}
/**
 * Generate summary statistics from benchmark results.
 */
export function summarizeResults(results) {
    if (results.length === 0) {
        return {
            totalTests: 0,
            averageCompressionRatio: 0,
            averageEncodeThroughput: 0,
            averageDecodeThroughput: 0,
            bestConfig: { compressionRatio: 0, encodeThroughput: 0, decodeThroughput: 0 },
            worstConfig: { compressionRatio: 0, encodeThroughput: 0, decodeThroughput: 0 }
        };
    }
    const avgCompression = results.reduce((sum, r) => sum + r.metrics.compressionRatio, 0) / results.length;
    const avgEncode = results.reduce((sum, r) => sum + r.metrics.throughputVectorsPerSec, 0) / results.length;
    const avgDecode = results.reduce((sum, r) => sum + r.metrics.throughputVectorsPerSec, 0) / results.length;
    const first = results[0];
    const bestCompression = results.reduce((best, r) => r.metrics.compressionRatio > best.metrics.compressionRatio ? r : best, first);
    const worstCompression = results.reduce((worst, r) => r.metrics.compressionRatio < worst.metrics.compressionRatio ? r : worst, first);
    const bestEncode = results.reduce((best, r) => r.metrics.throughputVectorsPerSec > best.metrics.throughputVectorsPerSec ? r : best, first);
    const worstEncode = results.reduce((worst, r) => r.metrics.throughputVectorsPerSec < worst.metrics.throughputVectorsPerSec ? r : worst, first);
    return {
        totalTests: results.length,
        averageCompressionRatio: avgCompression,
        averageEncodeThroughput: avgEncode,
        averageDecodeThroughput: avgDecode,
        bestConfig: {
            compressionRatio: bestCompression.metrics.compressionRatio,
            encodeThroughput: bestEncode.metrics.throughputVectorsPerSec,
            decodeThroughput: bestEncode.metrics.throughputVectorsPerSec
        },
        worstConfig: {
            compressionRatio: worstCompression.metrics.compressionRatio,
            encodeThroughput: worstEncode.metrics.throughputVectorsPerSec,
            decodeThroughput: worstEncode.metrics.throughputVectorsPerSec
        }
    };
}
/**
 * Format results as table.
 */
export function formatResultsTable(results) {
    const header = '| Config | Encode (ms) | Decode (ms) | Ratio | Throughput (vec/s) | MSE |';
    const separator = '|--------|-------------|-------------|-------|-------------------|-----|';
    const rows = results.map(r => `| ${r.config.dimensions}x${r.config.vectorCount}@${r.config.bitsPerValue}b ` +
        `${r.config.rotationMode} ${r.config.codebookType} qjl:${r.config.useQJL} | ` +
        `${r.metrics.encodeTimeMs.toFixed(1)} | ` +
        `${r.metrics.decodeTimeMs.toFixed(1)} | ` +
        `${r.metrics.compressionRatio.toFixed(2)} | ` +
        `${r.metrics.throughputVectorsPerSec.toFixed(0)} | ` +
        `${r.metrics.mse.toExponential(2)} |`);
    return [header, separator, ...rows].join('\n');
}
/**
 * Export results as JSON.
 */
export function exportResultsJSON(results) {
    return JSON.stringify(results, null, 2);
}
