/**
 * Quality Benchmarks: Recall, MSE, and Cosine Similarity
 * Measures compression quality for search accuracy and reconstruction fidelity.
 */
import { UniformSymmetricCodebook } from './quantizer.js';
import { RotationEngine, FWHT_SIGN, NONE } from './rotation.js';
import { LloydMaxCodebook, BetaCodebook } from './codebook.js';
import { QJLResidualEstimator } from './qjl.js';
import { encodeCompressedDatabase, decodeCompressedDatabase, encodeBase64 } from './format.js';
/**
 * Generate test vectors with controlled characteristics.
 */
function generateVectors(dimensions, count, seed = 42, distribution = 'uniform') {
    const vectors = [];
    let state = seed;
    for (let i = 0; i < count; i++) {
        const vector = new Float32Array(dimensions);
        for (let j = 0; j < dimensions; j++) {
            state = (state * 1103515245 + 12345) & 0x7fffffff;
            const rand = state / 0x7fffffff;
            let value;
            switch (distribution) {
                case 'gaussian':
                    // Box-Muller transform
                    const u1 = rand;
                    state = (state * 1103515245 + 12345) & 0x7fffffff;
                    const u2 = state / 0x7fffffff;
                    value = Math.sqrt(-2 * Math.log(u1 || 0.0001)) * Math.cos(2 * Math.PI * u2);
                    value = value * 0.5; // Scale to [-1, 1] approx
                    break;
                case 'skewed':
                    value = Math.pow(rand, 3); // Skewed towards 0
                    break;
                default:
                    value = rand * 2 - 1;
            }
            vector[j] = Math.max(-1, Math.min(1, value));
        }
        vectors.push(vector);
    }
    return vectors;
}
/**
 * Calculate cosine similarity between two vectors.
 */
function cosineSimilarity(a, b) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dotProduct / denom : 0;
}
/**
 * Calculate mean cosine similarity between original and reconstructed vectors.
 */
function calculateAverageCosineSimilarity(originals, reconstructed) {
    const similarities = [];
    for (let i = 0; i < Math.min(originals.length, reconstructed.length); i++) {
        const sim = cosineSimilarity(originals[i], reconstructed[i]);
        if (!Number.isNaN(sim)) {
            similarities.push(sim);
        }
    }
    if (similarities.length === 0) {
        return { mean: 0, std: 0 };
    }
    const mean = similarities.reduce((a, b) => a + b, 0) / similarities.length;
    const variance = similarities.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / similarities.length;
    return { mean, std: Math.sqrt(variance) };
}
/**
 * Find k nearest neighbors using brute force.
 */
function findNearestNeighbors(query, vectors, k) {
    const similarities = vectors.map((vec, idx) => ({
        idx,
        sim: cosineSimilarity(query, vec)
    }));
    similarities.sort((a, b) => b.sim - a.sim);
    return similarities.slice(0, k).map(s => s.idx);
}
/**
 * Compress and decompress vectors using the configured pipeline.
 */
function compressAndDecompress(vectors, config) {
    const { dimensions, bitsPerValue, codebookType, useQJL, rotationMode } = config;
    // Initialize components
    const rotation = new RotationEngine(dimensions, 42, rotationMode === 'hadamard' ? FWHT_SIGN : NONE);
    const codebook = codebookType === 'lloydmax'
        ? new LloydMaxCodebook(bitsPerValue)
        : codebookType === 'beta'
            ? new BetaCodebook(bitsPerValue)
            : new UniformSymmetricCodebook(bitsPerValue);
    // Train codebook if needed
    if (codebookType === 'lloydmax' || codebookType === 'beta') {
        const trainingData = new Float32Array(vectors.length * dimensions);
        for (let i = 0; i < vectors.length; i++) {
            trainingData.set(vectors[i], i * dimensions);
        }
        codebook.train(trainingData);
    }
    // Initialize QJL if needed
    const qjl = useQJL ? new QJLResidualEstimator() : null;
    // Calculate norms
    const norms = new Float32Array(vectors.length);
    for (let i = 0; i < vectors.length; i++) {
        let sum = 0;
        for (let j = 0; j < dimensions; j++) {
            sum += vectors[i][j] * vectors[i][j];
        }
        norms[i] = Math.sqrt(sum);
    }
    // Quantize vectors
    const quantizedVectors = [];
    const residuals = [];
    for (let i = 0; i < vectors.length; i++) {
        const rotated = rotation.rotate(vectors[i]);
        const quantized = codebook.encode(rotated);
        quantizedVectors.push(quantized);
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
        const allResiduals = new Float32Array(vectors.length * dimensions);
        for (let i = 0; i < vectors.length; i++) {
            allResiduals.set(residuals[i], i * dimensions);
        }
        const qjlCompressed = qjl.compress(allResiduals);
        qjlSketch = qjlCompressed.sketch;
    }
    // Encode database
    const binary = encodeCompressedDatabase(quantizedVectors, dimensions, bitsPerValue, 42, norms, qjlSketch);
    // Decode
    const base64 = encodeBase64(binary);
    const decoded = decodeCompressedDatabase(base64);
    // Reconstruct vectors
    const reconstructed = [];
    for (let i = 0; i < vectors.length; i++) {
        const decodedVector = codebook.decode(decoded.vectors[i]);
        const rotated = rotation.inverseRotate(decodedVector);
        reconstructed.push(rotated);
    }
    return { compressed: binary, decompressed: reconstructed };
}
/**
 * Run quality benchmarks.
 */
export function runQualityBenchmarks(config) {
    const { dimensions, vectorCount, queryCount, kValues } = config;
    // Generate database and query vectors
    const database = generateVectors(dimensions, vectorCount, 42, 'uniform');
    const queries = generateVectors(dimensions, queryCount, 123, 'uniform');
    // Compress and decompress database
    const { decompressed } = compressAndDecompress(database, config);
    // Calculate MSE and max error
    let totalMSE = 0;
    let maxError = 0;
    let totalCount = 0;
    for (let i = 0; i < database.length; i++) {
        const origVec = database[i];
        const reconVec = decompressed[i];
        if (!origVec || !reconVec)
            continue;
        for (let j = 0; j < dimensions; j++) {
            const orig = origVec[j];
            const recon = reconVec[j];
            if (orig === undefined || recon === undefined)
                continue;
            if (Number.isNaN(orig) || Number.isNaN(recon))
                continue;
            const diff = orig - recon;
            totalMSE += diff * diff;
            if (Math.abs(diff) > maxError) {
                maxError = Math.abs(diff);
            }
            totalCount++;
        }
    }
    const mse = totalCount > 0 ? totalMSE / totalCount : 0;
    const rmse = Math.sqrt(mse);
    // Calculate cosine similarity
    const { mean: cosineSim, std: cosineSimStd } = calculateAverageCosineSimilarity(database, decompressed);
    // Calculate recall@k
    const searchResults = [];
    for (const query of queries) {
        const groundTruth = findNearestNeighbors(query, database, Math.max(...kValues));
        const retrieved = findNearestNeighbors(query, decompressed, Math.max(...kValues));
        const hits = groundTruth.filter(gt => retrieved.includes(gt)).length;
        searchResults.push({
            queryIndex: 0,
            groundTruth,
            retrieved,
            hits,
            recall: hits / groundTruth.length
        });
    }
    // Calculate recall at each k
    const recallAtK = [];
    for (const k of kValues) {
        let totalHits = 0;
        for (const result of searchResults) {
            const gtK = result.groundTruth.slice(0, k);
            const retK = result.retrieved.slice(0, k);
            const hits = gtK.filter(gt => retK.includes(gt)).length;
            totalHits += hits;
        }
        const expectedHits = k * queryCount;
        recallAtK.push(expectedHits > 0 ? totalHits / expectedHits : 0);
    }
    const meanRecall = recallAtK.reduce((a, b) => a + b, 0) / recallAtK.length;
    // Calculate compression ratio and bitrate
    const originalSize = vectorCount * dimensions * 32; // bits
    const { compressed } = compressAndDecompress(database, config);
    const compressedSize = compressed.length * 8; // bits
    const compressionRatio = originalSize / compressedSize;
    const bitrate = compressedSize / vectorCount;
    return {
        recallAtK,
        meanRecall,
        mse,
        rmse,
        maxError,
        cosineSimilarity: cosineSim,
        cosineSimilarityStd: cosineSimStd,
        compressionRatio,
        bitrate
    };
}
/**
 * Run comprehensive quality benchmark suite.
 */
export function runQualitySuite(baseConfig = {}) {
    const results = [];
    const configs = [
        {
            dimensions: baseConfig.dimensions ?? 128,
            vectorCount: baseConfig.vectorCount ?? 1000,
            queryCount: baseConfig.queryCount ?? 50,
            bitsPerValue: 4,
            codebookType: 'uniform',
            useQJL: false,
            rotationMode: 'none',
            kValues: [1, 5, 10, 20]
        },
        {
            dimensions: baseConfig.dimensions ?? 128,
            vectorCount: baseConfig.vectorCount ?? 1000,
            queryCount: baseConfig.queryCount ?? 50,
            bitsPerValue: 4,
            codebookType: 'uniform',
            useQJL: true,
            rotationMode: 'none',
            kValues: [1, 5, 10, 20]
        },
        {
            dimensions: baseConfig.dimensions ?? 128,
            vectorCount: baseConfig.vectorCount ?? 1000,
            queryCount: baseConfig.queryCount ?? 50,
            bitsPerValue: 4,
            codebookType: 'uniform',
            useQJL: false,
            rotationMode: 'hadamard',
            kValues: [1, 5, 10, 20]
        },
        {
            dimensions: baseConfig.dimensions ?? 128,
            vectorCount: baseConfig.vectorCount ?? 1000,
            queryCount: baseConfig.queryCount ?? 50,
            bitsPerValue: 2,
            codebookType: 'uniform',
            useQJL: false,
            rotationMode: 'none',
            kValues: [1, 5, 10, 20]
        },
        {
            dimensions: baseConfig.dimensions ?? 128,
            vectorCount: baseConfig.vectorCount ?? 1000,
            queryCount: baseConfig.queryCount ?? 50,
            bitsPerValue: 8,
            codebookType: 'uniform',
            useQJL: false,
            rotationMode: 'none',
            kValues: [1, 5, 10, 20]
        }
    ];
    for (const config of configs) {
        const metrics = runQualityBenchmarks(config);
        results.push({ config, metrics });
    }
    return results;
}
/**
 * Format quality results as table.
 */
export function formatQualityTable(results) {
    const header = '| Config | Recall@1 | Recall@10 | MSE | Cosine Sim | Compression |';
    const separator = '|--------|----------|----------|-----|------------|-------------|';
    const rows = results.map(r => `| ${r.config.bitsPerValue}b ${r.config.codebookType} qjl:${r.config.useQJL} hadamard:${r.config.rotationMode === 'hadamard'} | ` +
        `${r.metrics.recallAtK[0]?.toFixed(3) ?? 'N/A'} | ` +
        `${r.metrics.recallAtK[2]?.toFixed(3) ?? 'N/A'} | ` +
        `${r.metrics.mse.toExponential(2)} | ` +
        `${r.metrics.cosineSimilarity.toFixed(3)} | ` +
        `${r.metrics.compressionRatio.toFixed(2)}x |`);
    return [header, separator, ...rows].join('\n');
}
/**
 * Export quality results as JSON.
 */
export function exportQualityJSON(results) {
    return JSON.stringify(results, null, 2);
}
