/**
 * MCP tool: turboquant_quantize — expose TurboQuant codebook + quantization.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
import { TurboQuantBetaCodebook, computeCodebookMse } from '../core/codebook.js';
import { computeLloydMaxBetaCodebook } from '../core/beta_sphere.js';
import { RotationEngine, FWHT_SIGN } from '../core/rotation.js';
export function turboquantQuantize(input) {
    const bits = (input.bits ?? 4);
    const dimension = input.dimension;
    if (dimension < 3)
        throw new Error('Dimension must be >= 3');
    if (bits < 1 || bits > 6)
        throw new Error('Bits must be 1-6');
    const result = { action: input.action, dimension, bits };
    if (input.action === 'compute_codebook') {
        const cb = computeLloydMaxBetaCodebook(dimension, bits);
        result.codebook = {
            centroids: cb.centroids,
            boundaries: cb.boundaries,
            msePerCoord: cb.msePerCoord,
            mseTotal: cb.mseTotal,
        };
    }
    else if (input.action === 'quantize_vector') {
        if (!input.vector || input.vector.length === 0) {
            throw new Error('vector is required for quantize_vector action');
        }
        const vec = new Float32Array(input.vector);
        const d = vec.length;
        // Compute norm and normalize
        let norm = 0;
        for (let i = 0; i < d; i++)
            norm += vec[i] * vec[i];
        norm = Math.sqrt(norm);
        const normalized = new Float32Array(d);
        const invNorm = norm > 1e-10 ? 1 / norm : 0;
        for (let i = 0; i < d; i++)
            normalized[i] = vec[i] * invNorm;
        // Rotate
        const rotation = new RotationEngine(d, input.seed ?? 42, FWHT_SIGN);
        const rotated = rotation.rotate(normalized);
        // Quantize
        const beta = new TurboQuantBetaCodebook(d, bits);
        beta.compute();
        const quantizer = beta.getQuantizer();
        const indices = [];
        const quantizedRotated = new Float32Array(rotated.length);
        for (let i = 0; i < d; i++) {
            const idx = quantizer.quantizeIndex(rotated[i]);
            indices.push(idx);
            quantizedRotated[i] = quantizer.dequantize(idx);
        }
        // Inverse rotate and rescale
        const unrotated = rotation.inverseRotate(quantizedRotated);
        const reconstructed = [];
        for (let i = 0; i < d; i++) {
            reconstructed.push(unrotated[i] * norm);
        }
        // Compute MSE and cosine similarity
        let mse = 0;
        let dotProd = 0;
        let normA = 0;
        let normB = 0;
        for (let i = 0; i < d; i++) {
            mse += (input.vector[i] - reconstructed[i]) ** 2;
            dotProd += input.vector[i] * reconstructed[i];
            normA += input.vector[i] * input.vector[i];
            normB += reconstructed[i] * reconstructed[i];
        }
        mse /= d;
        const cosineSimilarity = dotProd / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
        result.quantized = { indices, norm, reconstructed, mse, cosineSimilarity };
    }
    else if (input.action === 'benchmark_mse') {
        const cb = computeLloydMaxBetaCodebook(dimension, bits);
        const mse = computeCodebookMse(cb);
        result.benchmark = {
            msePerCoord: mse,
            mseTotal: mse * dimension,
            compressionRatio: 32 / bits,
            bitsPerDimension: bits,
        };
    }
    return result;
}
/** MCP tool schema for turboquant_quantize */
export const turboquantQuantizeSchema = {
    name: 'turboquant_quantize',
    description: 'TurboQuant codebook computation, vector quantization, and MSE benchmarking. Uses paper-faithful Beta PDF + Lloyd-Max algorithm.',
    inputSchema: {
        type: 'object',
        properties: {
            action: {
                type: 'string',
                enum: ['compute_codebook', 'quantize_vector', 'benchmark_mse'],
                description: 'Action to perform',
            },
            dimension: {
                type: 'number',
                description: 'Vector dimension (>= 3)',
            },
            bits: {
                type: 'number',
                description: 'Bits per coordinate (1-6, default 4)',
            },
            vector: {
                type: 'array',
                items: { type: 'number' },
                description: 'Vector to quantize (for quantize_vector action)',
            },
            seed: {
                type: 'number',
                description: 'Rotation seed (default 42)',
            },
        },
        required: ['action', 'dimension'],
    },
};
