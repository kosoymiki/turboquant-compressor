/**
 * TurboQuant compression tool.
 * Implements TurboQuant-inspired vector compression with rotation and quantization.
 */
import { RotationEngine } from '../core/rotation.js';
import { UniformSymmetricCodebook } from '../core/quantizer.js';
import { encodeCompressedDatabase, encodeBase64 } from '../core/format.js';
import { validateCompressionParams } from '../core/limits.js';
import { parseCompressInput } from './validation.js';
export function compressVectors(input) {
    const parsed = parseCompressInput(input);
    const vectors = parsed.vectors;
    const dimensions = parsed.dimensions ?? vectors[0].length;
    const seed = parsed.seed ?? parsed.rotationSeed ?? 0;
    const bitsPerValue = parsed.bitsPerValue ?? 4;
    const includeQJL = parsed.includeQJL ?? false;
    validateCompressionParams(dimensions, vectors.length, bitsPerValue);
    const rotation = new RotationEngine(dimensions, seed);
    const codebook = new UniformSymmetricCodebook(bitsPerValue);
    const warnings = [];
    warnings.push('LEVEL_0_TURBOQUANT_INSPIRED_MVP: This is a proof-of-concept implementation');
    warnings.push('Uniform quantizer has fixed step size - may not be optimal for all distributions');
    if (includeQJL) {
        warnings.push('includeQJL was requested, but QJL residual sketch/correction is not implemented in LEVEL_0; no QJL data is stored and include_qjl is reported as false.');
    }
    else {
        warnings.push('QJL residual sketch/correction is not implemented in LEVEL_0.');
    }
    const encodedVectors = [];
    const norms = new Float32Array(vectors.length);
    for (let i = 0; i < vectors.length; i++) {
        const vector = vectors[i];
        // Convert to Float32Array for internal processing
        const floatVector = new Float32Array(vector.length);
        for (let j = 0; j < vector.length; j++) {
            const val = vector[j];
            if (val !== undefined) {
                floatVector[j] = val;
            }
        }
        const rotated = rotation.rotate(floatVector);
        const norm = Math.sqrt(rotated.reduce((sum, v) => sum + v * v, 0));
        norms[i] = norm;
        const normalizedRotated = norm === 0
            ? new Float32Array(rotated.length)
            : (() => {
                const normalized = new Float32Array(rotated.length);
                for (let k = 0; k < rotated.length; k++) {
                    const val = rotated[k];
                    if (val !== undefined) {
                        normalized[k] = val / norm;
                    }
                }
                return normalized;
            })();
        const encoded = codebook.encode(normalizedRotated);
        encodedVectors.push(encoded);
    }
    const binary = encodeCompressedDatabase(encodedVectors, dimensions, bitsPerValue, seed, norms);
    const compressedData = encodeBase64(binary);
    const originalSize = vectors.length * dimensions * 4;
    const compressedSize = binary.length;
    const compressionRatio = originalSize / compressedSize;
    return {
        compressed_database_b64: compressedData,
        dimensions: dimensions,
        vector_count: vectors.length,
        bits_per_value: bitsPerValue,
        include_qjl: false,
        algorithm_level: 'LEVEL_0_TURBOQUANT_INSPIRED_MVP',
        original_bytes_estimate: originalSize,
        compressed_bytes: compressedSize,
        compression_ratio: compressionRatio,
        format_version: 3,
        warnings
    };
}
