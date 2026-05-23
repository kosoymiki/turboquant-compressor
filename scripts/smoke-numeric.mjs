#!/usr/bin/env node
/**
 * Numeric Smoke Test - Validates compression correctness
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const { compressVectors, searchVectors } = await import(join(__dirname, '..', 'dist', 'tools', 'index.js'));
const { decodeBase64, encodeBase64, decodeCompressedDatabase } = await import(join(__dirname, '..', 'dist', 'core', 'format.js'));

let passed = 0, failed = 0;

function test(name, fn) {
    try {
        fn();
        console.log(`  ✓ ${name}`);
        passed++;
    } catch (e) {
        console.log(`  ✗ ${name}: ${e.message}`);
        failed++;
    }
}

function assertEqual(actual, expected, msg) {
    if (Math.abs(actual - expected) > 1e-6) {
        throw new Error(`${msg}: expected ${expected}, got ${actual}`);
    }
}

console.log('Numeric smoke test...\n');

// Test 1: 1536D compression via padding
test('1536D compression works via padding', () => {
    const vec = [];
    for (let i = 0; i < 1536; i++) vec.push((i % 10) / 10);
    const result = compressVectors({ vectors: [vec], bitsPerValue: 4 });
    if (!result.compressed_database_b64) throw new Error('No compressed_database_b64');
    if (result.vector_count !== 1) throw new Error('Wrong vector_count');
    if (result.dimensions !== 1536) throw new Error('Wrong dimensions');
});

// Test 2: Float32 norm roundtrip
test('Float32 norm roundtrip (1.25 -> decode -> 1.25)', () => {
    const result = compressVectors({ vectors: [[1.25, 0, 0, 0]], bitsPerValue: 4 });
    const db = decodeCompressedDatabase(result.compressed_database_b64);
    if (Math.abs(db.norms[0] - 1.25) > 1e-6) {
        throw new Error(`Norm mismatch: ${db.norms[0]} !== 1.25`);
    }
});

// Test 3: Cosine self-match stays high on a canonical exact-match vector
test('Cosine self-match score > 0.9', () => {
    const vec = [1, 0, 0, 0];
    const result = compressVectors({ vectors: [vec], bitsPerValue: 4 });
    const search = searchVectors(result.compressed_database_b64, vec, { k: 1 });
    if (search.results[0].score < 0.9) {
        throw new Error(`Self-match score too low: ${search.results[0].score}`);
    }
});

// Test 4: Dot product self-match ≈ 4 for [2,0,0,0] dot [2,0,0,0]
test('Dot self-match score ≈ 4 for [2,0,0,0] dot [2,0,0,0]', () => {
    const vec = [2, 0, 0, 0];
    const result = compressVectors({ vectors: [vec], bitsPerValue: 4 });
    const search = searchVectors(result.compressed_database_b64, vec, { k: 1, metric: 'dot' });
    // Allow tolerance for quantization error (4-bit quantization has ~0.13 step)
    if (Math.abs(search.results[0].score - 4) > 0.5) {
        throw new Error(`Dot product score: expected ~4, got ${search.results[0].score}`);
    }
});

// Test 5: Query dimension mismatch throws
test('Query dimension mismatch throws', () => {
    const vec4 = [1, 2, 3, 4];
    const vec8 = [1, 2, 3, 4, 5, 6, 7, 8];
    const result = compressVectors({ vectors: [vec4], bitsPerValue: 4 });
    try {
        searchVectors(result.compressed_database_b64, vec8, { k: 1 });
        throw new Error('Should have thrown');
    } catch (e) {
        if (!e.message.includes('does not match')) throw new Error('Wrong error: ' + e.message);
    }
});

// Test 6: Invalid base64 throws
test('Invalid base64 throws', () => {
    try {
        decodeBase64('not-valid-base64!!!');
        throw new Error('Should have thrown');
    } catch (e) {
        if (!e.message.includes('Invalid base64')) throw new Error('Wrong error: ' + e.message);
    }
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
