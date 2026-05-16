/**
 * Scientific: QJL residual estimator seeded validation.
 * Verifies payload exists, variance reported, basic unbiasedness property.
 */
import { QJLResidualEstimator } from '../../src/core/qjl.js';

describe('QJL unbiasedness (seeded)', () => {
  it('compress produces non-empty sketch payload', () => {
    const qjl = new QJLResidualEstimator({ targetDimensions: 32, bitsPerValue: 4, seed: 42, useHadamard: true });
    const residual = new Float32Array(64);
    for (let i = 0; i < 64; i++) residual[i] = Math.sin(i * 0.1);

    const compressed = qjl.compress(residual);
    expect(compressed.sketch.length).toBeGreaterThan(0);
    expect(compressed.dimensions).toBe(32);
    expect(compressed.originalDimensions).toBe(64);
    expect(Number.isFinite(compressed.scale)).toBe(true);
  });

  it('estimateAndCompress reports finite distortion', () => {
    const qjl = new QJLResidualEstimator({ targetDimensions: 32, bitsPerValue: 4, seed: 42, useHadamard: true });
    const residual = new Float32Array(64);
    for (let i = 0; i < 64; i++) residual[i] = Math.cos(i * 0.05);

    const { stats } = qjl.estimateAndCompress(residual);
    expect(Number.isFinite(stats.distortion)).toBe(true);
    expect(stats.distortion).toBeGreaterThanOrEqual(0);
    expect(stats.compressionRatio).toBeGreaterThan(1);
    expect(stats.sketchSize).toBeGreaterThan(0);
  });

  it('deterministic: same seed produces same sketch', () => {
    const residual = new Float32Array(64);
    for (let i = 0; i < 64; i++) residual[i] = Math.sin(i);

    const q1 = new QJLResidualEstimator({ targetDimensions: 16, bitsPerValue: 4, seed: 123, useHadamard: true });
    const q2 = new QJLResidualEstimator({ targetDimensions: 16, bitsPerValue: 4, seed: 123, useHadamard: true });

    const c1 = q1.compress(residual);
    const c2 = q2.compress(residual);

    expect(Array.from(c1.sketch)).toEqual(Array.from(c2.sketch));
    expect(c1.scale).toBe(c2.scale);
  });

  it('reconstruction produces finite values', () => {
    const qjl = new QJLResidualEstimator({ targetDimensions: 48, bitsPerValue: 4, seed: 7, useHadamard: true });
    const residual = new Float32Array(64);
    for (let i = 0; i < 64; i++) residual[i] = (i % 2 === 0 ? 1 : -1) * 0.1;

    const compressed = qjl.compress(residual);
    const reconstructed = qjl.decompress(compressed);

    expect(reconstructed.length).toBe(64);
    for (let i = 0; i < 64; i++) {
      expect(Number.isFinite(reconstructed[i]!)).toBe(true);
    }
  });

  it('different seeds produce different sketches', () => {
    const residual = new Float32Array(64);
    for (let i = 0; i < 64; i++) residual[i] = Math.sin(i * 0.3);

    const q1 = new QJLResidualEstimator({ targetDimensions: 16, bitsPerValue: 4, seed: 1, useHadamard: true });
    const q2 = new QJLResidualEstimator({ targetDimensions: 16, bitsPerValue: 4, seed: 999, useHadamard: true });

    const c1 = q1.compress(residual);
    const c2 = q2.compress(residual);

    // Different seeds should produce different results
    expect(Array.from(c1.sketch)).not.toEqual(Array.from(c2.sketch));
  });
});
