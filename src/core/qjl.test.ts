import { QJLResidualEstimator, optimalQJLDimensions, estimateQJLCompressionRatio } from './qjl.js';
import { deterministicFloatArray } from './test-utils.js';

describe('QJLResidualEstimator', () => {
  test('initializes with default config', () => {
    const estimator = new QJLResidualEstimator();
    const config = estimator.getConfig();

    expect(config.targetDimensions).toBe(64);
    expect(config.bitsPerValue).toBe(4);
    expect(config.seed).toBe(42);
    expect(config.useHadamard).toBe(true);
  });

  test('initializes with custom config', () => {
    const estimator = new QJLResidualEstimator({
      targetDimensions: 32,
      bitsPerValue: 2,
      seed: 123,
      useHadamard: false
    });
    const config = estimator.getConfig();

    expect(config.targetDimensions).toBe(32);
    expect(config.bitsPerValue).toBe(2);
    expect(config.seed).toBe(123);
    expect(config.useHadamard).toBe(false);
  });

  test('compress and decompress roundtrip', () => {
    const estimator = new QJLResidualEstimator();
    const residual = deterministicFloatArray(100, 123, -0.05, 0.05);

    const compressed = estimator.compress(residual);
    const reconstructed = estimator.decompress(compressed);

    expect(reconstructed.length).toBe(residual.length);
  });

  test('estimateAndCompress returns stats', () => {
    const estimator = new QJLResidualEstimator();
    const residual = deterministicFloatArray(100, 456, -0.05, 0.05);

    const { compressed, stats } = estimator.estimateAndCompress(residual);

    expect(compressed).toBeDefined();
    expect(compressed.sketch).toBeDefined();
    expect(compressed.dimensions).toBeDefined();
    expect(compressed.originalDimensions).toBe(residual.length);
    expect(compressed.scale).toBeDefined();

    expect(stats.compressionRatio).toBeGreaterThan(1);
    expect(stats.distortion).toBeGreaterThanOrEqual(0);
    expect(stats.sketchSize).toBeGreaterThan(0);
  });

  test('handles zero residual', () => {
    const estimator = new QJLResidualEstimator();
    const residual = new Float32Array(50);
    residual.fill(0);

    const compressed = estimator.compress(residual);
    const reconstructed = estimator.decompress(compressed);

    expect(reconstructed.length).toBe(50);
  });

  test('handles non-power-of-2 dimensions with Hadamard', () => {
    const estimator = new QJLResidualEstimator({ useHadamard: true });
    const residual = new Float32Array(75); // Not power of 2

    const compressed = estimator.compress(residual);
    expect(compressed.originalDimensions).toBe(75);
  });

  test('getIsInitialized returns correct state', () => {
    const estimator = new QJLResidualEstimator();
    expect(estimator.getIsInitialized()).toBe(false);

    const residual = new Float32Array(50);
    estimator.compress(residual);
    expect(estimator.getIsInitialized()).toBe(true);
  });
});

describe('optimalQJLDimensions', () => {
  test('returns valid dimensions', () => {
    const dims = optimalQJLDimensions(1000);
    expect(dims).toBeGreaterThan(0);
  });

  test('increases with more points', () => {
    const dims100 = optimalQJLDimensions(100);
    const dims1000 = optimalQJLDimensions(1000);
    const dims10000 = optimalQJLDimensions(10000);

    expect(dims100).toBeLessThanOrEqual(dims1000);
    expect(dims1000).toBeLessThanOrEqual(dims10000);
  });

  test('decreases with higher distortion tolerance', () => {
    const dims01 = optimalQJLDimensions(1000, 0.1);
    const dims05 = optimalQJLDimensions(1000, 0.5);

    expect(dims01).toBeGreaterThanOrEqual(dims05);
  });
});

describe('estimateQJLCompressionRatio', () => {
  test('returns valid ratio', () => {
    const ratio = estimateQJLCompressionRatio(100, 32, 4);
    expect(ratio).toBeGreaterThan(1);
  });

  test('increases with larger original dimensions', () => {
    const ratio100 = estimateQJLCompressionRatio(100, 32, 4);
    const ratio1000 = estimateQJLCompressionRatio(1000, 32, 4);

    expect(ratio100).toBeLessThan(ratio1000);
  });

  test('decreases with more bits per value', () => {
    const ratio2 = estimateQJLCompressionRatio(100, 32, 2);
    const ratio4 = estimateQJLCompressionRatio(100, 32, 4);
    const ratio8 = estimateQJLCompressionRatio(100, 32, 8);

    expect(ratio2).toBeGreaterThan(ratio4);
    expect(ratio4).toBeGreaterThan(ratio8);
  });
});