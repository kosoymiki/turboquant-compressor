import {
  analyzeCharacteristics,
  estimateMSE,
  estimateCompressionRatio,
  recommend,
  QuantizationOracle
} from './oracle.js';
import { deterministicFloatArray } from './test-utils.js';

describe('Oracle', () => {
  test('analyzeCharacteristics computes basic stats', () => {
    const data = new Float32Array([1, 2, 3, 4, 5]);
    const stats = analyzeCharacteristics(data);

    expect(stats.min).toBe(1);
    expect(stats.max).toBe(5);
    expect(stats.mean).toBe(3);
    expect(stats.dynamicRange).toBe(4);
    expect(stats.variance).toBeGreaterThan(0);
    expect(stats.stdDev).toBeGreaterThan(0);
  });

  test('analyzeCharacteristics handles uniform data', () => {
    const data = new Float32Array([0.5, 0.5, 0.5, 0.5]);
    const stats = analyzeCharacteristics(data);

    expect(stats.variance).toBe(0);
    expect(stats.stdDev).toBe(0);
    expect(stats.skewness).toBe(0);
    expect(stats.kurtosis).toBe(0);
  });

  test('estimateMSE increases with fewer bits', () => {
    const stats = {
      min: 0,
      max: 1,
      mean: 0.5,
      variance: 0.1,
      stdDev: 0.316,
      skewness: 0,
      kurtosis: 0,
      entropy: 2,
      dynamicRange: 1
    };

    const mse2 = estimateMSE(stats, 2);
    const mse4 = estimateMSE(stats, 4);
    const mse8 = estimateMSE(stats, 8);

    expect(mse2).toBeGreaterThan(mse4);
    expect(mse4).toBeGreaterThan(mse8);
  });

  test('estimateCompressionRatio increases with fewer bits', () => {
    const stats = {
      min: 0,
      max: 1,
      mean: 0.5,
      variance: 0.1,
      stdDev: 0.316,
      skewness: 0,
      kurtosis: 0,
      entropy: 2,
      dynamicRange: 1
    };

    const cr2 = estimateCompressionRatio(stats, 2);
    const cr4 = estimateCompressionRatio(stats, 4);
    const cr8 = estimateCompressionRatio(stats, 8);

    expect(cr2).toBeGreaterThan(cr4);
    expect(cr4).toBeGreaterThan(cr8);
  });

  test('recommend returns valid recommendation', () => {
    const data = deterministicFloatArray(100, 111, -1, 1);

    const rec = recommend(data);

    expect([2, 3, 4, 8]).toContain(rec.bitsPerValue);
    expect(rec.compressionRatio).toBeGreaterThan(1);
    expect(rec.estimatedMSE).toBeGreaterThan(0);
    expect(rec.confidence).toBeGreaterThanOrEqual(0);
    expect(rec.confidence).toBeLessThanOrEqual(1);
  });

  test('recommend respects targetMSE constraint', () => {
    const data = deterministicFloatArray(100, 222, -1, 1);

    const rec = recommend(data, 0.001);
    expect(rec.estimatedMSE).toBeLessThanOrEqual(0.001);
  });

  test('QuantizationOracle fit and predict', () => {
    const data = deterministicFloatArray(50, 333, -1, 1);

    const oracle = new QuantizationOracle();
    oracle.fit(data);

    expect(oracle.optimalBits(0.1)).toBeDefined();
    expect(typeof oracle.compressionRatio(4)).toBe('number');
    expect(typeof oracle.predictError(4)).toBe('number');
  });

  test('QuantizationOracle throws when not fitted', () => {
    const oracle = new QuantizationOracle();

    expect(() => oracle.optimalBits()).toThrow();
    expect(() => oracle.compressionRatio(4)).toThrow();
    expect(() => oracle.predictError(4)).toThrow();
    expect(() => oracle.recommend()).toThrow();
  });

  test('QuantizationOracle recommend returns recommendation', () => {
    const data = deterministicFloatArray(50, 555, -1, 1);

    const oracle = new QuantizationOracle();
    oracle.fit(data);
    const rec = oracle.recommend();

    expect(rec.bitsPerValue).toBeDefined();
    expect(rec.compressionRatio).toBeDefined();
    expect(rec.estimatedMSE).toBeDefined();
    expect(rec.confidence).toBeDefined();
  });
});