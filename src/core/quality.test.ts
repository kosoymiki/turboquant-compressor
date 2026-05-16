import { runQualityBenchmarks, runQualitySuite, formatQualityTable } from './quality.js';

describe('Quality Benchmarks', () => {
  test('runQualityBenchmarks executes without errors', () => {
    const config = {
      dimensions: 64,
      vectorCount: 50,
      queryCount: 10,
      bitsPerValue: 4,
      codebookType: 'uniform' as const,
      useQJL: false,
      rotationMode: 'none' as const,
      kValues: [1, 5, 10]
    };

    const metrics = runQualityBenchmarks(config);

    expect(metrics.recallAtK.length).toBe(3);
    expect(metrics.meanRecall).toBeGreaterThanOrEqual(0);
    expect(metrics.mse).toBeGreaterThanOrEqual(0);
    expect(metrics.rmse).toBeGreaterThanOrEqual(0);
    expect(metrics.maxError).toBeGreaterThanOrEqual(0);
    expect(metrics.cosineSimilarity).toBeLessThanOrEqual(1);
    expect(metrics.compressionRatio).toBeGreaterThan(0);
    expect(metrics.bitrate).toBeGreaterThan(0);
  });

  test('runQualityBenchmarks with QJL', () => {
    const config = {
      dimensions: 64,
      vectorCount: 50,
      queryCount: 10,
      bitsPerValue: 4,
      codebookType: 'uniform' as const,
      useQJL: true,
      rotationMode: 'none' as const,
      kValues: [1, 5, 10]
    };

    const metrics = runQualityBenchmarks(config);

    expect(metrics.mse).toBeGreaterThanOrEqual(0);
    expect(metrics.cosineSimilarity).toBeLessThanOrEqual(1);
  });

  test('runQualityBenchmarks with hadamard rotation', () => {
    const config = {
      dimensions: 64,
      vectorCount: 50,
      queryCount: 10,
      bitsPerValue: 4,
      codebookType: 'uniform' as const,
      useQJL: false,
      rotationMode: 'hadamard' as const,
      kValues: [1, 5, 10]
    };

    const metrics = runQualityBenchmarks(config);

    expect(metrics.recallAtK.length).toBe(3);
    expect(metrics.compressionRatio).toBeGreaterThan(0);
  });

  test('runQualityBenchmarks with different bits per value', () => {
    const config2bit = {
      dimensions: 64,
      vectorCount: 50,
      queryCount: 10,
      bitsPerValue: 2,
      codebookType: 'uniform' as const,
      useQJL: false,
      rotationMode: 'none' as const,
      kValues: [1, 5, 10]
    };

    const config8bit = {
      dimensions: 64,
      vectorCount: 50,
      queryCount: 10,
      bitsPerValue: 8,
      codebookType: 'uniform' as const,
      useQJL: false,
      rotationMode: 'none' as const,
      kValues: [1, 5, 10]
    };

    const metrics2bit = runQualityBenchmarks(config2bit);
    const metrics8bit = runQualityBenchmarks(config8bit);

    // Higher bitrate should have better quality
    expect(metrics8bit.mse).toBeLessThanOrEqual(metrics2bit.mse);
    expect(metrics8bit.compressionRatio).toBeLessThanOrEqual(metrics2bit.compressionRatio);
  });

  test('runQualitySuite executes without errors', () => {
    const results = runQualitySuite({
      dimensions: 32,
      vectorCount: 20,
      queryCount: 5
    });

    expect(results.length).toBe(5);
  });

  test('runQualitySuite returns valid metrics for each config', () => {
    const results = runQualitySuite({
      dimensions: 32,
      vectorCount: 20,
      queryCount: 5
    });

    for (const result of results) {
      expect(result.config).toBeDefined();
      expect(result.metrics).toBeDefined();
      expect(result.metrics.recallAtK.length).toBeGreaterThan(0);
      expect(result.metrics.mse).toBeGreaterThanOrEqual(0);
      expect(result.metrics.cosineSimilarity).toBeLessThanOrEqual(1);
    }
  });

  test('formatQualityTable produces valid output', () => {
    const results = runQualitySuite({
      dimensions: 32,
      vectorCount: 20,
      queryCount: 5
    });

    const table = formatQualityTable(results);

    expect(table).toContain('|');
    expect(table).toContain('Recall');
    expect(table).toContain('MSE');
    expect(table).toContain('Cosine');
    expect(table).toContain('Compression');
  });

  test('recall values are between 0 and 1', () => {
    const config = {
      dimensions: 64,
      vectorCount: 50,
      queryCount: 10,
      bitsPerValue: 4,
      codebookType: 'uniform' as const,
      useQJL: false,
      rotationMode: 'none' as const,
      kValues: [1, 5, 10]
    };

    const metrics = runQualityBenchmarks(config);

    for (const recall of metrics.recallAtK) {
      expect(recall).toBeGreaterThanOrEqual(0);
      expect(recall).toBeLessThanOrEqual(1);
    }
    expect(metrics.meanRecall).toBeGreaterThanOrEqual(0);
    expect(metrics.meanRecall).toBeLessThanOrEqual(1);
  });

  test('cosine similarity is between -1 and 1', () => {
    const config = {
      dimensions: 64,
      vectorCount: 50,
      queryCount: 10,
      bitsPerValue: 4,
      codebookType: 'uniform' as const,
      useQJL: false,
      rotationMode: 'none' as const,
      kValues: [1, 5, 10]
    };

    const metrics = runQualityBenchmarks(config);

    expect(metrics.cosineSimilarity).toBeGreaterThanOrEqual(-1);
    expect(metrics.cosineSimilarity).toBeLessThanOrEqual(1);
  });

  test('compression ratio is greater than 1', () => {
    const config = {
      dimensions: 64,
      vectorCount: 50,
      queryCount: 10,
      bitsPerValue: 4,
      codebookType: 'uniform' as const,
      useQJL: false,
      rotationMode: 'none' as const,
      kValues: [1, 5, 10]
    };

    const metrics = runQualityBenchmarks(config);

    expect(metrics.compressionRatio).toBeGreaterThan(1);
  });
});