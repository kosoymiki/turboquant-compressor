import { runBenchmarkMatrix, summarizeResults, formatResultsTable } from './benchmark.js';
import { NONE, FWHT_SIGN } from './rotation.js';

describe('Benchmark', () => {
  test('runBenchmarkMatrix executes without errors', () => {
    const results = runBenchmarkMatrix({
      dimensions: [64],
      vectorCounts: [10],
      bitsPerValue: [4],
      rotationModes: [NONE],
      codebookTypes: ['uniform'],
      useQJL: [false]
    });

    expect(results.length).toBeGreaterThan(0);
  });

  test('runBenchmarkMatrix with multiple configs', () => {
    const results = runBenchmarkMatrix({
      dimensions: [64, 128],
      vectorCounts: [10, 20],
      bitsPerValue: [2, 4],
      rotationModes: [NONE, FWHT_SIGN],
      codebookTypes: ['uniform'],
      useQJL: [false]
    });

    expect(results.length).toBe(16); // 2 dims * 2 counts * 2 bits * 2 modes * 1 codebook * 1 qjl
  });

  test('summarizeResults calculates correct statistics', () => {
    const results = runBenchmarkMatrix({
      dimensions: [64],
      vectorCounts: [10],
      bitsPerValue: [4],
      rotationModes: [NONE],
      codebookTypes: ['uniform'],
      useQJL: [false]
    });

    const summary = summarizeResults(results);

    expect(summary.totalTests).toBe(results.length);
    expect(summary.averageCompressionRatio).toBeGreaterThan(0);
    expect(summary.averageEncodeThroughput).toBeGreaterThan(0);
    expect(summary.averageDecodeThroughput).toBeGreaterThan(0);
  });

  test('summarizeResults handles empty results', () => {
    const summary = summarizeResults([]);

    expect(summary.totalTests).toBe(0);
    expect(summary.averageCompressionRatio).toBe(0);
  });

  test('formatResultsTable produces valid output', () => {
    const results = runBenchmarkMatrix({
      dimensions: [64],
      vectorCounts: [5],
      bitsPerValue: [4],
      rotationModes: [NONE],
      codebookTypes: ['uniform'],
      useQJL: [false]
    });

    const table = formatResultsTable(results);

    expect(table).toContain('|');
    expect(table).toContain('Encode');
    expect(table).toContain('Decode');
    expect(table).toContain('Ratio');
  });

  test('benchmark results have valid metrics', () => {
    const results = runBenchmarkMatrix({
      dimensions: [64],
      vectorCounts: [10],
      bitsPerValue: [4],
      rotationModes: [NONE],
      codebookTypes: ['uniform'],
      useQJL: [false]
    });

    for (const result of results) {
      expect(result.metrics.encodeTimeMs).toBeGreaterThanOrEqual(0);
      expect(result.metrics.decodeTimeMs).toBeGreaterThanOrEqual(0);
      expect(result.metrics.compressionRatio).toBeGreaterThan(0);
      expect(result.metrics.throughputVectorsPerSec).toBeGreaterThan(0);
      expect(result.metrics.mse).toBeGreaterThanOrEqual(0);
      expect(result.metrics.maxError).toBeGreaterThanOrEqual(0);
    }
  });

  test('different codebook types work', () => {
    const results = runBenchmarkMatrix({
      dimensions: [64],
      vectorCounts: [10],
      bitsPerValue: [4],
      rotationModes: [NONE],
      codebookTypes: ['uniform', 'lloydmax'],
      useQJL: [false]
    });

    expect(results.length).toBe(2);
    expect(results[0]!.config.codebookType).toBe('uniform');
    expect(results[1]!.config.codebookType).toBe('lloydmax');
  });

  test('QJL option works', () => {
    const results = runBenchmarkMatrix({
      dimensions: [64],
      vectorCounts: [10],
      bitsPerValue: [4],
      rotationModes: [NONE],
      codebookTypes: ['uniform'],
      useQJL: [true]
    });

    expect(results.length).toBe(1);
    expect(results[0]!.config.useQJL).toBe(true);
  });
});