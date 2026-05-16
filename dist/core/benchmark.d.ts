/**
 * Termux Benchmark Matrix
 * Measures compression performance, memory usage, and throughput
 * across different configurations and data sizes.
 */
import { RotationMode } from './rotation.js';
export interface BenchmarkConfig {
    dimensions: number[];
    vectorCounts: number[];
    bitsPerValue: number[];
    rotationModes: RotationMode[];
    codebookTypes: ('uniform' | 'lloydmax' | 'beta')[];
    useQJL: boolean[];
}
export interface BenchmarkResult {
    config: {
        dimensions: number;
        vectorCount: number;
        bitsPerValue: number;
        rotationMode: RotationMode;
        codebookType: string;
        useQJL: boolean;
    };
    metrics: {
        encodeTimeMs: number;
        decodeTimeMs: number;
        compressionRatio: number;
        throughputVectorsPerSec: number;
        throughputMBPerSec: number;
        memoryUsageBytes: number;
        mse: number;
        maxError: number;
    };
}
export interface BenchmarkSummary {
    totalTests: number;
    averageCompressionRatio: number;
    averageEncodeThroughput: number;
    averageDecodeThroughput: number;
    bestConfig: {
        compressionRatio: number;
        encodeThroughput: number;
        decodeThroughput: number;
    };
    worstConfig: {
        compressionRatio: number;
        encodeThroughput: number;
        decodeThroughput: number;
    };
}
/**
 * Run full benchmark matrix.
 */
export declare function runBenchmarkMatrix(config?: Partial<BenchmarkConfig>): BenchmarkResult[];
/**
 * Generate summary statistics from benchmark results.
 */
export declare function summarizeResults(results: BenchmarkResult[]): BenchmarkSummary;
/**
 * Format results as table.
 */
export declare function formatResultsTable(results: BenchmarkResult[]): string;
/**
 * Export results as JSON.
 */
export declare function exportResultsJSON(results: BenchmarkResult[]): string;
