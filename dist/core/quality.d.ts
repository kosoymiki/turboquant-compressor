/**
 * Quality Benchmarks: Recall, MSE, and Cosine Similarity
 * Measures compression quality for search accuracy and reconstruction fidelity.
 */
export interface QualityMetrics {
    recallAtK: number[];
    meanRecall: number;
    mse: number;
    rmse: number;
    maxError: number;
    cosineSimilarity: number;
    cosineSimilarityStd: number;
    compressionRatio: number;
    bitrate: number;
}
export interface SearchResult {
    queryIndex: number;
    groundTruth: number[];
    retrieved: number[];
    hits: number;
    recall: number;
}
export interface QualityConfig {
    dimensions: number;
    vectorCount: number;
    queryCount: number;
    bitsPerValue: number;
    codebookType: 'uniform' | 'lloydmax' | 'beta';
    useQJL: boolean;
    rotationMode: 'none' | 'hadamard';
    kValues: number[];
}
/**
 * Run quality benchmarks.
 */
export declare function runQualityBenchmarks(config: QualityConfig): QualityMetrics;
/**
 * Run comprehensive quality benchmark suite.
 */
export declare function runQualitySuite(baseConfig?: Partial<QualityConfig>): {
    config: QualityConfig;
    metrics: QualityMetrics;
}[];
/**
 * Format quality results as table.
 */
export declare function formatQualityTable(results: {
    config: QualityConfig;
    metrics: QualityMetrics;
}[]): string;
/**
 * Export quality results as JSON.
 */
export declare function exportQualityJSON(results: {
    config: QualityConfig;
    metrics: QualityMetrics;
}[]): string;
