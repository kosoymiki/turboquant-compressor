/**
 * TurboQuant vector search tool.
 * Searches for nearest neighbors in compressed vector database.
 */
import type { SearchResult } from './types.js';
export type SearchVectorsInput = {
    compressed_database_b64?: string;
    query_vector?: number[];
    dimensions?: number;
    top_k?: number;
    metric?: 'cosine' | 'dot';
    compressedData?: string;
    queryVector?: number[];
    k?: number;
    useQjl?: boolean;
};
export declare function searchVectors(input: SearchVectorsInput): SearchResult;
export declare function searchVectors(compressedData: string, queryVector: number[], options?: {
    k?: number;
    useQjl?: boolean;
    metric?: 'cosine' | 'dot' | string;
}): SearchResult;
