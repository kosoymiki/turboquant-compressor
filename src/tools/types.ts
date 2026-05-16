/**
 * TurboQuant Compressor API types.
 * Canonical format for MCP integration.
 */

export interface CompressResult {
  compressed_database_b64: string;
  dimensions: number;
  vector_count: number;
  bits_per_value: number;
  include_qjl: boolean;
  algorithm_level: string;
  original_bytes_estimate: number;
  compressed_bytes: number;
  compression_ratio: number;
  format_version: number;
  warnings: string[];
}

export interface SearchResult {
  results: SearchResultItem[];
  metric: string;
  vector_count: number;
  algorithm_level: string;
  warnings: string[];
}

export interface SearchResultItem {
  index: number;
  score: number;
  distance?: number;
}

export interface SearchMetadata {
  dimensions: number;
  vectorsSearched: number;
  k: number;
  useQjl: boolean;
  searchTimeMs: number;
  metric?: string;
}