/**
 * TurboQuant vector search tool.
 * Searches for nearest neighbors in compressed vector database.
 */

import { parseSearchInput } from './validation.js';
import { RotationEngine } from '../core/rotation.js';
import { UniformSymmetricCodebook } from '../core/quantizer.js';
import { decodeCompressedDatabase } from '../core/format.js';
import { validateSearchParams } from '../core/limits.js';
import type { SearchResult, SearchResultItem } from './types.js';

export type SearchVectorsInput = {
  compressed_database_b64?: string;
  query_vector?: number[];
  dimensions?: number;
  top_k?: number;
  metric?: 'cosine' | 'dot';

  // legacy aliases
  compressedData?: string;
  queryVector?: number[];
  k?: number;
  useQjl?: boolean;
};

function l2Norm(values: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    sum += v * v;
  }
  return Math.sqrt(sum);
}

function pushTopK(top: SearchResultItem[], item: SearchResultItem, k: number): void {
  if (top.length < k) {
    top.push(item);
    top.sort((a, b) => a.score - b.score);
    return;
  }
  if (top[0] !== undefined && item.score > top[0].score) {
    top[0] = item;
    top.sort((a, b) => a.score - b.score);
  }
}

function runSearch(input: SearchVectorsInput): SearchResult {
  const parsed = parseSearchInput(input);

  const compressedData = parsed.compressed_database_b64 ?? parsed.compressedData;
  const query = parsed.query_vector ?? parsed.queryVector;
  const k = parsed.top_k ?? parsed.k ?? 5;
  const metric = parsed.metric ?? 'cosine';

  if (!compressedData || !query) {
    throw new Error('compressed_database_b64 and query_vector are required');
  }

  const db = decodeCompressedDatabase(compressedData);
  validateSearchParams(db.dimensions, k, db.vectors.length);

  if (input.dimensions !== undefined && input.dimensions !== db.dimensions) {
    throw new Error(`Input dimensions ${input.dimensions} does not match database dimensions ${db.dimensions}`);
  }

  if (query.length !== db.dimensions) {
    throw new Error(`Query vector length ${query.length} does not match database dimensions ${db.dimensions}`);
  }

  const codebook = new UniformSymmetricCodebook(db.bitsPerValue);
  const rotation = new RotationEngine(db.dimensions, db.rotationSeed);

  // Convert query to Float32Array for internal processing
  const queryFloat = new Float32Array(query.length);
  for (let i = 0; i < query.length; i++) {
    const val = query[i];
    if (val !== undefined) {
      queryFloat[i] = val;
    }
  }

  const rotatedQuery = rotation.rotate(queryFloat);

  const queryNorm = l2Norm(rotatedQuery);
  const normalizedQuery = queryNorm === 0
    ? new Float32Array(rotatedQuery.length)
    : (() => {
        const normalized = new Float32Array(rotatedQuery.length);
        for (let i = 0; i < rotatedQuery.length; i++) {
          const val = rotatedQuery[i];
          if (val !== undefined) {
            normalized[i] = val / queryNorm;
          }
        }
        return normalized;
      })();

  const top: SearchResultItem[] = [];

  for (let i = 0; i < db.vectors.length; i++) {
    const decoded = codebook.decode(db.vectors[i]!, db.paddedDimensions);

    let score: number;
    if (metric === 'cosine') {
      let dot = 0;
      for (let j = 0; j < normalizedQuery.length; j++) {
        const qVal = normalizedQuery[j];
        const dVal = decoded[j];
        if (qVal !== undefined && dVal !== undefined) {
          dot += qVal * dVal;
        }
      }
      score = Math.max(-1, Math.min(1, dot));
    } else {
      const vectorNorm = db.norms[i]!;
      let dot = 0;
      for (let j = 0; j < rotatedQuery.length; j++) {
        const rVal = rotatedQuery[j];
        const dVal = decoded[j];
        if (rVal !== undefined && dVal !== undefined) {
          dot += rVal * dVal;
        }
      }
      score = vectorNorm * dot;
    }

    const item: SearchResultItem = { index: i, score };
    if (metric === 'cosine') {
      item.distance = 1 - score;
    }
    pushTopK(top, item, k);
  }

  top.sort((a, b) => b.score - a.score);

  const warnings: string[] = [];
  if (metric === 'dot') {
    warnings.push('Dot product score is not normalized - higher values indicate more similarity');
  }
  if (input.useQjl === true) {
    warnings.push('useQjl was requested, but public LEVEL_0 databases store no QJL payload and no QJL correction is applied.');
  }

  return {
    results: top,
    metric,
    vector_count: db.vectors.length,
    algorithm_level: 'LEVEL_0_TURBOQUANT_INSPIRED_MVP',
    warnings,
  };
}

export function searchVectors(input: SearchVectorsInput): SearchResult;
export function searchVectors(
  compressedData: string,
  queryVector: number[],
  options?: { k?: number; useQjl?: boolean; metric?: 'cosine' | 'dot' | string }
): SearchResult;
export function searchVectors(
  inputOrCompressedData: SearchVectorsInput | string,
  queryVector?: number[],
  options: { k?: number; useQjl?: boolean; metric?: 'cosine' | 'dot' | string } = {}
): SearchResult {
  if (typeof inputOrCompressedData === 'string') {
    return runSearch({
      compressedData: inputOrCompressedData,
      queryVector,
      k: options.k,
      useQjl: options.useQjl,
      metric: options.metric === 'dot' ? 'dot' : 'cosine',
    });
  }

  return runSearch(inputOrCompressedData);
}