#!/usr/bin/env node

import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { extname, join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { compressVectors, searchVectors } from '../dist/index.js';
import {
  buildHashedTfidfVectorizer,
  createTokenHashVectorizer,
} from '../dist/context/vectorizer.js';
import { computeLloydMaxBetaCodebook } from '../dist/core/beta_sphere.js';
import { RotationEngine } from '../dist/core/rotation.js';

const DIM = Number(process.env.TQ_DIM ?? 384);
const CHUNK_BYTES = Number(process.env.TQ_CHUNK_BYTES ?? 4096);
const QUERY_LIMIT = Number(process.env.TQ_QUERY_LIMIT ?? 50);
const BITS_PER_VALUE = Number(process.env.TQ_BITS_PER_VALUE ?? 4);
const ROTATION_SEED = Number(process.env.TQ_ROTATION_SEED ?? 42);
const SHIPPED_CODEBOOK = BITS_PER_VALUE === 8 ? 'uniform' : 'turboquant_beta';

function walkFiles(root, predicate) {
  const out = [];

  function visit(path) {
    const st = statSync(path);
    if (st.isDirectory()) {
      for (const entry of readdirSync(path)) {
        visit(join(path, entry));
      }
      return;
    }

    if (st.isFile() && predicate(path)) {
      out.push(path.replaceAll('\\', '/'));
    }
  }

  visit(root);
  return out;
}

function listFiles() {
  const candidates = [];

  if (existsSync('README.md')) candidates.push('README.md');

  for (const root of ['docs', 'src']) {
    if (!existsSync(root)) continue;
    candidates.push(...walkFiles(root, (file) => {
      const ext = extname(file).toLowerCase();
      return (ext === '.md' || ext === '.ts') && !file.endsWith('.test.ts');
    }));
  }

  return candidates.sort();
}

function approximateTokens(text) {
  return Math.ceil(Buffer.byteLength(text, 'utf8') / 4);
}

function chunkText(text, maxBytes) {
  const chunks = [];
  let current = '';

  for (const line of text.split(/\r?\n/)) {
    const next = current + line + '\n';
    if (Buffer.byteLength(next, 'utf8') > maxBytes && current.length > 0) {
      chunks.push(current);
      current = '';
    }
    current += line + '\n';
  }

  if (current.length > 0) chunks.push(current);
  return chunks;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function percentile(values, p) {
  if (values.length === 0) return 0;
  const idx = Math.min(values.length - 1, Math.floor(values.length * p));
  return values[idx];
}

function fileStemQuery(file, text) {
  const pathTerms = file
    .replace(/\.[^.]+$/, '')
    .replaceAll('/', ' ')
    .replaceAll('-', ' ')
    .replaceAll('_', ' ')
    .trim();

  const heading = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => line.startsWith('#') || /^(export )?(class|function|interface|type|const)\s+/u.test(line));

  return `${pathTerms} ${heading ?? ''}`.trim();
}

function buildQuantizationProfile(queryVectors, exactScorer, approxScorer) {
  let rankingAgreementAt1 = 0;
  let rankingOverlapAt5 = 0;
  const exactTopKErrors = [];
  const unionTopKErrors = [];

  for (const query of queryVectors) {
    const exactScores = exactScorer(query)
      .map((score, index) => ({ index, score }))
      .sort((a, b) => b.score - a.score);

    const exactTopK = exactScores.slice(0, 5);
    const approxTopK = approxScorer(query).slice(0, 5);

    if (approxTopK[0]?.index === exactTopK[0]?.index) {
      rankingAgreementAt1++;
    }

    const exactTopKSet = new Set(exactTopK.map((item) => item.index));
    let overlap = 0;
    for (const item of approxTopK) {
      if (exactTopKSet.has(item.index)) {
        overlap++;
      }
    }
    rankingOverlapAt5 += overlap / 5;

    const approxByIndex = new Map(approxTopK.map((item) => [item.index, item.score]));
    const worstApprox = approxTopK.at(-1)?.score ?? 0;
    for (const exact of exactTopK) {
      const approxScore = approxByIndex.get(exact.index);
      exactTopKErrors.push(Math.abs((approxScore ?? worstApprox) - exact.score));
    }

    const union = new Set([
      ...exactTopK.map((item) => item.index),
      ...approxTopK.map((item) => item.index),
    ]);
    const exactByIndex = new Map(exactScores.map((item) => [item.index, item.score]));
    for (const idx of union) {
      const exactScore = exactByIndex.get(idx) ?? 0;
      const approxScore = approxByIndex.get(idx) ?? 0;
      unionTopKErrors.push(Math.abs(approxScore - exactScore));
    }
  }

  exactTopKErrors.sort((a, b) => a - b);
  unionTopKErrors.sort((a, b) => a - b);

  const queryCount = Math.max(1, queryVectors.length);
  return {
    query_count: queryVectors.length,
    ranking_agreement_at_1: rankingAgreementAt1 / queryCount,
    ranking_overlap_at_5: rankingOverlapAt5 / queryCount,
    ranking_loss_at_5: 1 - rankingOverlapAt5 / queryCount,
    score_mae_on_exact_topk: exactTopKErrors.reduce((s, x) => s + x, 0) / Math.max(1, exactTopKErrors.length),
    score_mse_on_exact_topk: exactTopKErrors.reduce((s, x) => s + x * x, 0) / Math.max(1, exactTopKErrors.length),
    score_p95_abs_error_on_exact_topk: percentile(exactTopKErrors, 0.95),
    score_mae_on_union_topk: unionTopKErrors.reduce((s, x) => s + x, 0) / Math.max(1, unionTopKErrors.length),
    score_mse_on_union_topk: unionTopKErrors.reduce((s, x) => s + x * x, 0) / Math.max(1, unionTopKErrors.length),
  };
}

function buildRetrievalProfile(queryRecords, dbVectors, approxScorer) {
  let relevantRecallAt1 = 0;
  let relevantRecallAt5 = 0;
  let relevantMrr = 0;

  for (const record of queryRecords) {
    const approxTopK = approxScorer(record.vector).slice(0, 5);
    if (approxTopK[0] && record.relevantSet.has(approxTopK[0].index)) {
      relevantRecallAt1++;
    }
    const hitIndex = approxTopK.findIndex((item) => record.relevantSet.has(item.index));
    if (hitIndex >= 0) {
      relevantRecallAt5++;
      relevantMrr += 1 / (hitIndex + 1);
    }
  }

  const denom = Math.max(1, queryRecords.length);
  return {
    query_count: queryRecords.length,
    relevant_recall_at_1: relevantRecallAt1 / denom,
    relevant_recall_at_5: relevantRecallAt5 / denom,
    relevant_mrr: relevantMrr / denom,
  };
}

function clipUnit(x) {
  return Math.max(-1, Math.min(1, x));
}

function buildLloydMaxAux(centroids, boundaries) {
  return {
    encode(values) {
      const indices = new Uint8Array(values.length);
      for (let i = 0; i < values.length; i++) {
        const v = clipUnit(values[i] ?? 0);
        let idx = centroids.length - 1;
        for (let j = 0; j < centroids.length; j++) {
          if (v <= boundaries[j + 1]) {
            idx = j;
            break;
          }
        }
        indices[i] = idx;
      }
      return indices;
    },
    decode(indices) {
      const out = new Float32Array(indices.length);
      for (let i = 0; i < indices.length; i++) {
        out[i] = centroids[indices[i]] ?? 0;
      }
      return out;
    },
  };
}

function normalizeFloat32(values) {
  let norm2 = 0;
  for (let i = 0; i < values.length; i++) {
    norm2 += values[i] * values[i];
  }
  const invNorm = norm2 > 0 ? 1 / Math.sqrt(norm2) : 1;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    out[i] = values[i] * invNorm;
  }
  return out;
}

const files = listFiles();
const chunks = [];
const fileTexts = new Map();
let approximateTokenCount = 0;

for (const file of files) {
  const text = readFileSync(file, 'utf8');
  fileTexts.set(file, text);
  for (const chunk of chunkText(text, CHUNK_BYTES)) {
    approximateTokenCount += approximateTokens(chunk);
    chunks.push({ file, text: chunk });
  }
}

if (chunks.length === 0) {
  throw new Error('No chunks generated for open test');
}

const chunkIndicesByFile = new Map();
for (let i = 0; i < chunks.length; i++) {
  const list = chunkIndicesByFile.get(chunks[i].file) ?? [];
  list.push(i);
  chunkIndicesByFile.set(chunks[i].file, list);
}

const fileQueries = files
  .slice(0, QUERY_LIMIT)
  .map((file) => ({
    file,
    query: fileStemQuery(file, fileTexts.get(file) ?? ''),
    relevantSet: new Set(chunkIndicesByFile.get(file) ?? []),
  }))
  .filter((item) => item.query.length > 0 && item.relevantSet.size > 0);

const vectorizerEntries = [
  ['token_hash', createTokenHashVectorizer(DIM)],
  ['hashed_tfidf', buildHashedTfidfVectorizer(DIM, chunks.map((chunk) => chunk.text))],
];

const vectorizerProfiles = {};
let canonicalProfile = null;

for (const [vectorizerName, vectorizer] of vectorizerEntries) {
  const vectors = chunks.map((chunk) => vectorizer.embed(chunk.text));
  const queryRecords = fileQueries.map((item) => ({
    ...item,
    vector: vectorizer.embed(item.query),
  }));
  const queryVectors = queryRecords.map((item) => item.vector);

  const t0 = performance.now();
  const compressed = compressVectors({
    vectors,
    dimensions: DIM,
    bitsPerValue: BITS_PER_VALUE,
    seed: ROTATION_SEED,
    codebookType: SHIPPED_CODEBOOK,
    includeQJL: false,
  });
  const t1 = performance.now();

  const exactScorer = (queryVector) => vectors.map((vector) => dot(queryVector, vector));
  const publicApproxScorer = (queryVector) => searchVectors({
    compressed_database_b64: compressed.compressed_database_b64,
    query_vector: queryVector,
    top_k: 5,
    metric: 'cosine',
  }).results;

  const publicQuantization = buildQuantizationProfile(queryVectors, exactScorer, publicApproxScorer);
  const publicRetrieval = buildRetrievalProfile(queryRecords, vectors, publicApproxScorer);

  const rotation = new RotationEngine(DIM, ROTATION_SEED);
  const lloydMax = computeLloydMaxBetaCodebook(DIM, 4);
  const lloydAux = buildLloydMaxAux(lloydMax.centroids, lloydMax.boundaries);
  const quantizedVectors = vectors.map((vector) => {
    const source = Float32Array.from(vector);
    const rotated = rotation.rotate(source);
    const normalized = normalizeFloat32(rotated);
    const encoded = lloydAux.encode(normalized);
    return lloydAux.decode(encoded);
  });

  const lloydApproxScorer = (queryVector) => {
    const rotatedQuery = normalizeFloat32(rotation.rotate(Float32Array.from(queryVector)));
    return quantizedVectors
      .map((vector, index) => ({ index, score: dot(rotatedQuery, vector) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);
  };

  const lloydQuantization = buildQuantizationProfile(queryVectors, exactScorer, lloydApproxScorer);
  const lloydRetrieval = buildRetrievalProfile(queryRecords, vectors, lloydApproxScorer);

  const profile = {
    vectorizer_id: vectorizer.id,
    vectorizer_kind: vectorizer.kind,
    codebook_type: compressed.codebook_type,
    compression_ratio: compressed.compression_ratio,
    compressed_bytes: compressed.compressed_bytes,
    compress_ms: t1 - t0,
    query_count: queryRecords.length,
    quantization: {
      [compressed.codebook_type]: publicQuantization,
      lloyd_max_beta: lloydQuantization,
    },
    retrieval: {
      [compressed.codebook_type]: publicRetrieval,
      lloyd_max_beta: lloydRetrieval,
    },
    comparison: {
      quantization_ranking_loss_delta:
        lloydQuantization.ranking_loss_at_5 - publicQuantization.ranking_loss_at_5,
      retrieval_recall_at_5_delta:
        lloydRetrieval.relevant_recall_at_5 - publicRetrieval.relevant_recall_at_5,
      score_mae_exact_topk_delta:
        lloydQuantization.score_mae_on_exact_topk - publicQuantization.score_mae_on_exact_topk,
    },
    algorithm_level: compressed.algorithm_level,
    format_version: compressed.format_version,
    include_qjl: compressed.include_qjl,
    warnings: [
      ...compressed.warnings,
      'Lloyd-Max comparison is retained as a sidecar benchmark for contrast with the shipped public path.',
      'Experimental QJL is kept research-only until an estimator/search path is reproducible on this corpus.',
    ],
  };

  vectorizerProfiles[vectorizerName] = profile;
  if (vectorizerName === 'hashed_tfidf') {
    canonicalProfile = profile;
  }
}

if (!canonicalProfile) {
  throw new Error('hashed_tfidf canonical profile missing');
}

function assertFinite(name, value) {
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite, got ${value}`);
  }
}

const report = {
  test_name: 'turboquant_open_test_local',
  files: files.length,
  chunks: chunks.length,
  approximate_tokens: approximateTokenCount,
  token_count_source: 'approximate_bytes_div_4',
  dimensions: DIM,
  vector_count: chunks.length,
  compression_ratio: canonicalProfile.compression_ratio,
  compressed_bytes: canonicalProfile.compressed_bytes,
  original_bytes_estimate: chunks.length * DIM * 4,
  compress_ms: canonicalProfile.compress_ms,
  query_count: canonicalProfile.query_count,
  codebook_type: canonicalProfile.codebook_type,
  recall_at_1: canonicalProfile.retrieval[canonicalProfile.codebook_type].relevant_recall_at_1,
  recall_at_5: canonicalProfile.retrieval[canonicalProfile.codebook_type].relevant_recall_at_5,
  mrr: canonicalProfile.retrieval[canonicalProfile.codebook_type].relevant_mrr,
  score_mae_on_exact_topk: canonicalProfile.quantization[canonicalProfile.codebook_type].score_mae_on_exact_topk,
  score_mse_on_exact_topk: canonicalProfile.quantization[canonicalProfile.codebook_type].score_mse_on_exact_topk,
  score_p95_abs_error_on_exact_topk: canonicalProfile.quantization[canonicalProfile.codebook_type].score_p95_abs_error_on_exact_topk,
  score_mae_on_union_topk: canonicalProfile.quantization[canonicalProfile.codebook_type].score_mae_on_union_topk,
  score_mse_on_union_topk: canonicalProfile.quantization[canonicalProfile.codebook_type].score_mse_on_union_topk,
  algorithm_level: canonicalProfile.algorithm_level,
  format_version: canonicalProfile.format_version,
  include_qjl: canonicalProfile.include_qjl,
  warnings: canonicalProfile.warnings,
  profiles: vectorizerProfiles,
};

if (report.files <= 0) throw new Error('open test must scan at least one file');
if (report.chunks <= 0) throw new Error('open test must produce at least one chunk');
if (report.query_count <= 0) throw new Error('query_count must be positive');
if (report.approximate_tokens <= 0) throw new Error('open test approximate token count must be positive');
if (report.compression_ratio <= 1) throw new Error(`compression_ratio must be > 1, got ${report.compression_ratio}`);
if (report.format_version !== 3) throw new Error(`format_version must be 3, got ${report.format_version}`);
if (report.include_qjl !== false) throw new Error('include_qjl must be false for LEVEL_1 public beta');
if (report.codebook_type !== SHIPPED_CODEBOOK) {
  throw new Error(`codebook_type must be ${SHIPPED_CODEBOOK}, got ${report.codebook_type}`);
}
if (report.algorithm_level !== 'LEVEL_1_PUBLIC_BETA') {
  throw new Error(`algorithm_level must be LEVEL_1_PUBLIC_BETA for open test, got ${report.algorithm_level}`);
}

for (const [name, value] of Object.entries(report)) {
  if (typeof value === 'number') {
    assertFinite(name, value);
  }
}

const outputPath = process.env.TQ_OPEN_TEST_OUTPUT;
const json = JSON.stringify(report, null, 2);

if (outputPath) {
  writeFileSync(outputPath, `${json}\n`, 'utf8');
}

console.log(json);
