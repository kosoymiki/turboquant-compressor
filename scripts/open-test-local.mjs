#!/usr/bin/env node

import { readFileSync, existsSync, writeFileSync } from 'node:fs';
import { statSync, readdirSync } from 'node:fs';
import { join, extname } from 'node:path';
import { performance } from 'node:perf_hooks';
import { compressVectors, searchVectors } from '../dist/index.js';

const DIM = Number(process.env.TQ_DIM ?? 384);
const CHUNK_BYTES = Number(process.env.TQ_CHUNK_BYTES ?? 4096);
const QUERY_LIMIT = Number(process.env.TQ_QUERY_LIMIT ?? 50);

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

function hash32(str, seed = 2166136261) {
  let h = seed >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function tokenHashVector(text, dim) {
  const out = new Array(dim).fill(0);
  const tokens = text.split(/\s+/).filter(Boolean);

  for (const token of tokens) {
    const h = hash32(token);
    const idx = h % dim;
    const sign = (h & 1) === 0 ? 1 : -1;
    out[idx] += sign;
  }

  let norm = Math.sqrt(out.reduce((s, x) => s + x * x, 0));
  if (norm === 0) norm = 1;

  return out.map((x) => x / norm);
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

const files = listFiles();
const chunks = [];
let approximateTokenCount = 0;

for (const file of files) {
  const text = readFileSync(file, 'utf8');
  for (const chunk of chunkText(text, CHUNK_BYTES)) {
    approximateTokenCount += approximateTokens(chunk);
    chunks.push({ file, text: chunk });
  }
}

if (chunks.length === 0) {
  throw new Error('No chunks generated for open test');
}

const vectors = chunks.map((c) => tokenHashVector(c.text, DIM));

const t0 = performance.now();
const compressed = compressVectors({
  vectors,
  dimensions: DIM,
  bitsPerValue: 4,
  seed: 42,
});
const t1 = performance.now();

let recall1 = 0;
let recall5 = 0;
let mrr = 0;
const exactTopKErrors = [];
const unionTopKErrors = [];
const queryCount = Math.min(chunks.length, QUERY_LIMIT);

for (let i = 0; i < queryCount; i++) {
  const query = vectors[i];

  const exactScores = vectors
    .map((v, index) => ({ index, score: dot(query, v) }))
    .sort((a, b) => b.score - a.score);

  const exactTopK = exactScores.slice(0, 5);

  const result = searchVectors({
    compressed_database_b64: compressed.compressed_database_b64,
    query_vector: query,
    top_k: 5,
    metric: 'cosine',
  });

  const approxTopK = result.results;

  // Ranking metrics
  if (result.results[0]?.index === exactScores[0]?.index) recall1++;
  if (result.results.some((r) => r.index === exactScores[0]?.index)) recall5++;

  // MRR
  const exactBest = exactScores[0]?.index;
  const rank = result.results.findIndex((r) => r.index === exactBest);
  mrr += rank >= 0 ? 1 / (rank + 1) : 0;

  // Score calibration on exact top-k (by same index)
  const approxByIndex = new Map(approxTopK.map((r) => [r.index, r.score]));
  const worstApprox = approxTopK.at(-1)?.score ?? 0;

  for (const exact of exactTopK) {
    const approxScore = approxByIndex.get(exact.index);
    if (approxScore !== undefined) {
      exactTopKErrors.push(Math.abs(approxScore - exact.score));
    } else {
      // Candidate missing from approximate top-k: penalize
      exactTopKErrors.push(Math.abs(worstApprox - exact.score));
    }
  }

  // Score calibration on union top-k (all candidates seen by either method)
  const union = new Set([
    ...exactTopK.map((x) => x.index),
    ...approxTopK.map((x) => x.index),
  ]);

  const exactByIndex = new Map(exactScores.map((x) => [x.index, x.score]));

  for (const idx of union) {
    const exactScore = exactByIndex.get(idx) ?? 0;
    const approxScore = approxByIndex.get(idx) ?? 0;
    unionTopKErrors.push(Math.abs(approxScore - exactScore));
  }
}

exactTopKErrors.sort((a, b) => a - b);
unionTopKErrors.sort((a, b) => a - b);

const score_mae_on_exact_topk = exactTopKErrors.reduce((s, x) => s + x, 0) / Math.max(1, exactTopKErrors.length);
const score_mse_on_exact_topk = exactTopKErrors.reduce((s, x) => s + x * x, 0) / Math.max(1, exactTopKErrors.length);
const score_p95_abs_error_on_exact_topk = percentile(exactTopKErrors, 0.95);
const score_mae_on_union_topk = unionTopKErrors.reduce((s, x) => s + x, 0) / Math.max(1, unionTopKErrors.length);
const score_mse_on_union_topk = unionTopKErrors.reduce((s, x) => s + x * x, 0) / Math.max(1, unionTopKErrors.length);

function assertFinite(name, value) {
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite, got ${value}`);
  }
}

function assertInRange(name, value, min, max) {
  if (value < min || value > max) {
    throw new Error(`${name} must be in [${min}, ${max}], got ${value}`);
  }
}

const report = {
  test_name: 'turboquant_open_test_local',
  files: files.length,
  chunks: chunks.length,
  approximate_tokens: approximateTokenCount,
  token_count_source: 'approximate_bytes_div_4',
  dimensions: DIM,
  vector_count: vectors.length,
  compression_ratio: compressed.compression_ratio,
  compressed_bytes: compressed.compressed_bytes,
  original_bytes_estimate: compressed.original_bytes_estimate,
  compress_ms: t1 - t0,
  query_count: queryCount,
  // Ranking metrics
  recall_at_1: recall1 / queryCount,
  recall_at_5: recall5 / queryCount,
  mrr: mrr / queryCount,
  // Score calibration metrics (by same index)
  score_mae_on_exact_topk,
  score_mse_on_exact_topk,
  score_p95_abs_error_on_exact_topk,
  score_mae_on_union_topk,
  score_mse_on_union_topk,
  algorithm_level: compressed.algorithm_level,
  format_version: compressed.format_version,
  include_qjl: compressed.include_qjl,
  warnings: compressed.warnings,
};

// Hard assertions for fail-closed gate
if (report.files <= 0) throw new Error('open test must scan at least one file');
if (report.chunks <= 0) throw new Error('open test must produce at least one chunk');
if (report.vector_count <= 0) throw new Error('open test must produce at least one vector');
if (report.approximate_tokens <= 0) throw new Error('open test approximate token count must be positive');
if (report.compression_ratio <= 1) throw new Error(`compression_ratio must be > 1, got ${report.compression_ratio}`);
if (report.format_version !== 3) throw new Error(`format_version must be 3, got ${report.format_version}`);
if (report.include_qjl !== false) throw new Error('include_qjl must be false for LEVEL_0');
if (report.algorithm_level !== 'LEVEL_0_TURBOQUANT_INSPIRED_MVP') {
  throw new Error(`algorithm_level must be LEVEL_0_TURBOQUANT_INSPIRED_MVP for open test, got ${report.algorithm_level}`);
}
if (report.recall_at_5 < report.recall_at_1) {
  throw new Error(`recall_at_5 ${report.recall_at_5} must be >= recall_at_1 ${report.recall_at_1}`);
}
if (report.query_count <= 0) throw new Error('query_count must be positive');

// Validate score metrics
assertInRange('mrr', report.mrr, 0, 1);
assertFinite('score_mae_on_exact_topk', report.score_mae_on_exact_topk);
assertFinite('score_mse_on_exact_topk', report.score_mse_on_exact_topk);
assertFinite('score_p95_abs_error_on_exact_topk', report.score_p95_abs_error_on_exact_topk);
assertFinite('score_mae_on_union_topk', report.score_mae_on_union_topk);
assertFinite('score_mse_on_union_topk', report.score_mse_on_union_topk);

for (const [name, value] of Object.entries(report)) {
  if (typeof value === 'number') assertFinite(name, value);
}

// Save report if TQ_OPEN_TEST_OUTPUT is set
const outputPath = process.env.TQ_OPEN_TEST_OUTPUT;
const json = JSON.stringify(report, null, 2);

if (outputPath) {
  writeFileSync(outputPath, `${json}\n`, 'utf8');
}

console.log(json);
