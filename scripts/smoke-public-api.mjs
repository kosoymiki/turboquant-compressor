#!/usr/bin/env node

const mod = await import('../dist/index.js');

if (typeof mod.compressVectors !== 'function') {
  throw new Error('compressVectors export missing');
}

if (typeof mod.searchVectors !== 'function') {
  throw new Error('searchVectors export missing');
}

// Verify research namespace
if (!mod.research) {
  throw new Error('research namespace export missing');
}

if (typeof mod.research.LloydMaxCodebook !== 'function') {
  throw new Error('research.LloydMaxCodebook export missing');
}

if (typeof mod.research.BetaCodebook !== 'function') {
  throw new Error('research.BetaCodebook export missing');
}

if (typeof mod.research.QJLResidualEstimator !== 'function') {
  throw new Error('research.QJLResidualEstimator export missing');
}

const compressed = mod.compressVectors({
  vectors: [[1, 0, 0, 0], [0, 1, 0, 0]],
  dimensions: 4,
  bitsPerValue: 4,
  seed: 42,
});

if (!compressed.compressed_database_b64) {
  throw new Error('compressVectors did not return compressed_database_b64');
}

const result = mod.searchVectors({
  compressed_database_b64: compressed.compressed_database_b64,
  query_vector: [1, 0, 0, 0],
  dimensions: 4,
  top_k: 1,
  metric: 'cosine',
});

if (result.results[0]?.index !== 0) {
  throw new Error('public API self-match failed');
}

console.log('[OK] Public API smoke passed');