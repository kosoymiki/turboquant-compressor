#!/usr/bin/env node

const mod = await import('../dist/index.js');

const r = mod.compressVectors({
  vectors: [[1, 0, 0, 0]],
  dimensions: 4,
  bitsPerValue: 4,
  seed: 42
});

const db = mod.decodeCompressedDatabase(r.compressed_database_b64);

function fail(msg) {
  console.error(`ERROR: ${msg}`);
  process.exit(1);
}

if (r.format_version !== db.version) {
  fail(`compress output format_version ${r.format_version} != decoded version ${db.version}`);
}
if (db.version !== 3) fail(`expected format v3, got ${db.version}`);
if (db.headerLength !== 80) fail(`expected headerLength 80, got ${db.headerLength}`);
if (db.qjlLength !== 0) fail(`public LEVEL_1 beta path must not store QJL payload`);

console.log('[OK] format contract verified');
