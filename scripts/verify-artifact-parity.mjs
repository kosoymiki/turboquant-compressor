#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, mkdtempSync, rmSync, readdirSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';

function fail(msg) {
  console.error(`ERROR: ${msg}`);
  process.exit(1);
}

function run(cmd, args) {
  const res = spawnSync(cmd, args, { encoding: 'utf8' });
  if (res.status !== 0) {
    fail(`${cmd} ${args.join(' ')} failed:\n${res.stderr || res.stdout}`);
  }
  return res.stdout;
}

function sha256(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function walk(root, exclude = new Set()) {
  const out = [];

  function visit(dir) {
    for (const entry of readdirSync(dir)) {
      if (exclude.has(entry)) continue;
      const full = join(dir, entry);
      const st = statSync(full);
      if (st.isDirectory()) visit(full);
      else if (st.isFile()) out.push(relative(root, full).replaceAll('\\', '/'));
    }
  }

  visit(root);
  return out.sort();
}

const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
const version = pkg.version;

const source = process.env.TQ_SOURCE_ARCHIVE ?? `../turboquant-compressor-${version}-source.tar.gz`;
const portable = process.env.TQ_PORTABLE_ARCHIVE ?? `../turboquant-compressor-${version}-portable.tar.gz`;

const tmp = mkdtempSync(join(tmpdir(), 'tq-parity-'));

try {
  run('mkdir', ['-p', `${tmp}/source`, `${tmp}/portable`]);
  run('tar', ['-xzf', source, '-C', `${tmp}/source`]);
  run('tar', ['-xzf', portable, '-C', `${tmp}/portable`]);

  const srcRoot = `${tmp}/source/turboquant-compressor`;
  const portRoot = `${tmp}/portable/turboquant-compressor`;

  const sourceList = walk(srcRoot);
  const portableList = walk(portRoot, new Set(['node_modules', 'dist']));

  // Exclude generated runtime evidence from parity comparison
  function isGeneratedEvidence(rel) {
    return (
      rel.startsWith('forensics/') ||
      rel === 'forensics/mcp-conformance-transcript.json' ||
      /transcript.*\.(json|jsonl)$/.test(rel)
    );
  }

  const srcFiltered = sourceList.filter(x => !isGeneratedEvidence(x));
  const portFiltered = portableList.filter(x => !isGeneratedEvidence(x));

  const srcSet = new Set(srcFiltered);
  const portSet = new Set(portFiltered);

  const sourceOnly = srcFiltered.filter((x) => !portSet.has(x));
  const portableOnly = portFiltered.filter((x) => !srcSet.has(x));

  if (sourceOnly.length || portableOnly.length) {
    console.error('Source-only files:');
    console.error(sourceOnly.join('\n') || '(none)');
    console.error('Portable-only files:');
    console.error(portableOnly.join('\n') || '(none)');
    fail('source and portable archives are not built from the same source tree');
  }

  for (const file of srcFiltered) {
    const srcPath = join(srcRoot, file);
    const portPath = join(portRoot, file);
    const srcHash = sha256(srcPath);
    const portHash = sha256(portPath);
    if (srcHash !== portHash) {
      fail(`source/portable content mismatch: ${file}`);
    }
  }

  console.log('[OK] source/portable artifact parity verified');
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
