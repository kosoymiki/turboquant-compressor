#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { readFileSync, existsSync } from 'node:fs';
import { createInterface } from 'node:readline';

function fail(msg) {
  console.error(`ERROR: ${msg}`);
  process.exit(1);
}

function listTarStreaming(path, onLine) {
  return new Promise((resolve, reject) => {
    if (!existsSync(path)) {
      if (process.env.TQ_ALLOW_MISSING_ARCHIVES === '1') {
        console.error(`[WARN] archive not found yet: ${path}`);
        resolve(null);
        return;
      }
      fail(`archive not found: ${path}`);
    }

    const proc = spawn('tar', ['-tzf', path], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stderr = '';
    proc.stderr.on('data', (d) => { stderr += d.toString('utf8'); });

    const rl = createInterface({ input: proc.stdout });

    let lineCount = 0;
    rl.on('line', (line) => {
      lineCount++;
      onLine(line);
    });

    proc.on('error', (err) => {
      reject(new Error(`tar spawn failed for ${path}: ${err.message}`));
    });

    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`tar -tzf failed for ${path}: ${stderr}`));
      } else {
        resolve(lineCount);
      }
    });
  });
}

const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
const version = pkg.version;

const archives = [
  {
    path: process.env.TQ_SOURCE_ARCHIVE ?? `../turboquant-compressor-${version}-source.tar.gz`,
    source: true,
  },
  {
    path: process.env.TQ_PORTABLE_ARCHIVE ?? `../turboquant-compressor-${version}-portable.tar.gz`,
    source: false,
  },
];

const forbiddenCommon = [
  /\.tar\.gz$/,
  /\.tgz$/,
  /\.claude\/settings\.local\.json$/,
  /\.claude\/.*\.local\.json$/,
  /(^|\/)\.git(\/|$)/,
];

const forbiddenSource = [
  /(^|\/)node_modules(\/|$)/,
  /(^|\/)dist(\/|$)/,
  /(^|\/)coverage(\/|$)/,
  /(^|\/)forensics(\/|$)/,
  /mcp-conformance-transcript\.json$/,
  /(^|\/).*transcript.*\.jsonl$/,
  /(^|\/)native\/.*\/build(\/|$)/,
  /CMakeCache\.txt$/,
  /(^|\/)CMakeFiles(\/|$)/,
  /cmake_install\.cmake$/,
  /\/build\/tq_opencl_cli$/,
  /\/build\/test_.*_parity$/,
];

async function verifyArchive(archive) {
  let foundForbidden = null;
  let lineCount = 0;

  try {
    lineCount = await listTarStreaming(archive.path, (line) => {
      for (const pattern of forbiddenCommon) {
        if (pattern.test(line)) {
          foundForbidden = { pattern, line };
          return;
        }
      }
      if (archive.source) {
        for (const pattern of forbiddenSource) {
          if (pattern.test(line)) {
            foundForbidden = { pattern, line };
            return;
          }
        }
      }
    });

    if (foundForbidden) {
      fail(`${archive.path} contains forbidden path matching ${foundForbidden.pattern}: ${foundForbidden.line}`);
    }

    console.log(`[OK] ${archive.path} scanned (${lineCount} entries)`);
  } catch (err) {
    fail(err.message);
  }
}

(async () => {
  for (const archive of archives) {
    await verifyArchive(archive);
  }
  console.log('[OK] archive hygiene verified');
})();