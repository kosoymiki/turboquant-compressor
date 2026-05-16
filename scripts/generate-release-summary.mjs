#!/usr/bin/env node
/**
 * Generates docs/RELEASE_SUMMARY.md from actual command outputs.
 * Single source of truth for release status.
 */

import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, dirname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, { encoding: 'utf8', cwd: rootDir, ...opts });
  return {
    ok: res.status === 0,
    stdout: (res.stdout ?? '').trim(),
    stderr: (res.stderr ?? '').trim(),
    status: res.status,
  };
}

function section(title, content) {
  return `## ${title}\n\n${content}\n`;
}

function codeBlock(lang, content) {
  return `\`\`\`${lang}\n${content}\n\`\`\``;
}

function fileHash(filePath) {
  if (!existsSync(filePath)) return 'not found';
  const content = readFileSync(filePath);
  return createHash('sha256').update(content).digest('hex');
}

function resolveMaybeAbsolute(p) {
  return isAbsolute(p) ? p : resolve(rootDir, p);
}

const pkg = JSON.parse(readFileSync(join(rootDir, 'package.json'), 'utf8'));
const version = pkg.version;
const now = new Date().toISOString();
const nodeVersion = process.version;
const npmVersion = run('npm', ['-v']).stdout;

const lines = [];
lines.push(`# Release Summary — turboquant-compressor v${version}`);
lines.push('');
lines.push(`Generated: ${now}`);
lines.push(`Node: ${nodeVersion} | npm: ${npmVersion}`);
lines.push('');

// 1. Version
lines.push(section('Version', `**${version}**`));

// 2. Build
const distExists = existsSync(join(rootDir, 'dist', 'server.js'));
lines.push(section('Build', distExists
  ? `dist/server.js present. Build OK.`
  : `**WARN**: dist/server.js not found — run \`npm run build\` first.`));

// 3. tools/list via MCP
let toolNames = [];
if (distExists) {
  const { spawn } = await import('node:child_process');
  const toolsResult = await new Promise((resolve) => {
    const proc = spawn('node', [join(rootDir, 'dist', 'server.js')], { stdio: ['pipe', 'pipe', 'pipe'], cwd: rootDir });
    let buf = '';
    proc.stdout.on('data', d => { buf += d.toString(); });
    const send = (msg) => proc.stdin.write(JSON.stringify(msg) + '\n');

    const deadline = setTimeout(() => { proc.kill(); resolve(null); }, 8000);

    send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'release-summary', version: '1.0' } } });

    const waitForLine = () => {
      const nl = buf.indexOf('\n');
      if (nl !== -1) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        return line;
      }
      return null;
    };

    const poll = () => {
      const line = waitForLine();
      if (line) {
        try {
          const msg = JSON.parse(line);
          if (msg.id === 1) {
            send({ jsonrpc: '2.0', method: 'notifications/initialized', params: {} });
            send({ jsonrpc: '2.0', id: 2, method: 'tools/list' });
          } else if (msg.id === 2) {
            clearTimeout(deadline);
            proc.kill();
            resolve(msg.result?.tools ?? []);
            return;
          }
        } catch (_) {}
      }
      setTimeout(poll, 20);
    };
    setTimeout(poll, 50);
  });

  if (Array.isArray(toolsResult)) {
    toolNames = toolsResult.map(t => t.name).sort();
  }
}

lines.push(section('MCP Tools (tools/list)',
  toolNames.length > 0
    ? codeBlock('', toolNames.map((n, i) => `${i + 1}. ${n}`).join('\n'))
    : '_Could not retrieve — server not built or failed to start._'
));

// 4. Test summary
const testRes = run('node', ['--experimental-vm-modules', 'node_modules/jest/bin/jest.js', '--runInBand', '--no-coverage'], { timeout: 120000 });
const testOutput = (testRes.stdout + '\n' + testRes.stderr).trim();
const summaryMatch = testOutput.match(/Tests?:.*\n?.*passed.*|Test Suites?:.*passed.*/g);
const testSummary = summaryMatch ? summaryMatch.slice(-3).join('\n') : (testRes.ok ? 'Tests passed' : 'Tests FAILED');
lines.push(section('Test Suite', codeBlock('', testSummary) + (testRes.ok ? '\n\nStatus: **PASS**' : '\n\nStatus: **FAIL**')));

// 5. verify:release
const verifyRes = run('node', ['scripts/verify-release.mjs'], { timeout: 30000 });
lines.push(section('verify:release', codeBlock('', (verifyRes.stdout + '\n' + verifyRes.stderr).trim().slice(0, 2000)) + (verifyRes.ok ? '\n\nStatus: **PASS**' : '\n\nStatus: **FAIL**')));

// 6. verify-host-matrix
const matrixRes = run('node', ['scripts/verify-host-matrix.mjs'], { timeout: 10000 });
lines.push(section('verify-host-matrix', codeBlock('', (matrixRes.stdout + '\n' + matrixRes.stderr).trim()) + (matrixRes.ok ? '\n\nStatus: **PASS**' : '\n\nStatus: **FAIL**')));

// 7. MCP conformance
const confRes = run('node', ['scripts/mcp-conformance.mjs'], { timeout: 60000 });
const confOutput = confRes.stdout + '\n' + confRes.stderr;
const confPassed = confOutput.match(/Total:\s*(\d+)\s*passed/);
const confFailed = confOutput.match(/(\d+)\s*failed/);
const confSummary = confPassed && confFailed
  ? `${confPassed[1]} passed, ${confFailed[1]} failed`
  : (confRes.ok ? 'All tests passed' : 'Tests FAILED');
lines.push(section('MCP Conformance', codeBlock('', confOutput.slice(0, 1500)) + (confRes.ok ? '\n\nStatus: **PASS**' : '\n\nStatus: **FAIL**')));

// 8. MCP transcript
const transRes = run('node', ['scripts/mcp-transcript.mjs'], { timeout: 30000 });
lines.push(section('MCP Transcript', codeBlock('', (transRes.stdout + '\n' + transRes.stderr).trim())));

// 9. Smoke tests
const smokeStdio = run('npm', ['run', 'smoke:stdio'], { timeout: 30000 });
const smokeMcp = run('npm', ['run', 'smoke:mcp'], { timeout: 30000 });
const smokeNumeric = run('npm', ['run', 'smoke:numeric'], { timeout: 30000 });
const smokeApi = run('npm', ['run', 'smoke:api'], { timeout: 30000 });

lines.push(section('Smoke Tests', [
  `- stdio: ${smokeStdio.ok ? '**PASS**' : 'FAIL'}`,
  `- mcp: ${smokeMcp.ok ? '**PASS**' : 'FAIL'}`,
  `- numeric: ${smokeNumeric.ok ? '**PASS**' : 'FAIL'}`,
  `- api: ${smokeApi.ok ? '**PASS**' : 'FAIL'}`,
].join('\n')));

// 10. Archive status with env override support
const defaultSource = `../turboquant-compressor-${version}-source.tar.gz`;
const defaultPortable = `../turboquant-compressor-${version}-portable.tar.gz`;
const sourceArchiveInput = process.env.TQ_SOURCE_ARCHIVE || defaultSource;
const portableArchiveInput = process.env.TQ_PORTABLE_ARCHIVE || defaultPortable;

const sourceArchivePath = resolveMaybeAbsolute(sourceArchiveInput);
const portableArchivePath = resolveMaybeAbsolute(portableArchiveInput);

const srcExists = existsSync(sourceArchivePath);
const portExists = existsSync(portableArchivePath);
const srcHash = srcExists ? fileHash(sourceArchivePath) : 'not found';
const portHash = portExists ? fileHash(portableArchivePath) : 'not found';

// Check if archives were explicitly provided via env
const archiveEnvExplicit = Boolean(process.env.TQ_SOURCE_ARCHIVE) || Boolean(process.env.TQ_PORTABLE_ARCHIVE);

lines.push(section('Archives', [
  `- Source (\`${sourceArchiveInput}\`): ${srcExists ? `**present** (SHA256: ${srcHash})` : '_not found_'}`,
  `- Portable (\`${portableArchiveInput}\`): ${portExists ? `**present** (SHA256: ${portHash})` : '_not found_'}`,
].join('\n')));

// 11. Archive verification with env override
let archiveVerOk = false;
let parityOk = false;
if (srcExists && portExists) {
  const archRes = run('node', ['scripts/verify-archives.mjs'], {
    timeout: 30000,
    env: { ...process.env, TQ_SOURCE_ARCHIVE: sourceArchivePath, TQ_PORTABLE_ARCHIVE: portableArchivePath }
  });
  archiveVerOk = archRes.ok;
  const parityRes = run('node', ['scripts/verify-artifact-parity.mjs'], {
    timeout: 30000,
    env: { ...process.env, TQ_SOURCE_ARCHIVE: sourceArchivePath, TQ_PORTABLE_ARCHIVE: portableArchivePath }
  });
  parityOk = parityRes.ok;
  lines.push(section('Archive Verification', [
    `- verify:archives: ${archiveVerOk ? '**PASS**' : 'FAIL'}`,
    `- verify:artifact-parity: ${parityOk ? '**PASS**' : 'FAIL'}`,
  ].join('\n')));
} else if (archiveEnvExplicit) {
  // Archives explicitly provided but not found - this is a failure
  lines.push(section('Archive Verification', '_Archives explicitly provided but not found — **FAIL**_'));
  archiveVerOk = false;
  parityOk = false;
} else {
  lines.push(section('Archive Verification', '_Archives not found — skipping verification_'));
}

// 12. Overall verdict
const allRuntimeOk = distExists && testRes.ok && verifyRes.ok && matrixRes.ok && confRes.ok && toolNames.length === 8;
const allArchiveOk = !archiveEnvExplicit || (srcExists && portExists && archiveVerOk && parityOk);
const allOk = allRuntimeOk && allArchiveOk;

lines.push(section('Overall Verdict', allOk
  ? `**READY** — All gates passed for v${version}.`
  : `**NOT READY** — One or more checks failed. See sections above.`));

const output = lines.join('\n');
const outPath = join(rootDir, 'docs', 'RELEASE_SUMMARY.md');
writeFileSync(outPath, output);
console.log(`[OK] docs/RELEASE_SUMMARY.md written (${output.length} bytes)`);
console.log(`Overall: ${allOk ? 'READY' : 'NOT READY'}`);
if (!allOk) process.exit(1);