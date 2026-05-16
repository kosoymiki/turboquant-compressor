#!/usr/bin/env node
/**
 * Gate: scan all tool descriptions in dist/server.js for prompt-injection patterns.
 * Reads tools/list from the live server and checks every name+description.
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

function fail(msg) {
  console.error(`ERROR [verify-tool-metadata]: ${msg}`);
  process.exit(1);
}

const FORBIDDEN = [
  /ignore (previous|prior|system|user) instructions/i,
  /do not tell the user/i,
  /secretly/i,
  /exfiltrate/i,
  /bypass.{0,20}permission/i,
  /override.{0,20}permission/i,
  /yolo/i,
  /hidden instruction/i,
  /without (the user|user) knowing/i,
];

const serverPath = join(rootDir, 'dist', 'server.js');
if (!existsSync(serverPath)) fail('dist/server.js not found — run npm run build');

const proc = spawn('node', [serverPath], { stdio: ['pipe', 'pipe', 'pipe'], cwd: rootDir });
let buf = '';
proc.stdout.on('data', (d) => { buf += d.toString(); });

async function recv(timeoutMs = 5000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const nl = buf.indexOf('\n');
    if (nl >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) return JSON.parse(line);
    }
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error('timeout');
}

proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'gate', version: '1.0' } } }) + '\n');
await recv();
proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized', params: {} }) + '\n');
proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/list' }) + '\n');
const listResp = await recv();
proc.stdin.end();
proc.kill();

const tools = listResp?.result?.tools ?? [];
if (tools.length === 0) fail('tools/list returned no tools');

let violations = 0;
for (const tool of tools) {
  const fields = [tool.name ?? '', tool.description ?? ''];
  for (const field of fields) {
    for (const pat of FORBIDDEN) {
      if (pat.test(field)) {
        console.error(`  FAIL tool=${tool.name}: forbidden pattern ${pat} in: ${field.slice(0, 100)}`);
        violations++;
      }
    }
  }
}

if (violations > 0) fail(`${violations} forbidden metadata pattern(s) found`);
console.log(`[OK] verify-tool-metadata: ${tools.length} tools scanned, 0 forbidden patterns`);
