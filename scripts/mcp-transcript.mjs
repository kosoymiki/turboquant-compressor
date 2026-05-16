#!/usr/bin/env node
/**
 * MCP raw transcript capture — single session queue, no listener leaks.
 */

import { spawn } from 'child_process';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = join(__dirname, '..');

const forensicsDir = join(rootDir, 'forensics');
mkdirSync(forensicsDir, { recursive: true });

function sampleArgs(toolName) {
  switch (toolName) {
    case 'turboquant_compress':
      return { vectors: [[1, 2, 3, 4], [4, 3, 2, 1]], dimensions: 4, bitsPerValue: 4 };
    case 'turboquant_vector_search':
      return { query: [1, 2, 3, 4], index: [[1, 2, 3, 4], [5, 6, 7, 8]], top_k: 1 };
    case 'turboquant_cost_analyze':
      return { usage: [{ inputTokens: 100, outputTokens: 20, cacheCreationInputTokens: 30, cacheReadInputTokens: 50, model: 'manual-test', timestamp: '2026-05-15T00:00:00Z' }] };
    case 'turboquant_cache_plan':
      return { blocks: [{ label: 'stable', text: 'Stable project context.' }, { label: 'volatile', text: 'Current user request.' }], target: 'generic_mcp_context_hygiene' };
    case 'turboquant_prompt_cache_lint':
      return { blocks: [{ label: 'system', text: 'Stable instructions.' }, { label: 'task', text: 'Current task.' }] };
    case 'turboquant_context_pack_build':
      return { files: [{ path: 'README.md', text: 'TurboQuant context pack test content.' }], dimensions: 16, chunkBytes: 256, bitsPerValue: 4, storageMode: 'inline_text' };
    case 'turboquant_cli_mcp_profile':
      return { host: 'generic_stdio' };
    case 'turboquant_quantize':
      return { action: 'benchmark_mse', dimension: 64, bits: 3 };
    case 'turboquant_kv_analyze':
      return { action: 'estimate_savings', headDim: 128, numKvHeads: 8, numLayers: 32, seqLen: 4096 };
    case 'turboquant_backend_probe':
      return { deep: false, timeoutMs: 250 };
    case 'turboquant_opencl_probe':
      return { deep: false, timeoutMs: 500 };
    case 'turboquant_adreno_loader_probe':
      return { deep: false };
    default:
      return {};
  }
}

class Session {
  constructor(proc) {
    this.proc = proc;
    this.buf = '';
    this.stderr = '';
    proc.stdout.on('data', (d) => { this.buf += d.toString('utf8'); });
    proc.stderr.on('data', (d) => { this.stderr += d.toString('utf8'); });
  }

  send(msg) {
    this.proc.stdin.write(JSON.stringify(msg) + '\n');
  }

  async recv(timeoutMs = 6000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const idx = this.buf.indexOf('\n');
      if (idx >= 0) {
        const line = this.buf.slice(0, idx).trim();
        this.buf = this.buf.slice(idx + 1);
        if (!line) continue;
        return JSON.parse(line);
      }
      await new Promise((r) => setTimeout(r, 20));
    }
    throw new Error(`Timeout waiting for response. stderr=${this.stderr.slice(-200)}`);
  }

  async call(id, method, params = {}) {
    const req = { jsonrpc: '2.0', id, method, params };
    this.send(req);
    const response = await this.recv();
    return { request: req, response };
  }
}

async function captureTranscript() {
  const serverPath = join(rootDir, 'dist', 'server.js');
  if (!existsSync(serverPath)) {
    console.error('dist/server.js not found — run npm run build');
    process.exit(1);
  }

  const server = spawn('node', [serverPath], {
    stdio: ['pipe', 'pipe', 'pipe'],
    cwd: rootDir,
  });
  server.on('error', (err) => { console.error('spawn error:', err.message); process.exit(1); });

  const session = new Session(server);
  const transcript = [];
  let callId = 1;

  try {
    // initialize
    const init = await session.call(callId++, 'initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'transcript-capture', version: '1.0.0' },
    });
    transcript.push({ phase: 'initialize', ...init });

    // initialized notification
    const notif = { jsonrpc: '2.0', method: 'notifications/initialized', params: {} };
    session.send(notif);
    transcript.push({ phase: 'initialized', notification: notif });

    // tools/list
    const listEntry = await session.call(callId++, 'tools/list');
    transcript.push({ phase: 'tools/list', ...listEntry });

    const tools = listEntry.response?.result?.tools ?? [];

    // tools/call for each tool (two-step for context_pack_search)
    let builtManifest = null;
    let compressDb64 = null;
    for (const tool of tools) {
      if (tool.name === 'turboquant_context_pack_search') continue;
      if (tool.name === 'turboquant_vector_search') continue; // handled after compress

      const args = sampleArgs(tool.name);
      const entry = await session.call(callId++, 'tools/call', { name: tool.name, arguments: args });
      transcript.push({ phase: `tools/call:${tool.name}`, ...entry });

      if (tool.name === 'turboquant_context_pack_build') {
        try {
          const text = entry.response?.result?.content?.[0]?.text;
          if (text) {
            const parsed = JSON.parse(text);
            builtManifest = parsed.manifest ?? parsed;
          }
        } catch (_) {}
      }
      if (tool.name === 'turboquant_compress') {
        try {
          const text = entry.response?.result?.content?.[0]?.text;
          if (text) {
            const parsed = JSON.parse(text);
            compressDb64 = parsed.compressed_database_b64 ?? parsed.compressedDatabaseB64;
          }
        } catch (_) {}
      }
    }

    // vector_search using compressed_database_b64 from compress output
    const vsArgs = compressDb64
      ? { compressed_database_b64: compressDb64, query_vector: [1, 2, 3, 4], top_k: 1 }
      : { compressed_database_b64: '', query_vector: [1, 2, 3, 4], top_k: 1 };
    const vsEntry = await session.call(callId++, 'tools/call', {
      name: 'turboquant_vector_search',
      arguments: vsArgs,
    });
    transcript.push({ phase: 'tools/call:turboquant_vector_search', ...vsEntry });

    // context_pack_search using built manifest
    const searchArgs = builtManifest
      ? { manifest: builtManifest, query: 'test content', top_k: 1 }
      : { manifest: null, query: 'test', top_k: 1 };
    const searchEntry = await session.call(callId++, 'tools/call', {
      name: 'turboquant_context_pack_search',
      arguments: searchArgs,
    });
    transcript.push({ phase: 'tools/call:turboquant_context_pack_search', ...searchEntry });

    // invalid method
    const invalidEntry = await session.call(callId++, 'invalid/method');
    transcript.push({ phase: 'invalid_method', ...invalidEntry });

    server.stdin.end();
    transcript.push({ phase: 'server_shutdown' });

  } catch (err) {
    console.error('Capture error:', err.message);
    transcript.push({ phase: 'error', error: err.message });
  } finally {
    server.kill();
  }

  // Check for isError in any tools/call response
  const errors = transcript.filter((e) =>
    e.phase?.startsWith('tools/call') && e.response?.result?.isError === true
  );
  if (errors.length > 0) {
    console.error('\nWARNING: isError=true in transcript:');
    for (const e of errors) {
      const text = e.response?.result?.content?.[0]?.text ?? '';
      console.error(`  ${e.phase}: ${text.slice(0, 200)}`);
    }
  }

  const transcriptPath = join(forensicsDir, 'mcp-stdio-transcript.jsonl');
  writeFileSync(transcriptPath, transcript.map((r) => JSON.stringify(r)).join('\n') + '\n');
  console.log(`Transcript written to ${transcriptPath}`);
  console.log(`Total entries: ${transcript.length}, isError entries: ${errors.length}`);

  if (errors.length > 0) process.exit(1);
}

captureTranscript().catch((err) => {
  console.error('Fatal:', err.message);
  process.exit(1);
});
