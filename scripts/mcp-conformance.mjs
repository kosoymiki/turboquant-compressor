/**
 * MCP conformance test harness — pure JS, no TypeScript annotations.
 * Tests stdio JSON-RPC 2.0 protocol compliance for turboquant-compressor.
 */

import { spawn } from 'child_process';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = join(__dirname, '..');

const EXPECTED_TOOLS = [
  'turboquant_compress',
  'turboquant_vector_search',
  'turboquant_cost_analyze',
  'turboquant_cache_plan',
  'turboquant_prompt_cache_lint',
  'turboquant_context_pack_build',
  'turboquant_context_pack_search',
  'turboquant_cli_mcp_profile',
  'turboquant_quantize',
  'turboquant_kv_analyze',
  'turboquant_backend_probe',
  'turboquant_opencl_probe',
  'turboquant_adreno_loader_probe',
];

function sampleArgs(toolName) {
  switch (toolName) {
    case 'turboquant_compress':
      return { vectors: [[1, 2, 3, 4], [4, 3, 2, 1]], dimensions: 4, bitsPerValue: 4 };
    case 'turboquant_vector_search':
      return { query: [1, 2, 3, 4], index: [[1, 2, 3, 4], [5, 6, 7, 8]], top_k: 1 };
    case 'turboquant_cost_analyze':
      return { usage: [{ inputTokens: 100, outputTokens: 20, cacheCreationInputTokens: 30, cacheReadInputTokens: 50, model: 'manual-test', timestamp: '2026-05-15T00:00:00Z' }] };
    case 'turboquant_cache_plan':
      return { blocks: [{ label: 'stable', text: 'Stable project context that rarely changes.' }, { label: 'volatile', text: 'Current user request changes every turn.' }], target: 'generic_mcp_context_hygiene' };
    case 'turboquant_prompt_cache_lint':
      return { blocks: [{ label: 'system', text: 'Stable instructions.' }, { label: 'task', text: 'Current task.' }] };
    case 'turboquant_context_pack_build':
      return { files: [{ path: 'README.md', text: 'TurboQuant context pack test content for conformance harness.' }], dimensions: 16, chunkBytes: 256, bitsPerValue: 4, storageMode: 'inline_text' };
    case 'turboquant_context_pack_search':
      return { manifest: null, query: 'test', top_k: 1 }; // replaced by two-step
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

function assertToolSuccess(toolName, resp) {
  if (resp.error) {
    throw new Error(`${toolName} JSON-RPC error: ${resp.error.message}`);
  }
  if (resp.result && resp.result.isError === true) {
    const text = resp.result?.content?.[0]?.text ?? '';
    throw new Error(`${toolName} returned MCP isError=true: ${text.slice(0, 200)}`);
  }
  const text = resp.result?.content?.[0]?.text;
  if (typeof text !== 'string') {
    throw new Error(`${toolName} did not return text content`);
  }
  try {
    JSON.parse(text);
  } catch (_) {
    throw new Error(`${toolName} returned non-JSON text: ${text.slice(0, 120)}`);
  }
  return text;
}

class Session {
  constructor(proc) {
    this.proc = proc;
    this.buf = '';
    this.stderr = '';
    this.transcript = [];
    this._listeners = [];

    proc.stdout.on('data', (data) => {
      this.buf += data.toString();
    });
    proc.stderr.on('data', (data) => {
      this.stderr += data.toString();
    });
  }

  send(msg) {
    const line = JSON.stringify(msg) + '\n';
    this.transcript.push({ direction: 'send', msg });
    this.proc.stdin.write(line, 'utf8');
  }

  async recv(timeoutMs = 6000) {
    return new Promise((resolve, reject) => {
      const deadline = Date.now() + timeoutMs;
      const poll = () => {
        const nl = this.buf.indexOf('\n');
        if (nl !== -1) {
          const line = this.buf.slice(0, nl).trim();
          this.buf = this.buf.slice(nl + 1);
          if (!line) { setImmediate(poll); return; }
          try {
            const msg = JSON.parse(line);
            this.transcript.push({ direction: 'recv', msg });
            resolve(msg);
          } catch (e) {
            reject(new Error(`Invalid JSON from server: ${line.slice(0, 120)}`));
          }
          return;
        }
        if (Date.now() > deadline) {
          reject(new Error(`Timeout after ${timeoutMs}ms`));
          return;
        }
        setTimeout(poll, 20);
      };
      poll();
    });
  }

  sendNotification(method, params = {}) {
    this.send({ jsonrpc: '2.0', method, params });
  }

  async call(id, method, params = {}) {
    this.send({ jsonrpc: '2.0', id, method, params });
    return this.recv();
  }
}

function pass(name, detail) {
  return { test: name, status: 'PASS', detail: detail || null };
}

function fail(name, detail) {
  return { test: name, status: 'FAIL', detail: detail || null };
}

async function runConformance() {
  const serverPath = join(rootDir, 'dist', 'server.js');
  if (!existsSync(serverPath)) {
    console.error('Server not built. Run: npm run build');
    process.exit(1);
  }

  const proc = spawn('node', [serverPath], {
    stdio: ['pipe', 'pipe', 'pipe'],
    cwd: rootDir,
  });

  proc.on('error', (err) => {
    console.error('Failed to start server:', err.message);
    process.exit(1);
  });

  const session = new Session(proc);
  const results = [];
  let toolsList = [];

  try {
    // 1. initialize
    const initResp = await session.call(1, 'initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'conformance-harness', version: '1.0.0' },
    });
    if (initResp.result && initResp.result.serverInfo) {
      results.push(pass('initialize', `serverInfo.name=${initResp.result.serverInfo.name}`));
    } else {
      results.push(fail('initialize', JSON.stringify(initResp).slice(0, 200)));
    }

    // 2. initialized notification (no response expected)
    session.sendNotification('notifications/initialized');
    results.push(pass('initialized_notification', 'sent, no response expected'));

    // 3. tools/list — must return the full shipped public tool surface
    const listResp = await session.call(2, 'tools/list');
    if (listResp.result && Array.isArray(listResp.result.tools)) {
      toolsList = listResp.result.tools;
      const names = toolsList.map((t) => t.name).sort();
      const expected = [...EXPECTED_TOOLS].sort();
      const missing = expected.filter((n) => !names.includes(n));
      const extra = names.filter((n) => !expected.includes(n));
      if (missing.length === 0 && extra.length === 0) {
        results.push(pass('tools/list', `${names.length} tools`));
      } else {
        results.push(fail('tools/list', `missing=${JSON.stringify(missing)} extra=${JSON.stringify(extra)}`));
      }
    } else {
      results.push(fail('tools/list', JSON.stringify(listResp).slice(0, 200)));
    }

    // 4-13. tools/call for direct tools — strict: isError=true counts as FAIL
    const directTools = [
      'turboquant_compress',
      'turboquant_cost_analyze',
      'turboquant_cache_plan',
      'turboquant_prompt_cache_lint',
      'turboquant_cli_mcp_profile',
      'turboquant_quantize',
      'turboquant_kv_analyze',
      'turboquant_backend_probe',
      'turboquant_opencl_probe',
      'turboquant_adreno_loader_probe',
    ];
    let callId = 10;
    for (const toolName of directTools) {
      const resp = await session.call(callId++, 'tools/call', {
        name: toolName,
        arguments: sampleArgs(toolName),
      });
      try {
        const text = assertToolSuccess(toolName, resp);
        results.push(pass(`tools/call:${toolName}`, `ok, ${text.length}b`));
      } catch (e) {
        results.push(fail(`tools/call:${toolName}`, e.message));
      }
    }

    // 10. two-step: compress → vector_search (uses compressed_database_b64 from compress output)
    const compressResp = await session.call(callId++, 'tools/call', {
      name: 'turboquant_compress',
      arguments: { vectors: [[1, 2, 3, 4], [5, 6, 7, 8]], dimensions: 4, bitsPerValue: 4 },
    });
    let compressResult = null;
    try {
      const ct = assertToolSuccess('turboquant_compress(step1)', compressResp);
      compressResult = JSON.parse(ct);
    } catch (e) {
      results.push(fail('tools/call:turboquant_vector_search', `compress step failed: ${e.message}`));
    }
    if (compressResult !== null) {
      const db64 = compressResult.compressed_database_b64 ?? compressResult.compressedDatabaseB64;
      const searchResp = await session.call(callId++, 'tools/call', {
        name: 'turboquant_vector_search',
        arguments: { compressed_database_b64: db64, query_vector: [1, 2, 3, 4], top_k: 1 },
      });
      try {
        const st = assertToolSuccess('turboquant_vector_search', searchResp);
        results.push(pass('tools/call:turboquant_vector_search', `two-step compress→search ok, ${st.length}b`));
      } catch (e) {
        results.push(fail('tools/call:turboquant_vector_search', e.message));
      }
    }

    // 11. two-step: context_pack_build → context_pack_search
    const buildResp = await session.call(callId++, 'tools/call', {
      name: 'turboquant_context_pack_build',
      arguments: sampleArgs('turboquant_context_pack_build'),
    });
    let builtManifest = null;
    try {
      const bt = assertToolSuccess('turboquant_context_pack_build', buildResp);
      const parsed = JSON.parse(bt);
      builtManifest = parsed.manifest ?? parsed;
    } catch (e) {
      results.push(fail('tools/call:turboquant_context_pack_build', e.message));
    }
    if (builtManifest !== null) {
      results.push(pass('tools/call:turboquant_context_pack_build', 'manifest built'));
      const packSearchResp = await session.call(callId++, 'tools/call', {
        name: 'turboquant_context_pack_search',
        arguments: { manifest: builtManifest, query: 'test content', top_k: 1 },
      });
      try {
        const st = assertToolSuccess('turboquant_context_pack_search', packSearchResp);
        results.push(pass('tools/call:turboquant_context_pack_search', `two-step build→search ok, ${st.length}b`));
      } catch (e) {
        results.push(fail('tools/call:turboquant_context_pack_search', e.message));
      }
    }

    // 12. invalid method → must return error
    const invalidResp = await session.call(callId++, 'invalid/method');
    if (invalidResp.error) {
      results.push(pass('invalid_method', `code=${invalidResp.error.code}`));
    } else {
      results.push(fail('invalid_method', 'expected error, got result'));
    }

    // 13. malformed JSON — server must not crash (send then verify still alive)
    proc.stdin.write('not valid json\n', 'utf8');
    await new Promise((r) => setTimeout(r, 200));
    const aliveResp = await session.call(callId++, 'tools/list');
    if (aliveResp.result?.tools) {
      results.push(pass('malformed_json_does_not_crash', 'server still responds after malformed input'));
    } else {
      results.push(fail('malformed_json_does_not_crash', 'server did not respond after malformed input'));
    }

    // 14. stdout purity — stderr must not bleed into stdout
    const stderrLines = session.stderr.split('\n').filter((l) => l.trim());
    const stdoutBleed = session.transcript
      .filter((e) => e.direction === 'recv')
      .some((e) => {
        const s = JSON.stringify(e.msg);
        return /ANTHROPIC_API_KEY|sk-ant-|ghp_|Bearer /.test(s);
      });
    if (!stdoutBleed) {
      results.push(pass('stdout_purity', `stderr lines=${stderrLines.length}, no secrets in stdout`));
    } else {
      results.push(fail('stdout_purity', 'secret pattern found in stdout'));
    }

  } catch (err) {
    results.push(fail('harness_error', err.message));
  } finally {
    proc.stdin.end();
    proc.kill();
  }

  // Print results
  console.log('\n=== MCP Conformance Results ===\n');
  for (const r of results) {
    const icon = r.status === 'PASS' ? '✓' : '✗';
    console.log(`${icon} ${r.test}: ${r.status}${r.detail ? ' — ' + r.detail : ''}`);
  }
  const passed = results.filter((r) => r.status === 'PASS').length;
  const failed = results.filter((r) => r.status === 'FAIL').length;
  console.log(`\nTotal: ${passed} passed, ${failed} failed`);

  // Write transcript
  const transcriptPath = join(rootDir, 'forensics', 'mcp-conformance-transcript.json');
  writeFileSync(transcriptPath, JSON.stringify({ results, transcript: [] }, null, 2));
  console.log('Transcript written to forensics/mcp-conformance-transcript.json');

  if (failed > 0) process.exit(1);
}

runConformance().catch((err) => {
  console.error('Fatal:', err.message);
  process.exit(1);
});
