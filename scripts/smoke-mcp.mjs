#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const NODE = process.execPath;
const SERVER = join(ROOT, 'dist', 'server.js');
const TIMEOUT_MS = 15000;

let stdout = '';
let stderr = '';
let server;

function fail(message, details = '') {
  console.error(`[FAIL] ${message}`);
  if (details) console.error(details);
  if (server && !server.killed) server.kill('SIGKILL');
  process.exit(1);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function send(msg) {
  server.stdin.write(`${JSON.stringify(msg)}\n`);
}

function parseMessages() {
  return stdout
    .split('\n')
    .map((x) => x.trim())
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        fail('MCP server wrote non-JSON to stdout', line);
      }
    });
}

function byId(messages, id) {
  return messages.find((m) => m.id === id);
}

async function main() {
  console.log('[MCP] Starting MCP contract smoke');

  server = spawn(NODE, [SERVER], {
    cwd: ROOT,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env, NODE_ENV: 'test' },
  });

  server.stdout.on('data', (chunk) => {
    stdout += chunk.toString();
  });

  server.stderr.on('data', (chunk) => {
    stderr += chunk.toString();
  });

  server.on('error', (err) => {
    fail('Failed to spawn MCP server', err.stack ?? String(err));
  });

  const timer = setTimeout(() => {
    fail('MCP smoke timed out', `STDOUT:\n${stdout}\nSTDERR:\n${stderr}`);
  }, TIMEOUT_MS);

  send({
    jsonrpc: '2.0',
    id: 1,
    method: 'initialize',
    params: {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'turboquant-smoke', version: '1.0.0' },
    },
  });

  await sleep(500);

  send({
    jsonrpc: '2.0',
    method: 'notifications/initialized',
    params: {},
  });

  send({
    jsonrpc: '2.0',
    id: 2,
    method: 'tools/list',
    params: {},
  });

  await sleep(500);

  send({
    jsonrpc: '2.0',
    id: 3,
    method: 'tools/call',
    params: {
      name: 'turboquant_compress',
      arguments: {
        vectors: [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
        ],
        dimensions: 4,
        seed: 42,
        bitsPerValue: 4,
        includeQJL: false,
      },
    },
  });

  await sleep(700);

  let messages = parseMessages();

  const init = byId(messages, 1);
  if (!init || init.error) {
    fail('initialize failed', JSON.stringify(init, null, 2));
  }

  const toolsList = byId(messages, 2);
  if (!toolsList || toolsList.error) {
    fail('tools/list failed', JSON.stringify(toolsList, null, 2));
  }

  const tools = toolsList.result?.tools ?? [];
  const toolNames = tools.map((t) => t.name);
  if (!toolNames.includes('turboquant_compress')) {
    fail('turboquant_compress missing from tools/list', JSON.stringify(tools, null, 2));
  }
  if (!toolNames.includes('turboquant_vector_search')) {
    fail('turboquant_vector_search missing from tools/list', JSON.stringify(tools, null, 2));
  }

  // Runtime schema verification
  const compressTool = tools.find((t) => t.name === 'turboquant_compress');
  const searchTool = tools.find((t) => t.name === 'turboquant_vector_search');

  const bitsSchema = compressTool?.inputSchema?.properties?.bitsPerValue;
  if (JSON.stringify(bitsSchema?.enum) !== JSON.stringify([2, 3, 4, 8])) {
    fail('turboquant_compress bitsPerValue schema must be enum [2,3,4,8]', JSON.stringify(bitsSchema, null, 2));
  }

  const dimSchema = compressTool?.inputSchema?.properties?.dimensions;
  if (dimSchema?.minimum !== 1 || dimSchema?.maximum !== 8192) {
    fail('turboquant_compress dimensions schema must be 1..8192', JSON.stringify(dimSchema, null, 2));
  }

  const topKSchema = searchTool?.inputSchema?.properties?.top_k;
  if (topKSchema?.minimum !== 1 || topKSchema?.maximum !== 100) {
    fail('turboquant_vector_search top_k schema must be 1..100', JSON.stringify(topKSchema, null, 2));
  }

  const compress = byId(messages, 3);
  if (!compress || compress.error) {
    fail('turboquant_compress call failed', JSON.stringify(compress, null, 2));
  }

  const compressText = compress.result?.content?.[0]?.text;
  if (!compressText) {
    fail('compress result missing content[0].text', JSON.stringify(compress, null, 2));
  }

  let compressPayload;
  try {
    compressPayload = JSON.parse(compressText);
  } catch {
    fail('compress result text is not JSON', compressText);
  }

  if (!compressPayload.compressed_database_b64) {
    fail('compress payload missing compressed_database_b64', JSON.stringify(compressPayload, null, 2));
  }

  send({
    jsonrpc: '2.0',
    id: 4,
    method: 'tools/call',
    params: {
      name: 'turboquant_vector_search',
      arguments: {
        query_vector: [1, 0, 0, 0],
        compressed_database_b64: compressPayload.compressed_database_b64,
        dimensions: 4,
        top_k: 1,
        metric: 'cosine',
      },
    },
  });

  await sleep(700);

  messages = parseMessages();

  const search = byId(messages, 4);
  if (!search || search.error) {
    fail('turboquant_vector_search call failed', JSON.stringify(search, null, 2));
  }

  const searchText = search.result?.content?.[0]?.text;
  if (!searchText) {
    fail('search result missing content[0].text', JSON.stringify(search, null, 2));
  }

  let searchPayload;
  try {
    searchPayload = JSON.parse(searchText);
  } catch {
    fail('search result text is not JSON', searchText);
  }

  if (!Array.isArray(searchPayload.results) || searchPayload.results.length !== 1) {
    fail('search payload missing results', JSON.stringify(searchPayload, null, 2));
  }

  if (searchPayload.results[0].index !== 0) {
    fail('self-nearest vector should rank index 0 first', JSON.stringify(searchPayload, null, 2));
  }

  // Test negative case: ragged vectors should fail
  send({
    jsonrpc: '2.0',
    id: 5,
    method: 'tools/call',
    params: {
      name: 'turboquant_compress',
      arguments: {
        vectors: [
          [1, 0, 0, 0],
          [1, 0],
        ],
        dimensions: 4,
        bitsPerValue: 4,
      },
    },
  });

  await sleep(700);

  messages = parseMessages();
  const badCompress = byId(messages, 5);
  const failedAsRpc = Boolean(badCompress?.error);
  const failedAsTool = Boolean(badCompress?.result?.isError);
  if (!failedAsRpc && !failedAsTool) {
    fail('ragged vector MCP call should fail', JSON.stringify(badCompress, null, 2));
  }

  // Test negative case: invalid bitsPerValue should fail
  send({
    jsonrpc: '2.0',
    id: 6,
    method: 'tools/call',
    params: {
      name: 'turboquant_compress',
      arguments: {
        vectors: [[1, 0, 0, 0]],
        dimensions: 4,
        bitsPerValue: 5,
      },
    },
  });

  await sleep(700);

  messages = parseMessages();
  const badBits = byId(messages, 6);
  const failedBadBits = Boolean(badBits?.error) || Boolean(badBits?.result?.isError);
  if (!failedBadBits) {
    fail('invalid bitsPerValue=5 MCP call should fail', JSON.stringify(badBits, null, 2));
  }

  clearTimeout(timer);
  server.kill('SIGTERM');

  console.log('[OK] MCP smoke passed');
}

main().catch((err) => {
  fail('MCP smoke crashed', err?.stack ?? String(err));
});