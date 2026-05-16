import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import { spawn, ChildProcess } from 'child_process';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '../..');
const serverPath = join(rootDir, 'dist', 'server.js');

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

function sendRecv(proc: ChildProcess, msg: object, timeoutMs = 5000): Promise<object> {
  return new Promise((resolve, reject) => {
    let buf = '';
    const deadline = setTimeout(() => reject(new Error('timeout')), timeoutMs);
    const onData = (data: Buffer) => {
      buf += data.toString();
      const nl = buf.indexOf('\n');
      if (nl !== -1) {
        clearTimeout(deadline);
        proc.stdout!.off('data', onData);
        try { resolve(JSON.parse(buf.slice(0, nl))); } catch (e) { reject(e); }
      }
    };
    proc.stdout!.on('data', onData);
    proc.stdin!.write(JSON.stringify(msg) + '\n');
  });
}

describe('MCP Tool Schema Snapshots', () => {
  let server: ChildProcess;
  let tools: any[];

  beforeAll(async () => {
    if (!existsSync(serverPath)) throw new Error('dist/server.js not found — run npm run build');
    server = spawn('node', [serverPath], { stdio: ['pipe', 'pipe', 'pipe'], cwd: rootDir });
    await sendRecv(server, {
      jsonrpc: '2.0', id: 1, method: 'initialize',
      params: { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'jest', version: '1.0' } },
    });
    server.stdin!.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized', params: {} }) + '\n');
    const resp: any = await sendRecv(server, { jsonrpc: '2.0', id: 2, method: 'tools/list' });
    tools = resp.result?.tools ?? [];
  });

  afterAll(() => { server?.kill(); });

  it('returns exactly 13 tools', () => {
    expect(tools).toHaveLength(13);
  });

  it('all tool names match expected set', () => {
    const names = tools.map((t: any) => t.name).sort();
    expect(names).toEqual([...EXPECTED_TOOLS].sort());
  });

  it('every tool has name, description, and inputSchema', () => {
    for (const tool of tools) {
      expect(typeof tool.name).toBe('string');
      expect(typeof tool.description).toBe('string');
      expect(tool.inputSchema).toBeDefined();
      expect(tool.inputSchema.type).toBe('object');
    }
  });

  it('all tool names are prefixed turboquant_', () => {
    for (const tool of tools) {
      expect(tool.name).toMatch(/^turboquant_/);
    }
  });

  it('turboquant_context_pack_search has externalStore in inputSchema', () => {
    const searchTool = tools.find((t: any) => t.name === 'turboquant_context_pack_search');
    expect(searchTool).toBeDefined();
    expect(searchTool.inputSchema.properties.externalStore).toBeDefined();
    expect(searchTool.inputSchema.properties.externalStore.properties.kind.enum).toContain('inline_entries');
    expect(searchTool.inputSchema.properties.externalStore.properties.entries.maxItems).toBe(1000);
    expect(searchTool.inputSchema.properties.externalStore.properties.entries.items.properties.text.maxLength).toBe(1000000);
    expect(searchTool.inputSchema.properties.externalStore.required).toContain('kind');
    expect(searchTool.inputSchema.properties.externalStore.required).toContain('entries');
  });

  it('turboquant_context_pack_search has required manifest and query', () => {
    const searchTool = tools.find((t: any) => t.name === 'turboquant_context_pack_search');
    expect(searchTool).toBeDefined();
    expect(searchTool.inputSchema.required).toContain('manifest');
    expect(searchTool.inputSchema.required).toContain('query');
  });
});
