import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import { spawn, ChildProcess } from 'child_process';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '../..');
const serverPath = join(rootDir, 'dist', 'server.js');

function sendRecv(proc: ChildProcess, msg: object, timeoutMs = 6000): Promise<object> {
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

describe('MCP Conformance', () => {
  let server: ChildProcess;

  beforeAll(async () => {
    if (!existsSync(serverPath)) throw new Error('dist/server.js not found — run npm run build');
    server = spawn('node', [serverPath], { stdio: ['pipe', 'pipe', 'pipe'], cwd: rootDir });
    // initialize handshake
    await sendRecv(server, {
      jsonrpc: '2.0', id: 1, method: 'initialize',
      params: { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'jest', version: '1.0' } },
    });
    server.stdin!.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized', params: {} }) + '\n');
  });

  afterAll(() => { server?.kill(); });

  it('responds to initialize with serverInfo', async () => {
    const s2 = spawn('node', [serverPath], { stdio: ['pipe', 'pipe', 'pipe'], cwd: rootDir });
    const resp: any = await sendRecv(s2, {
      jsonrpc: '2.0', id: 1, method: 'initialize',
      params: { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'jest', version: '1.0' } },
    });
    s2.kill();
    expect(resp.result?.serverInfo?.name).toBeDefined();
  });

  it('tools/list returns exactly 13 tools', async () => {
    const resp: any = await sendRecv(server, { jsonrpc: '2.0', id: 2, method: 'tools/list' });
    expect(Array.isArray(resp.result?.tools)).toBe(true);
    expect(resp.result.tools).toHaveLength(13);
  });

  it('tools/list includes turboquant_cli_mcp_profile', async () => {
    const resp: any = await sendRecv(server, { jsonrpc: '2.0', id: 3, method: 'tools/list' });
    const names = resp.result.tools.map((t: any) => t.name);
    expect(names).toContain('turboquant_cli_mcp_profile');
  });

  it('returns error for invalid method', async () => {
    const resp: any = await sendRecv(server, { jsonrpc: '2.0', id: 4, method: 'invalid/method' });
    expect(resp.error).toBeDefined();
  });

  it('server survives malformed JSON and still responds', async () => {
    server.stdin!.write('not valid json\n');
    await new Promise(r => setTimeout(r, 150));
    const resp: any = await sendRecv(server, { jsonrpc: '2.0', id: 5, method: 'tools/list' });
    expect(resp.result?.tools).toBeDefined();
  });
});
