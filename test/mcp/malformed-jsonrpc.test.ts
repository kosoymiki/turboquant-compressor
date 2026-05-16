import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import { spawn, ChildProcess } from 'child_process';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '../..');
const serverPath = join(rootDir, 'dist', 'server.js');

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

describe('MCP Malformed Input', () => {
  let server: ChildProcess;

  beforeAll(async () => {
    if (!existsSync(serverPath)) throw new Error('dist/server.js not found — run npm run build');
    server = spawn('node', [serverPath], { stdio: ['pipe', 'pipe', 'pipe'], cwd: rootDir });
    await sendRecv(server, {
      jsonrpc: '2.0', id: 1, method: 'initialize',
      params: { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'jest', version: '1.0' } },
    });
    server.stdin!.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized', params: {} }) + '\n');
  });

  afterAll(() => { server?.kill(); });

  it('server survives incomplete JSON and still responds', async () => {
    server.stdin!.write('{incomplete\n');
    await new Promise(r => setTimeout(r, 150));
    const resp: any = await sendRecv(server, { jsonrpc: '2.0', id: 10, method: 'tools/list' });
    expect(resp.result?.tools).toBeDefined();
  });

  it('server survives message with no method and still responds', async () => {
    server.stdin!.write(JSON.stringify({ jsonrpc: '2.0', id: 11 }) + '\n');
    await new Promise(r => setTimeout(r, 150));
    const resp: any = await sendRecv(server, { jsonrpc: '2.0', id: 20, method: 'tools/list' });
    expect(resp.result?.tools).toBeDefined();
  });

  it('returns error for unknown method', async () => {
    const resp: any = await sendRecv(server, { jsonrpc: '2.0', id: 12, method: 'no/such/method' });
    expect(resp.error).toBeDefined();
  });

  it('server survives null params and still responds', async () => {
    server.stdin!.write(JSON.stringify({ jsonrpc: '2.0', id: 13, method: 'tools/list', params: null }) + '\n');
    await new Promise(r => setTimeout(r, 150));
    const resp: any = await sendRecv(server, { jsonrpc: '2.0', id: 14, method: 'tools/list' });
    expect(resp.result?.tools).toBeDefined();
  });
});
