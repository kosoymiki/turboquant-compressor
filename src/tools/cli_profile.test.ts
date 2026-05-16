import { describe, it, expect } from '@jest/globals';
import { getCliProfile, listSupportedHosts } from './cli_profile.js';

describe('CLI Profile', () => {
  it('returns documented status for claude_code (no live smoke evidence)', () => {
    const result = getCliProfile({ host: 'claude_code' });
    expect(result.supportStatus).toBe('documented');
  });

  it('returns verified status for generic_stdio', () => {
    const result = getCliProfile({ host: 'generic_stdio' });
    expect(result.supportStatus).toBe('verified');
  });

  it('returns documented status for codex_cli', () => {
    const result = getCliProfile({ host: 'codex_cli' });
    expect(result.supportStatus).toBe('documented');
  });

  it('returns unverified status for unknown host', () => {
    const result = getCliProfile({ host: 'unknown_host' });
    expect(result.supportStatus).toBe('unverified');
  });

  it('returns all 8 expected tools for claude_code', () => {
    const result = getCliProfile({ host: 'claude_code' });
    expect(result.expectedTools).toHaveLength(8);
    expect(result.expectedTools).toContain('turboquant_cli_mcp_profile');
  });

  it('lists all 7 supported hosts', () => {
    const hosts = listSupportedHosts();
    expect(hosts).toHaveLength(7);
    expect(hosts).toContain('claude_code');
    expect(hosts).toContain('codex_cli');
    expect(hosts).toContain('generic_stdio');
  });

  it('configSnippet for codex_cli uses TOML keyed table format', () => {
    const result = getCliProfile({ host: 'codex_cli' });
    expect(result.configSnippet).toContain('[mcp_servers."turboquant-compressor"]');
    expect(result.configSnippet).not.toContain('"mcpServers"');
  });
});
