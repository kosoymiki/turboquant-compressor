/**
 * MCP host compatibility matrix documentation.
 * Generated from client profiles.
 */

import { clientRegistry } from '../src/profiles/clients/registry.js';

const matrix = Object.entries(clientRegistry).map(([id, profile]) => ({
  Host: profile.name,
  Transport: profile.transport,
  ConfigPath: '~/.claude.json or MCP config',
  CommandFormat: `node dist/server.js`,
  EnvPassing: 'stdin/stdout JSON-RPC',
  SmokeStatus: profile.supportStatus,
  Notes: profile.warnings.join('; ') || 'No known issues',
}));

console.log('| Host | Transport | Config path | Command format | Env passing | Smoke status | Notes |');
console.log('|---|---|---|---|---|---|---|');
matrix.forEach(row => {
  console.log(`| ${row.Host} | ${row.Transport} | ${row.ConfigPath} | ${row.CommandFormat} | ${row.EnvPassing} | ${row.SmokeStatus} | ${row.Notes} |`);
});
