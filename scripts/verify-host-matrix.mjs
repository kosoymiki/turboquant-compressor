#!/usr/bin/env node
/**
 * Gate: host matrix status consistency check.
 * Verifies registry, docs/MCP_HOST_MATRIX.md, docs/clients/<HOST>.md, and server schema
 * all agree on the same supportStatus for every host.
 */

import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

function fail(msg) {
  console.error(`ERROR [verify-host-matrix]: ${msg}`);
  process.exit(1);
}

// Ground truth: expected status for each host
const EXPECTED = {
  generic_stdio: 'verified',
  claude_code: 'documented',
  codex_cli: 'documented',
  cursor: 'documented',
  gemini_cli: 'documented',
  opencode: 'documented',
  crush: 'documented',
};

const HOST_DOC_MAP = {
  generic_stdio: 'GENERIC_STDIO.md',
  claude_code: 'CLAUDE_CODE.md',
  codex_cli: 'CODEX_CLI.md',
  cursor: 'CURSOR.md',
  gemini_cli: 'GEMINI_CLI.md',
  opencode: 'OPENCODE.md',
  crush: 'CRUSH.md',
};

// 1. Read registry.ts and extract id → supportStatus
const registryPath = join(rootDir, 'src', 'profiles', 'clients', 'registry.ts');
if (!existsSync(registryPath)) fail('src/profiles/clients/registry.ts not found');
const registrySrc = readFileSync(registryPath, 'utf8');

function extractRegistryStatus(host) {
  // Find the profile block for this host id and extract supportStatus
  const idPattern = new RegExp(`id:\\s*['"]${host}['"]`);
  const idMatch = idPattern.exec(registrySrc);
  if (!idMatch) return null;
  // Look for supportStatus within ~2000 chars after the id (long configSnippets need more)
  const slice = registrySrc.slice(idMatch.index, idMatch.index + 2000);
  const statusMatch = /supportStatus:\s*['"](\w+)['"]/.exec(slice);
  return statusMatch ? statusMatch[1] : null;
}

// 2. Read MCP_HOST_MATRIX.md and extract host → status from table
const matrixPath = join(rootDir, 'docs', 'MCP_HOST_MATRIX.md');
if (!existsSync(matrixPath)) fail('docs/MCP_HOST_MATRIX.md not found');
const matrixSrc = readFileSync(matrixPath, 'utf8');

function extractMatrixStatus(host) {
  // Match table row: | ... | `host_id` | ... | status |
  const pattern = new RegExp(`\\|[^|]*\\|\\s*\`${host}\`\\s*\\|[^|]*\\|\\s*(\\w+)\\s*\\|`);
  const m = pattern.exec(matrixSrc);
  return m ? m[1] : null;
}

// 3. Read docs/clients/<HOST>.md and check it contains the expected status
const docsClientsDir = join(rootDir, 'docs', 'clients');

function extractClientDocStatus(host) {
  const docFile = HOST_DOC_MAP[host];
  const docPath = join(docsClientsDir, docFile);
  if (!existsSync(docPath)) return null;
  const src = readFileSync(docPath, 'utf8').toLowerCase();
  if (src.includes('status: verified') || src.includes('`verified`')) return 'verified';
  if (src.includes('status: documented') || src.includes('`documented`')) return 'documented';
  if (src.includes('status: unverified') || src.includes('`unverified`')) return 'unverified';
  return null;
}

// 4. Check server.ts enum
const serverPath = join(rootDir, 'src', 'server.ts');
if (!existsSync(serverPath)) fail('src/server.ts not found');
const serverSrc = readFileSync(serverPath, 'utf8');

// 5. Snippet shape checks
const snippetRules = {
  codex_cli: {
    mustContain: ['[mcp_servers."turboquant-compressor"]'],
    mustNotContain: ['[mcp_servers.turboquant-compressor]'],
  },
  opencode: {
    mustContain: ['"mcp"'],
    mustNotContain: ['"mcpServers"'],
  },
  gemini_cli: {
    mustContain: ['"mcpServers"'],
    mustNotContain: [],
  },
  crush: {
    mustContain: [],
    mustNotContain: ['"mcpServers"'], // unless verified with live transcript
  },
};

function checkSnippet(host, configSnippet) {
  const rules = snippetRules[host];
  if (!rules) return true;

  for (const pattern of rules.mustContain) {
    if (!configSnippet.includes(pattern)) {
      console.error(`  FAIL snippet: ${host} must contain "${pattern}"`);
      return false;
    }
  }
  for (const pattern of rules.mustNotContain) {
    if (configSnippet.includes(pattern)) {
      console.error(`  FAIL snippet: ${host} must not contain "${pattern}"`);
      return false;
    }
  }
  return true;
}

let failures = 0;

for (const [host, expectedStatus] of Object.entries(EXPECTED)) {
  // Registry check
  const regStatus = extractRegistryStatus(host);
  if (!regStatus) {
    console.error(`  FAIL registry: host ${host} not found`);
    failures++;
  } else if (regStatus !== expectedStatus) {
    console.error(`  FAIL registry: ${host} status=${regStatus}, expected=${expectedStatus}`);
    failures++;
  }

  // Matrix check
  const matStatus = extractMatrixStatus(host);
  if (!matStatus) {
    console.error(`  FAIL matrix: host ${host} not found in MCP_HOST_MATRIX.md table`);
    failures++;
  } else if (matStatus !== expectedStatus) {
    console.error(`  FAIL matrix: ${host} status=${matStatus}, expected=${expectedStatus}`);
    failures++;
  }

  // Client doc check
  const docFile = HOST_DOC_MAP[host];
  const docPath = join(docsClientsDir, docFile);
  if (!existsSync(docPath)) {
    console.error(`  FAIL docs: docs/clients/${docFile} not found for host ${host}`);
    failures++;
  } else {
    const docStatus = extractClientDocStatus(host);
    if (!docStatus) {
      console.error(`  FAIL docs: docs/clients/${docFile} has no recognizable Status line`);
      failures++;
    } else if (docStatus !== expectedStatus) {
      console.error(`  FAIL docs: ${docFile} status=${docStatus}, expected=${expectedStatus}`);
      failures++;
    }
  }

  // Server enum check
  if (!serverSrc.includes(host)) {
    console.error(`  FAIL server: src/server.ts schema enum does not include host: ${host}`);
    failures++;
  }

  // Snippet shape check
  const idPattern = new RegExp(`id:\\s*['"]${host}['"]`);
  const idMatch = idPattern.exec(registrySrc);
  if (idMatch) {
    const slice = registrySrc.slice(idMatch.index, idMatch.index + 2000);
    const snippetMatch = slice.match(/configSnippet:\s*`([^`]+)`/);
    if (snippetMatch) {
      if (!checkSnippet(host, snippetMatch[1])) {
        failures++;
      }
    }
  }
}

// Extra: no host should be marked verified in docs unless it's in EXPECTED as verified
const verifiedHosts = Object.entries(EXPECTED).filter(([, s]) => s === 'verified').map(([h]) => h);
const documentedHosts = Object.entries(EXPECTED).filter(([, s]) => s === 'documented').map(([h]) => h);

if (failures > 0) fail(`${failures} host matrix consistency failure(s)`);

console.log(
  `[OK] verify-host-matrix: ${verifiedHosts.length} verified (${verifiedHosts.join(', ')}), ` +
  `${documentedHosts.length} documented, 0 unverified`
);
