#!/usr/bin/env node

import { readFileSync, existsSync } from 'node:fs';
import { execSync } from 'node:child_process';

function fail(msg) {
  console.error(`ERROR: ${msg}`);
  process.exit(1);
}

const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
const lock = JSON.parse(readFileSync('package-lock.json', 'utf8'));
const version = pkg.version;

if (lock.version !== version) fail(`package-lock version ${lock.version} != package version ${version}`);
if (lock.packages?.['']?.version !== version) fail(`package-lock packages[""].version mismatch`);

const server = readFileSync('src/server.ts', 'utf8');
if (!server.includes(`version: '${version}'`) && !server.includes(`version: "${version}"`)) {
  fail('src/server.ts version does not match package.json');
}

const readme = readFileSync('README.md', 'utf8');
if (!readme.includes(`# TurboQuant Compressor v${version}`)) {
  fail('README heading does not match package version');
}

if (readme.includes('format_version: 2')) {
  fail('README still documents format_version: 2');
}

if (existsSync('.claude/settings.local.json')) {
  fail('.claude/settings.local.json must not be present');
}

const deps = pkg.dependencies ?? {};
const mcpVersion = deps['@modelcontextprotocol/server'] ?? '';

// @cfworker/json-schema is required by MCP SDK v2 alpha
// Allow it as a direct dependency when using MCP SDK v2
const isMcpV2Alpha = mcpVersion.includes('alpha') || mcpVersion.includes('beta');

if (Object.prototype.hasOwnProperty.call(deps, '@cfworker/json-schema') && !isMcpV2Alpha) {
  fail('package.json must not keep direct @cfworker/json-schema dependency (only allowed with MCP SDK v2 alpha/beta)');
}

const lockRootDeps = lock.packages?.['']?.dependencies ?? {};
if (Object.prototype.hasOwnProperty.call(lockRootDeps, '@cfworker/json-schema') && !isMcpV2Alpha) {
  fail('package-lock root package must not keep direct @cfworker/json-schema dependency (only allowed with MCP SDK v2 alpha/beta)');
}

const distTests = execSync(`find dist -name "*.test.js" -o -name "*.test.d.ts"`, { encoding: 'utf8' }).trim();
if (distTests.length > 0) {
  fail(`dist contains compiled tests:\n${distTests}`);
}

console.log('[OK] package.json verification passed');