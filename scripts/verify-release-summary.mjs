#!/usr/bin/env node
/**
 * Verifies docs/RELEASE_SUMMARY.md meets all requirements.
 */

import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

function fail(msg) {
  console.error(`ERROR: ${msg}`);
  process.exit(1);
}

function section(title, content) {
  return `## ${title}\n\n${content}\n`;
}

function codeBlock(lang, content) {
  return `\`\`\`${lang}\n${content}\n\`\`\``;
}

const summaryPath = join(rootDir, 'docs', 'RELEASE_SUMMARY.md');

if (!existsSync(summaryPath)) {
  fail('docs/RELEASE_SUMMARY.md not found');
}

const summary = readFileSync(summaryPath, 'utf8');
const lines = summary.split('\n');

let errors = [];

// Check required sections
const requiredSections = [
  'Version',
  'Build',
  'MCP Tools',
  'Test Suite',
  'verify:release',
  'verify-host-matrix',
  'MCP Conformance',
  'MCP Transcript',
  'Smoke Tests',
  'Archives',
  'Archive Verification',
  'Overall Verdict',
];

for (const sectionName of requiredSections) {
  if (!summary.includes(`## ${sectionName}`)) {
    errors.push(`Missing section: ${sectionName}`);
  }
}

// Check Overall Verdict logic
const overallMatch = summary.match(/Overall Verdict[:\s]*(\*\*READY\*\*|\*\*NOT READY\*\*)/);
if (!overallMatch) {
  errors.push('Missing Overall Verdict');
} else {
  const verdict = overallMatch[1];

  // If READY, archives must be present
  if (verdict === '**READY**') {
    if (summary.includes('Archives not found') || summary.includes('_not found_')) {
      errors.push('Overall Verdict is READY but archives section says not found');
    }
    if (summary.includes('Archive Verification') && summary.includes('skipping verification')) {
      errors.push('Overall Verdict is READY but Archive Verification is skipped');
    }
  }
}

// Check for required content
const checks = [
  { pattern: /SHA256:/, name: 'SHA256 hashes' },
  { pattern: /verify:archives[:\s]+(\*\*PASS\*\*|PASS)/, name: 'verify:archives PASS' },
  { pattern: /verify:artifact-parity[:\s]+(\*\*PASS\*\*|PASS)/, name: 'verify:artifact-parity PASS' },
  { pattern: /mcp-conformance.*14.*passed.*0.*failed|Total:\s*14\s*passed/i, name: 'MCP conformance 14/14' },
  { pattern: /isError entries[:\s]*0/i, name: 'MCP transcript 0 isError' },
  { pattern: /Smoke Tests/, name: 'Smoke Tests section' },
  { pattern: /Test Suites?[:\s]+\d+\s+passed/i, name: 'Test count' },
];

for (const check of checks) {
  if (!check.pattern.test(summary)) {
    errors.push(`Missing or failed check: ${check.name}`);
  }
}

// Check for forbidden patterns
const forbidden = [
  { pattern: /Archives not found.*READY/i, name: 'READY with archives not found' },
  { pattern: /skipping verification.*READY/i, name: 'READY with verification skipped' },
];

for (const check of forbidden) {
  if (check.pattern.test(summary)) {
    errors.push(`Forbidden pattern: ${check.name}`);
  }
}

if (errors.length > 0) {
  console.error('RELEASE_SUMMARY.md validation failed:');
  for (const err of errors) {
    console.error(`  - ${err}`);
  }
  fail(`${errors.length} validation error(s)`);
}

console.log('[OK] RELEASE_SUMMARY.md passed all checks');