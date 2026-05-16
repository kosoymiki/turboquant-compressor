#!/usr/bin/env node
/**
 * Redact secrets from forensic output files before sharing.
 * Replaces known secret patterns with [REDACTED].
 */

import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const dir = process.argv[2];
if (!dir) {
  console.error('Usage: node scripts/redact-forensics.mjs <output-dir>');
  process.exit(1);
}

const PATTERNS = [
  // Full env var assignments
  /\b(ANTHROPIC_API_KEY|OPENAI_API_KEY|NVIDIA_API_KEY|GITHUB_TOKEN|HF_TOKEN|GOOGLE_API_KEY|HUGGINGFACE_TOKEN)\s*=\s*\S+/gi,
  // Generic *_TOKEN, *_SECRET, *_KEY patterns
  /\b\w+_(TOKEN|SECRET|KEY|PASSWORD|PASS|PWD)\s*=\s*\S+/gi,
  // Bearer tokens in headers
  /Authorization:\s*Bearer\s+\S+/gi,
  // sk- style API keys (OpenAI, Anthropic)
  /\bsk-[A-Za-z0-9_-]{20,}/g,
  // ant- style keys
  /\bant-[A-Za-z0-9_-]{20,}/g,
  // ghp_ GitHub tokens
  /\bghp_[A-Za-z0-9]{36,}/g,
  // hf_ HuggingFace tokens
  /\bhf_[A-Za-z0-9]{20,}/g,
];

let files;
try {
  files = readdirSync(dir);
} catch (e) {
  console.error(`Cannot read directory: ${dir}`);
  process.exit(1);
}

let totalRedactions = 0;

for (const file of files) {
  const path = join(dir, file);
  let content;
  try {
    content = readFileSync(path, 'utf8');
  } catch {
    continue; // skip non-text files
  }

  let redacted = content;
  let count = 0;
  for (const pattern of PATTERNS) {
    redacted = redacted.replace(pattern, (match) => {
      count++;
      // Keep the key name, replace value only
      const eqIdx = match.indexOf('=');
      if (eqIdx !== -1) {
        return match.slice(0, eqIdx + 1) + '[REDACTED]';
      }
      return '[REDACTED]';
    });
  }

  if (count > 0) {
    writeFileSync(path, redacted, 'utf8');
    console.log(`[redact] ${file}: ${count} redaction(s)`);
    totalRedactions += count;
  }
}

console.log(`[redact] total: ${totalRedactions} redaction(s) across ${files.length} files`);
