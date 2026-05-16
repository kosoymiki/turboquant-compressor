#!/usr/bin/env node

import { readFileSync, existsSync, writeFileSync } from 'node:fs';
import { normalizeUsageSnapshots, validateUsageSnapshot } from '../dist/cost/usage_ingest.js';
import { computeCacheEfficiency, classifyWaste } from '../dist/cost/cache_math.js';
import { lintCacheStability } from '../dist/cost/context_lint.js';
import { buildCachePlan } from '../dist/cost/recommendations.js';

const FIXTURE_PATH = 'scripts/fixtures/claude-usage-sample.jsonl';

function assertFinite(name, value) {
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite, got ${value}`);
  }
}

function assertInRange(name, value, min, max) {
  if (value < min || value > max) {
    throw new Error(`${name} must be in [${min}, ${max}], got ${value}`);
  }
}

let usageData = [];

if (existsSync(FIXTURE_PATH)) {
  const lines = readFileSync(FIXTURE_PATH, 'utf8').split('\n').filter(Boolean);
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      if (parsed.input_tokens !== undefined) {
        usageData.push({
          inputTokens: parsed.input_tokens,
          outputTokens: parsed.output_tokens,
          cacheCreationInputTokens: parsed.cache_creation_input_tokens || 0,
          cacheReadInputTokens: parsed.cache_read_input_tokens || 0,
          totalCostUsdEstimate: parsed.total_cost_usd_estimate,
          sessionId: parsed.session_id,
          model: parsed.model,
          timestamp: parsed.timestamp,
        });
      }
    } catch (e) {
      // Skip invalid lines
    }
  }
}

if (usageData.length === 0) {
  usageData = [
    {
      inputTokens: 100000,
      outputTokens: 20000,
      cacheCreationInputTokens: 40000,
      cacheReadInputTokens: 10000,
      totalCostUsdEstimate: 2.5,
      sessionId: 'test-session-001',
      model: 'claude-sonnet-4-20250514',
    },
  ];
}

const snapshots = normalizeUsageSnapshots(
  usageData.map((u) => ({ ...u, source: 'manual' }))
);

for (const snapshot of snapshots) {
  validateUsageSnapshot(snapshot);
}

const metrics = computeCacheEfficiency(snapshots);
const recommendations = classifyWaste(metrics);

const lintResult = lintCacheStability([
  { label: 'CLAUDE.md', text: '# CLAUDE.md\n\nThis is a stable instruction file.' },
  { label: 'session.log', text: '2024-01-15T10:30:00Z Session started\nError: something failed' },
]);

const planResult = buildCachePlan([
  { label: 'CLAUDE.md', text: '# CLAUDE.md\n\nThis is a stable instruction file.', volatilityHint: 'low' },
  { label: 'docs/ARCHITECTURE.md', text: '# Architecture\n\nThis document describes the system architecture.', volatilityHint: 'low' },
  { label: 'session.log', text: '2024-01-15T10:30:00Z Session started\nError: something failed', volatilityHint: 'high' },
], 'claude_code_context_hygiene');

const report = {
  test_name: 'turboquant_open_test_cost',
  algorithm_level: 'COST_AWARE_BRAIN_V1',
  usage_snapshot_count: snapshots.length,
  cache_metrics: metrics,
  recommendations,
  cache_lint: {
    issues: lintResult.issues,
    cache_stability_score: lintResult.cacheStabilityScore,
  },
  cache_plan: {
    block_count: planResult.blocks.length,
    recommendations: planResult.recommendations,
  },
  warnings: [
    'This is a deterministic test; real usage may vary.',
    'Measure with ccusage for accurate baseline.',
  ],
};

// Hard assertions
if (report.usage_snapshot_count <= 0) {
  throw new Error('usage_snapshot_count must be positive');
}
assertFinite('cache_read_to_write_ratio', report.cache_metrics.cacheReadToWriteRatio);
assertInRange('cache_stability_score', report.cache_lint.cache_stability_score, 0, 1);
assertFinite('total_input_tokens', report.cache_metrics.totalInputTokens);
assertFinite('cache_write_tokens', report.cache_metrics.cacheWriteTokens);
assertFinite('cache_read_tokens', report.cache_metrics.cacheReadTokens);

const outputPath = process.env.TQ_OPEN_TEST_OUTPUT;
const json = JSON.stringify(report, null, 2);

if (outputPath) {
  writeFileSync(outputPath, `${json}\n`, 'utf8');
}

console.log(json);