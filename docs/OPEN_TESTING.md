# Open Testing Protocol

## Build

```bash
npm run release:all
npm run open:test:local
npm run open:test:local:save
npm run open:test:cost
```

For public test reports, use `open:test:local:save` to persist results in `bench/results/`.


## Cost-Aware Brain Testing

```bash
npm run open:test:cost
```

This runs the cost analysis test with multi-signal waste diagnosis.

## Context Pack Testing

Build and search a context pack:

```bash
# Build a context pack from files
node -e "
import('./dist/index.js').then(m => {
  const result = m.turboquantContextPackBuild({
    files: [
      { path: 'docs/FORMAT.md', text: '...' },
      { path: 'README.md', text: '...' }
    ],
    dimensions: 384,
    storageMode: 'inline_text'
  });
  console.log(JSON.stringify(result.manifest, null, 2));
});
"

# Search the context pack
node -e "
import('./dist/index.js').then(m => {
  const result = m.turboquantContextPackSearch({
    manifest: { /* manifest from build */ },
    query: 'binary format documentation',
    top_k: 5
  });
  console.log(JSON.stringify(result, null, 2));
});
"
```

## Metrics Explained

### Ranking Metrics

- **recall_at_1**: Fraction of queries where the approximate search returns the exact top-1 result as the correct answer.
- **recall_at_5**: Fraction of queries where the exact top-1 result appears within the approximate top-5 results.
- **mrr (Mean Reciprocal Rank)**: Average of 1/rank for the exact top-1 result. MRR = 1.0 means perfect ranking; MRR = 0.2 means the correct answer is typically at rank 5.

### Score Calibration Metrics

- **score_mae_on_exact_topk**: Mean Absolute Error between approximate and exact scores for the exact top-5 candidates (matched by index). Penalizes when approximate search misses a candidate from the exact top-k.
- **score_mse_on_exact_topk**: Mean Squared Error for exact top-k candidates (more sensitive to large errors).
- **score_p95_abs_error_on_exact_topk**: 95th percentile absolute error for exact top-k candidates.
- **score_mae_on_union_topk**: MAE across all candidates seen by either exact or approximate search (union of top-k sets).

### Cost Waste Signals

- **write_heavy**: Cache writes dominate reads
- **low_cache_reuse**: Cache reuse is low
- **context_bloat**: Large context detected
- **output_heavy**: Output tokens are significant

### Context Pack Storage Modes

- **inline_text**: Full chunk text embedded in manifest (up to maxInlineChunkBytes)
- **preview_only**: Only text previews (200 chars) embedded
- **external_store**: No text embedded; caller must resolve from external store

### Cache Tiers

- **static_1h_candidate**: Stable, reusable content for 1-hour cache
- **session_5m_candidate**: Session-level content for 5-minute cache
- **context_pack_candidate**: Large stable content for local compression
- **dynamic_uncached**: Volatile content that should not be cached

### Track A: Deterministic Token-Hash (Current)

The open test uses a deterministic token-hash vectorization:
- Each token maps to a dimension via hash32
- Sign is determined by hash parity
- Vectors are L2-normalized
- This is NOT semantic embedding quality—it's a reproducible baseline for compression testing.

### Track B: Anthropic Count Tokens API (Future)

Will use Anthropic's token counting API for more accurate token estimates.

### Track C: Real Embeddings (Future)

Will use actual embedding models (e.g., OpenAI, Anthropic, or local models) for semantic similarity testing.

## Project MCP config

Create `.mcp.json`:

```json
{
  "mcpServers": {
    "turboquant-compressor": {
      "command": "node",
      "args": ["/absolute/path/to/turboquant-compressor/dist/server.js"],
      "env": {
        "NODE_ENV": "production"
      }
    }
  }
}
```

## Verify Claude Code sees it

```bash
claude mcp list
claude mcp get turboquant-compressor
```

## Manual tool request

Ask Claude Code:

```
Use turboquant_compress on vectors [[1,0,0,0],[0,1,0,0]] with dimensions=4 and bitsPerValue=4.
Then call turboquant_vector_search with query [1,0,0,0], top_k=1, metric=cosine.
Return the raw tool JSON only.
```

## Cost-Aware Brain Manual Test

Ask Claude Code:

```
Use turboquant_cache_plan on:
1. CLAUDE.md stable project instructions
2. docs/ARCHITECTURE.md with 40k characters of stable docs
3. session log with timestamps and stack traces

Return recommended tiers and explain what should go into stable prefix, what should become context_pack_candidate, and what should stay dynamic_uncached.
```

Expected:
- CLAUDE.md -> static_1h_candidate
- large ARCHITECTURE.md -> context_pack_candidate
- session logs -> dynamic_uncached
- warning about no direct Claude Code TTL control

## Acceptance

- MCP server appears in `claude mcp list`
- compress returns `compressed_database_b64`
- `format_version = 3`
- `include_qjl = false`
- `algorithm_level = LEVEL_0_TURBOQUANT_INSPIRED_MVP`
- search returns `index: 0`
- no stdout protocol corruption
- diagnostics go to stderr only
- cost_analyze shows multi-signal waste diagnosis
- context_pack_search returns non-empty text or explicit text_unavailable warning

## Failure Report Template

If testing fails, please report:

**Device Info:**
- Device model: [e.g., Pixel 7]
- Android version: [e.g., 14]
- Termux version: [e.g., 0.118]
- Node version: [e.g., v20.10.0]

**Archive Info:**
- Archive type: [source/portable]
- Archive path: [/path/to/turboquant-compressor-4.1.2-*.tar.gz]

**Command Run:**
```bash
[exact command that failed]
```

**Output:**
```
[stdout from command]
```

**Error:**
```
[stderr from command]
```

**Open Test Local Output:**
```json
[paste npm run open:test:local output]
```

**Open Test Cost Output:**
```json
[paste npm run open:test:cost output]
```

**MCP Config (redacted):**
```json
[.mcp.json with API keys removed]
```

Submit this report to help us reproduce and fix the issue.
