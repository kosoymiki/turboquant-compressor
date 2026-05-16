# Claude Code Cost Optimization

## What to Measure

- Total input tokens
- Output tokens
- Cache creation tokens
- Cache read tokens
- Cache read/write ratio
- Session count
- MCP tool overhead
- Subagent/team usage

## Multi-Signal Cost Diagnosis

The Cost-Aware Brain provides multi-signal waste diagnosis instead of a single classification:

### Waste Signal Types

- **write_heavy**: Cache writes dominate reads (cacheWriteShare >= 0.25 or writes > reads)
- **low_cache_reuse**: Cache reuse is low (cacheReadToWriteRatio < 0.5)
- **context_bloat**: Large context detected (totalInput >= 100000 tokens)
- **output_heavy**: Output tokens are significant (output/input >= 0.5)

### Severity Levels

- **high**: Immediate attention needed
- **medium**: Consider optimization
- **low**: Monitor and improve over time

### Example Output

```json
{
  "estimatedWasteClass": "context_bloat",
  "wasteSignals": [
    { "type": "context_bloat", "severity": "high" },
    { "type": "low_cache_reuse", "severity": "medium" },
    { "type": "write_heavy", "severity": "medium" }
  ]
}
```

## ccusage Workflow

```bash
npx ccusage@latest
npx ccusage@latest daily --json
npx ccusage@latest session --json
```

ccusage reads local Claude Code JSONL files and reports token/cost usage.

## Decision Tree

### If Cache Reads Dominate

- Reduce context size
- Keep static instructions stable

### If Cache Writes Dominate

- Stabilize prefix
- Avoid dynamic logs in stable blocks
- Split stable vs dynamic content
- Evaluate proxy/API cache_control only if request path is controllable

### If Output Dominates

- Use smaller prompts
- Reduce verbose tool outputs
- Adjust task granularity

### If Local Retrieval Dominates

- Build a compressed context pack with turboquant_context_pack_build

## Recommendations

1. **Measure First**: Use ccusage to establish a baseline before claiming any savings.

2. **Cache Hygiene**: Use turboquant_cache_plan to classify context blocks by volatility.

3. **Context Compression**: Use turboquant_context_pack_build to compress large documentation sets.

4. **Stability**: Use turboquant_prompt_cache_lint to identify cache-busting patterns.

5. **Proxy Control**: Only consider explicit cache_control if you control the API proxy path.

6. **Large Stable Content**: Move large stable documents to context packs instead of prompt cache.