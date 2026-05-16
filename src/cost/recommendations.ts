import type { CacheEfficiencyMetrics, CachePlan, ContextBlock } from './types.js';
import { classifyVolatility, recommendTier, tierReason } from './volatility.js';
import { sha256Text, approximateTokens } from './fingerprint.js';

export function recommendCostActions(
  metrics: CacheEfficiencyMetrics
): string[] {
  const actions: string[] = [];

  // Use wasteSignals for comprehensive recommendations
  for (const signal of metrics.wasteSignals) {
    switch (signal.type) {
      case 'write_heavy':
        actions.push('Stabilize reusable context blocks to reduce cache writes.');
        actions.push('Put static instructions before dynamic session outputs.');
        actions.push('Use /clear when dynamic logs pollute context.');
        break;
      case 'context_bloat':
        actions.push('Consider building a compressed context pack with turboquant_context_pack_build.');
        actions.push('Use turboquant_context_pack_search to retrieve only relevant chunks.');
        break;
      case 'low_cache_reuse':
        actions.push('Review what content is being cached and consider stabilizing more context.');
        break;
      case 'output_heavy':
        actions.push('Reduce verbose tool outputs where possible.');
        actions.push('Consider adjusting task granularity to reduce output volume.');
        break;
    }
  }

  if (metrics.wasteSignals.length === 0 && metrics.estimatedWasteClass === 'healthy_cache_reuse') {
    actions.push('Cache efficiency looks healthy; continue monitoring with ccusage.');
  }

  return actions;
}

export function buildCachePlan(
  blocks: Array<{ label: string; text: string; volatilityHint?: string }>,
  target: string
): CachePlan {
  const processedBlocks: ContextBlock[] = blocks.map((block, idx) => {
    const volatility = classifyVolatility(block.label, block.text, block.volatilityHint);
    const tier = recommendTier(volatility, Buffer.byteLength(block.text, 'utf8'));
    const fingerprint = sha256Text(block.text);
    const tokens = approximateTokens(block.text);
    const byteLength = Buffer.byteLength(block.text, 'utf8');

    return {
      id: `block_${idx}`,
      label: block.label,
      text: block.text,
      byteLength,
      approximateTokens: tokens,
      fingerprint,
      volatility,
      recommendedTier: tier,
      reason: tierReason(volatility, byteLength),
    };
  });

  const recommendations: string[] = [
    'Put static instructions before dynamic session outputs.',
    'Use /clear when dynamic logs pollute context.',
    'If using an API proxy, only then consider explicit ttl=1h cache_control.',
  ];

  // Add context pack recommendations for large stable content
  const largeStableBlocks = processedBlocks.filter(
    b => b.recommendedTier === 'context_pack_candidate'
  );
  if (largeStableBlocks.length > 0) {
    recommendations.push(
      `Large stable documents (${largeStableBlocks.length}) should be moved to compressed context pack and retrieved via context_pack_search.`
    );
  }

  const warnings: string[] = [
    'Claude Code CLI may not expose cache_control injection.',
    'This tool provides planning recommendations only; actual cache behavior depends on Claude Code internals.',
  ];

  const nonGoals: string[] = [
    'Does not reduce Anthropic API token billing directly.',
    'Does not alter Claude Code server-side prompt caching.',
    'Does not guarantee Claude Code token savings.',
  ];

  return {
    algorithm: 'COST_AWARE_BRAIN_V1',
    generatedAt: new Date().toISOString(),
    blocks: processedBlocks,
    recommendations,
    warnings,
    nonGoals,
  };
}