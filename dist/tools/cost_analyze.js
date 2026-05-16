import { z } from 'zod';
import { validateUsageSnapshot, normalizeUsageSnapshots } from '../cost/usage_ingest.js';
import { computeCacheEfficiency, classifyWaste } from '../cost/cache_math.js';
import { validateCostSnapshots } from '../cost/limits.js';
const CostAnalyzeInputSchema = z.object({
    usage: z.array(z.object({
        inputTokens: z.number().min(0),
        outputTokens: z.number().min(0),
        cacheCreationInputTokens: z.number().min(0),
        cacheReadInputTokens: z.number().min(0),
        totalCostUsdEstimate: z.number().optional(),
        sessionId: z.string().optional(),
        model: z.string().optional(),
        timestamp: z.string().optional(),
    })).min(1).max(10000),
});
export function turboquantCostAnalyze(rawInput) {
    const input = CostAnalyzeInputSchema.parse(rawInput);
    // Validate limits
    validateCostSnapshots(input.usage);
    const snapshots = input.usage.map((u) => ({
        ...u,
        source: 'manual',
    }));
    // Validate each snapshot
    for (const snapshot of snapshots) {
        validateUsageSnapshot(snapshot);
    }
    const normalizedSnapshots = normalizeUsageSnapshots(snapshots);
    const metrics = computeCacheEfficiency(normalizedSnapshots);
    const recommendations = classifyWaste(metrics);
    return {
        algorithm_level: 'COST_AWARE_BRAIN_V1',
        metrics,
        recommendations,
        warnings: [
            'This tool cannot change Claude Code server-side prompt caching.',
            'Cache metrics are advisory; actual billing depends on Anthropic API.',
            'Measure with ccusage for accurate baseline.',
        ],
        non_goals: [
            'Does not reduce Anthropic API token billing directly.',
            'Does not alter Claude Code server-side prompt caching.',
        ],
    };
}
