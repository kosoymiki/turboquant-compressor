import type { CacheEfficiencyMetrics, CachePlan } from './types.js';
export declare function recommendCostActions(metrics: CacheEfficiencyMetrics): string[];
export declare function buildCachePlan(blocks: Array<{
    label: string;
    text: string;
    volatilityHint?: string;
}>, target: string): CachePlan;
