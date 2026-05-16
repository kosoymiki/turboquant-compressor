import type { UsageSnapshot, CacheEfficiencyMetrics } from './types.js';
export declare function computeCacheEfficiency(snapshots: UsageSnapshot[], options?: {
    invalidMode?: 'reject' | 'clamp';
}): CacheEfficiencyMetrics;
export declare function classifyWaste(metrics: CacheEfficiencyMetrics): string[];
