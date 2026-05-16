import { normalizeUsageSnapshots, sanitizeUsageSnapshots } from './usage_ingest.js';
function createWasteSignal(type, severity, evidence) {
    return { type, severity, evidence };
}
export function computeCacheEfficiency(snapshots, options) {
    const invalidMode = options?.invalidMode ?? 'reject';
    let totalInput = 0;
    let totalCacheWrite = 0;
    let totalCacheRead = 0;
    let totalOutput = 0;
    // Use sanitize for clamp mode (clamps negatives), normalize for reject mode (throws on invalid)
    const normalizedSnapshots = invalidMode === 'clamp'
        ? sanitizeUsageSnapshots(snapshots)
        : normalizeUsageSnapshots(snapshots);
    for (const s of normalizedSnapshots) {
        if (invalidMode === 'reject') {
            if (s.inputTokens < 0 || s.outputTokens < 0 || s.cacheCreationInputTokens < 0 || s.cacheReadInputTokens < 0) {
                throw new Error(`Invalid token count: input=${s.inputTokens}, output=${s.outputTokens}, write=${s.cacheCreationInputTokens}, read=${s.cacheReadInputTokens}`);
            }
        }
        totalInput += s.inputTokens;
        totalCacheWrite += s.cacheCreationInputTokens;
        totalCacheRead += s.cacheReadInputTokens;
        totalOutput += s.outputTokens;
    }
    const cacheReadToWriteRatio = totalCacheWrite > 0 ? totalCacheRead / totalCacheWrite : 0;
    const cacheWriteShare = totalInput > 0 ? totalCacheWrite / totalInput : 0;
    const cacheReadShare = totalInput > 0 ? totalCacheRead / totalInput : 0;
    const wasteSignals = [];
    // Context bloat detection - only if large context with low reuse (not write-heavy)
    // Only flag as context_bloat if input is large AND cache reuse is healthy but still problematic
    // (i.e., large context that could be compressed but isn't being reused well)
    if (totalInput >= 100000 && cacheReadToWriteRatio >= 0.5 && cacheReadToWriteRatio < 1.0) {
        wasteSignals.push(createWasteSignal('context_bloat', 'high', {
            totalInputTokens: totalInput,
            threshold: 100000,
        }));
    }
    // Write heavy detection - only if cache writes dominate AND low reuse
    if (cacheWriteShare >= 0.25 && cacheReadToWriteRatio < 0.5) {
        const severity = cacheWriteShare >= 0.5 ? 'high' : cacheWriteShare >= 0.35 ? 'medium' : 'low';
        wasteSignals.push(createWasteSignal('write_heavy', severity, {
            cacheWriteShareOfInput: cacheWriteShare,
            cacheWriteTokens: totalCacheWrite,
            cacheReadTokens: totalCacheRead,
        }));
    }
    // Low cache reuse detection - only if meaningful cache activity
    if (cacheReadToWriteRatio < 0.5 && totalCacheWrite > 10000) {
        const severity = cacheReadToWriteRatio < 0.1 ? 'high' : cacheReadToWriteRatio < 0.3 ? 'medium' : 'low';
        wasteSignals.push(createWasteSignal('low_cache_reuse', severity, {
            cacheReadToWriteRatio,
            cacheWriteTokens: totalCacheWrite,
            cacheReadTokens: totalCacheRead,
        }));
    }
    // Output heavy detection - only if output dominates and context is small
    const outputRatio = totalInput > 0 ? totalOutput / totalInput : 0;
    if (outputRatio >= 0.5 && totalInput < 50000) {
        const severity = outputRatio >= 0.8 ? 'high' : outputRatio >= 0.65 ? 'medium' : 'low';
        wasteSignals.push(createWasteSignal('output_heavy', severity, {
            outputRatio,
            outputTokens: totalOutput,
            inputTokens: totalInput,
        }));
    }
    // Determine primary waste class for backward compatibility
    let estimatedWasteClass = 'unknown';
    if (totalCacheWrite === 0 && totalCacheRead === 0) {
        estimatedWasteClass = 'unknown';
    }
    else if (wasteSignals.length === 0) {
        estimatedWasteClass = 'healthy_cache_reuse';
    }
    else {
        // Sort by severity and pick the most severe
        const severityOrder = { high: 0, medium: 1, low: 2 };
        const sortedSignals = [...wasteSignals].sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);
        const primarySignal = sortedSignals[0];
        if (primarySignal && primarySignal.type === 'write_heavy') {
            estimatedWasteClass = 'write_heavy';
        }
        else if (primarySignal && primarySignal.type === 'context_bloat') {
            estimatedWasteClass = 'context_bloat';
        }
        else if (primarySignal && primarySignal.type === 'output_heavy') {
            estimatedWasteClass = 'output_heavy';
        }
        else {
            estimatedWasteClass = 'healthy_cache_reuse';
        }
    }
    return {
        totalInputTokens: totalInput,
        cacheWriteTokens: totalCacheWrite,
        cacheReadTokens: totalCacheRead,
        outputTokens: totalOutput,
        cacheReadToWriteRatio,
        cacheWriteShareOfInput: cacheWriteShare,
        cacheReadShareOfInput: cacheReadShare,
        estimatedWasteClass,
        wasteSignals,
    };
}
export function classifyWaste(metrics) {
    const recommendations = [];
    // Use wasteSignals for comprehensive recommendations
    for (const signal of metrics.wasteSignals) {
        switch (signal.type) {
            case 'write_heavy':
                recommendations.push('Cache writes dominate reads; stabilize reusable context and avoid dynamic content in early prompt prefix.');
                recommendations.push('Measure with ccusage before claiming savings.');
                break;
            case 'context_bloat':
                recommendations.push('Large context detected; consider using context_pack_build to compress local documentation.');
                recommendations.push('Use context_pack_search to retrieve only relevant chunks instead of dumping entire files.');
                break;
            case 'low_cache_reuse':
                recommendations.push('Cache reuse is low; review what content is being cached and consider stabilizing more context.');
                break;
            case 'output_heavy':
                recommendations.push('Output tokens are significant; consider reducing verbose tool outputs.');
                break;
        }
    }
    // Fallback for backward compatibility
    if (metrics.wasteSignals.length === 0) {
        switch (metrics.estimatedWasteClass) {
            case 'healthy_cache_reuse':
                recommendations.push('Cache efficiency looks healthy; continue monitoring with ccusage.');
                break;
            default:
                recommendations.push('Cache metrics unclear; ensure ccusage data is properly configured.');
        }
    }
    return recommendations;
}
