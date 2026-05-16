const DYNAMIC_PATTERNS = [
    /\btimestamp\b/i,
    /\bdate\b/i,
    /\btime\b/i,
    /\buuid\b/i,
    /\bguid\b/i,
    /session.*log/i,
    /\boutput\b/i,
    /\bresult\b/i,
    /\berror\b/i,
    /\bstack[-_]?trace\b/i,
];
const STATIC_PATTERNS = [
    /\.md$/,
    /\.json$/,
    /\.yaml$/,
    /\.yml$/,
    /CLAUDE\.md$/i,
    /README\.md$/i,
    /ARCHITECTURE\.md$/i,
    /docs\//,
];
export function classifyVolatility(label, text, volatilityHint) {
    if (volatilityHint) {
        const hint = volatilityHint.toLowerCase();
        if (hint === 'static')
            return 'static';
        if (hint === 'low')
            return 'low';
        if (hint === 'medium')
            return 'medium';
        if (hint === 'high')
            return 'high';
    }
    const lowerLabel = label.toLowerCase();
    if (DYNAMIC_PATTERNS.some((p) => p.test(lowerLabel))) {
        return 'high';
    }
    if (STATIC_PATTERNS.some((p) => p.test(lowerLabel))) {
        return 'static';
    }
    const hasDynamicContent = DYNAMIC_PATTERNS.some((p) => p.test(text));
    if (hasDynamicContent) {
        return 'high';
    }
    if (text.length < 1000) {
        return 'low';
    }
    else if (text.length < 10000) {
        return 'medium';
    }
    else {
        return 'low';
    }
}
export function recommendTier(volatility, byteLength) {
    // Large stable content should go to context pack, not cache
    if (byteLength !== undefined && byteLength >= 10000 && volatility !== 'high') {
        return 'context_pack_candidate';
    }
    switch (volatility) {
        case 'static':
            return 'static_1h_candidate';
        case 'low':
            return 'session_5m_candidate';
        case 'medium':
            return 'session_5m_candidate';
        case 'high':
            return 'dynamic_uncached';
    }
}
export function tierReason(volatility, byteLength) {
    // Large stable content gets context_pack_candidate
    if (byteLength !== undefined && byteLength >= 10000 && volatility !== 'high') {
        return 'Large stable content; recommend using context_pack_build for compression and context_pack_search for retrieval.';
    }
    switch (volatility) {
        case 'static':
            return 'Low volatility and high reuse potential; stable prefix candidate.';
        case 'low':
            return 'Low volatility with moderate size; session-level caching appropriate.';
        case 'medium':
            return 'Moderate volatility; session-level caching recommended.';
        case 'high':
            return 'High volatility; keep out of stable prefix to avoid cache invalidation.';
    }
}
