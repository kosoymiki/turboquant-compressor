const CACHE_BUSTING_PATTERNS = [
    {
        pattern: /\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/,
        severity: 'P1',
        problem: 'Dynamic timestamp found in reusable prefix',
        recommendation: 'Move timestamp to dynamic user message or remove it',
    },
    {
        pattern: /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/i,
        severity: 'P1',
        problem: 'UUID found in potentially reusable content',
        recommendation: 'Use stable identifiers or move to dynamic context',
    },
    {
        pattern: /session[-_ ]?log/i,
        severity: 'P1',
        problem: 'Session logs detected in stable prefix',
        recommendation: 'Keep session logs out of stable prefix to avoid cache invalidation',
    },
    {
        pattern: /\berror\b.*\bstack[-_]?trace\b/i,
        severity: 'P2',
        problem: 'Error/stack trace in potentially cached content',
        recommendation: 'Move error details to dynamic user message',
    },
    {
        pattern: /\boutput\b.*\bresult\b/i,
        severity: 'P2',
        problem: 'Tool output/results in stable prefix',
        recommendation: 'Consider if this content needs to be in stable prefix',
    },
    {
        pattern: /generated[-_]?file[-_]?list/i,
        severity: 'P2',
        problem: 'Generated file listing in stable prefix',
        recommendation: 'File listings may change; consider dynamic retrieval instead',
    },
    {
        pattern: /\benv\b.*\bdump\b/i,
        severity: 'P2',
        problem: 'Environment dump in stable prefix',
        recommendation: 'Environment variables change frequently; avoid in stable prefix',
    },
];
export function lintCacheStability(blocks) {
    const issues = [];
    let stabilityScore = 1.0;
    for (const block of blocks) {
        for (const check of CACHE_BUSTING_PATTERNS) {
            if (check.pattern.test(block.text)) {
                issues.push({
                    severity: check.severity,
                    block: block.label,
                    problem: check.problem,
                    recommendation: check.recommendation,
                });
                if (check.severity === 'P1') {
                    stabilityScore -= 0.15;
                }
                else {
                    stabilityScore -= 0.05;
                }
            }
        }
    }
    stabilityScore = Math.max(0, Math.min(1, stabilityScore));
    return {
        algorithm_level: 'COST_AWARE_BRAIN_V1',
        issues,
        cacheStabilityScore: stabilityScore,
        warnings: [
            'This tool cannot change Claude Code server-side prompt caching.',
            'Cache stability recommendations are advisory only.',
        ],
    };
}
