export function normalizeUsageSnapshots(snapshots) {
    // Only normalize shape, not values - validation happens separately
    return snapshots.map((s) => ({
        ...s,
        inputTokens: s.inputTokens,
        outputTokens: s.outputTokens,
        cacheCreationInputTokens: s.cacheCreationInputTokens,
        cacheReadInputTokens: s.cacheReadInputTokens,
    }));
}
export function sanitizeUsageSnapshots(snapshots) {
    // Clamp negative values to zero for permissive mode
    return snapshots.map((s) => ({
        ...s,
        inputTokens: Math.max(0, s.inputTokens),
        outputTokens: Math.max(0, s.outputTokens),
        cacheCreationInputTokens: Math.max(0, s.cacheCreationInputTokens),
        cacheReadInputTokens: Math.max(0, s.cacheReadInputTokens),
    }));
}
export function validateUsageSnapshot(snapshot) {
    if (!Number.isFinite(snapshot.inputTokens) || snapshot.inputTokens < 0) {
        throw new Error(`Invalid inputTokens: ${snapshot.inputTokens}`);
    }
    if (!Number.isFinite(snapshot.outputTokens) || snapshot.outputTokens < 0) {
        throw new Error(`Invalid outputTokens: ${snapshot.outputTokens}`);
    }
    if (!Number.isFinite(snapshot.cacheCreationInputTokens) ||
        snapshot.cacheCreationInputTokens < 0) {
        throw new Error(`Invalid cacheCreationInputTokens: ${snapshot.cacheCreationInputTokens}`);
    }
    if (!Number.isFinite(snapshot.cacheReadInputTokens) ||
        snapshot.cacheReadInputTokens < 0) {
        throw new Error(`Invalid cacheReadInputTokens: ${snapshot.cacheReadInputTokens}`);
    }
}
