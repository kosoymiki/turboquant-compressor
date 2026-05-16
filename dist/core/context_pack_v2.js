import { z } from 'zod';
const ContextPackManifestSchema = z.custom((val) => {
    if (typeof val !== 'object' || val === null)
        return false;
    const m = val;
    return (m['version'] === 1 &&
        typeof m['createdAt'] === 'string' &&
        typeof m['rootFingerprint'] === 'string' &&
        typeof m['dimensions'] === 'number' &&
        typeof m['chunkCount'] === 'number' &&
        typeof m['compressedDatabaseB64'] === 'string' &&
        Array.isArray(m['chunks']));
}, { message: 'Invalid ContextPackManifest' });
export const ContextPackV2Schema = z.object({
    version: z.number().refine((n) => n === 2, { message: 'Must be version 2' }),
    name: z.string().min(1).max(128),
    description: z.string().max(512).optional(),
    createdAt: z.string().datetime(),
    dependencies: z.array(z.string()).optional(),
    manifests: z.array(ContextPackManifestSchema),
    metadata: z.record(z.string(), z.string()).optional(),
});
const DEFAULT_OPTIONS = {
    maxDependencies: 100,
    maxManifests: 50,
    maxTotalChunks: 10000,
};
export function validateContextPackV2(raw) {
    return ContextPackV2Schema.parse(raw);
}
export function resolveContextPack(pack, options = {}) {
    const opts = { ...DEFAULT_OPTIONS, ...options };
    const warnings = [];
    // Validate dependencies count
    if (pack.dependencies && pack.dependencies.length > opts.maxDependencies) {
        warnings.push(`Dependencies count (${pack.dependencies.length}) exceeds recommended max (${opts.maxDependencies})`);
    }
    // Validate manifests count
    if (pack.manifests.length > opts.maxManifests) {
        warnings.push(`Manifests count (${pack.manifests.length}) exceeds recommended max (${opts.maxManifests})`);
    }
    // Validate total chunks
    const totalChunks = pack.manifests.reduce((sum, m) => sum + m.chunkCount, 0);
    if (totalChunks > opts.maxTotalChunks) {
        warnings.push(`Total chunks (${totalChunks}) exceeds recommended max (${opts.maxTotalChunks})`);
    }
    // Validate storage modes
    const storageModes = new Set(pack.manifests.map((m) => m.storageMode));
    if (storageModes.size > 1) {
        warnings.push('Mixed storage modes detected; ensure caller handles all modes');
    }
    return {
        version: 2,
        name: pack.name,
        resolvedManifests: pack.manifests,
        resolvedDependencies: pack.dependencies ?? [],
        warnings,
    };
}
export function createContextPack(name, manifests, options) {
    return {
        version: 2,
        name,
        description: options?.description,
        createdAt: new Date().toISOString(),
        dependencies: options?.dependencies,
        manifests,
        metadata: options?.metadata,
    };
}
export function mergeContextPacks(packs) {
    if (packs.length === 0) {
        throw new Error('Cannot merge empty array of context packs');
    }
    const allManifests = [];
    const allDependencies = new Set();
    const allMetadata = {};
    for (const pack of packs) {
        allManifests.push(...pack.manifests);
        pack.dependencies?.forEach((d) => allDependencies.add(d));
        if (pack.metadata) {
            Object.assign(allMetadata, pack.metadata);
        }
    }
    return {
        version: 2,
        name: packs.map((p) => p.name).join('+'),
        description: `Merged from ${packs.length} packs`,
        createdAt: new Date().toISOString(),
        dependencies: Array.from(allDependencies),
        manifests: allManifests,
        metadata: allMetadata,
    };
}
