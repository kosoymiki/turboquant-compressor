import { z } from 'zod';
import type { ContextPackManifest } from '../cost/types.js';

const ContextPackManifestSchema: z.ZodType<ContextPackManifest> = z.custom<ContextPackManifest>(
  (val) => {
    if (typeof val !== 'object' || val === null) return false;
    const m = val as Record<string, unknown>;
    return (
      m['version'] === 1 &&
      typeof m['createdAt'] === 'string' &&
      typeof m['rootFingerprint'] === 'string' &&
      typeof m['dimensions'] === 'number' &&
      typeof m['chunkCount'] === 'number' &&
      typeof m['compressedDatabaseB64'] === 'string' &&
      Array.isArray(m['chunks'])
    );
  },
  { message: 'Invalid ContextPackManifest' }
);

export const ContextPackV2Schema = z.object({
  version: z.number().refine((n) => n === 2, { message: 'Must be version 2' }),
  name: z.string().min(1).max(128),
  description: z.string().max(512).optional(),
  createdAt: z.string().datetime(),
  dependencies: z.array(z.string()).optional(),
  manifests: z.array(ContextPackManifestSchema),
  metadata: z.record(z.string(), z.string()).optional(),
});

export type ContextPackV2 = z.infer<typeof ContextPackV2Schema>;

export interface ResolvedContextPack {
  version: 2;
  name: string;
  resolvedManifests: ContextPackManifest[];
  resolvedDependencies: string[];
  warnings: string[];
}

export interface ContextPackResolverOptions {
  maxDependencies?: number;
  maxManifests?: number;
  maxTotalChunks?: number;
}

const DEFAULT_OPTIONS: Required<ContextPackResolverOptions> = {
  maxDependencies: 100,
  maxManifests: 50,
  maxTotalChunks: 10000,
};

export function validateContextPackV2(raw: unknown): ContextPackV2 {
  return ContextPackV2Schema.parse(raw);
}

export function resolveContextPack(
  pack: ContextPackV2,
  options: ContextPackResolverOptions = {}
): ResolvedContextPack {
  const opts: Required<ContextPackResolverOptions> = { ...DEFAULT_OPTIONS, ...options };
  const warnings: string[] = [];

  // Validate dependencies count
  if (pack.dependencies && pack.dependencies.length > opts.maxDependencies) {
    warnings.push(
      `Dependencies count (${pack.dependencies.length}) exceeds recommended max (${opts.maxDependencies})`
    );
  }

  // Validate manifests count
  if (pack.manifests.length > opts.maxManifests) {
    warnings.push(
      `Manifests count (${pack.manifests.length}) exceeds recommended max (${opts.maxManifests})`
    );
  }

  // Validate total chunks
  const totalChunks = pack.manifests.reduce((sum: number, m: ContextPackManifest) => sum + m.chunkCount, 0);
  if (totalChunks > opts.maxTotalChunks) {
    warnings.push(
      `Total chunks (${totalChunks}) exceeds recommended max (${opts.maxTotalChunks})`
    );
  }

  // Validate storage modes
  const storageModes = new Set(pack.manifests.map((m: ContextPackManifest) => m.storageMode));
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

export function createContextPack(
  name: string,
  manifests: ContextPackManifest[],
  options?: {
    description?: string;
    dependencies?: string[];
    metadata?: Record<string, string>;
  }
): ContextPackV2 {
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

export function mergeContextPacks(
  packs: ContextPackV2[]
): ContextPackV2 {
  if (packs.length === 0) {
    throw new Error('Cannot merge empty array of context packs');
  }

  const allManifests: ContextPackManifest[] = [];
  const allDependencies = new Set<string>();
  const allMetadata: Record<string, string> = {};

  for (const pack of packs) {
    allManifests.push(...pack.manifests);
    pack.dependencies?.forEach((d: string) => allDependencies.add(d));
    if (pack.metadata) {
      Object.assign(allMetadata, pack.metadata);
    }
  }

  return {
    version: 2,
    name: packs.map((p: ContextPackV2) => p.name).join('+'),
    description: `Merged from ${packs.length} packs`,
    createdAt: new Date().toISOString(),
    dependencies: Array.from(allDependencies),
    manifests: allManifests,
    metadata: allMetadata,
  };
}
