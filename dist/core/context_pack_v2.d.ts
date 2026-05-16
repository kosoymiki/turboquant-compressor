import { z } from 'zod';
import type { ContextPackManifest } from '../cost/types.js';
export declare const ContextPackV2Schema: z.ZodObject<{
    version: z.ZodNumber & z.ZodType<2, number, z.core.$ZodTypeInternals<2, number>>;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    createdAt: z.ZodString;
    dependencies: z.ZodOptional<z.ZodArray<z.ZodString>>;
    manifests: z.ZodArray<z.ZodType<ContextPackManifest, unknown, z.core.$ZodTypeInternals<ContextPackManifest, unknown>>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
}, z.core.$strip>;
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
export declare function validateContextPackV2(raw: unknown): ContextPackV2;
export declare function resolveContextPack(pack: ContextPackV2, options?: ContextPackResolverOptions): ResolvedContextPack;
export declare function createContextPack(name: string, manifests: ContextPackManifest[], options?: {
    description?: string;
    dependencies?: string[];
    metadata?: Record<string, string>;
}): ContextPackV2;
export declare function mergeContextPacks(packs: ContextPackV2[]): ContextPackV2;
