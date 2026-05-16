/**
 * ContextStoreResolver interface and built-in implementations.
 *
 * Resolves chunk text from a storage backend given a chunk_id and fingerprint.
 * The external_store mode stores no text in the manifest — callers must supply
 * a resolver that fetches text from their own store.
 */
export type ResolveSource = 'inline' | 'preview' | 'external' | 'missing';
export interface ResolveResult {
    text: string;
    source: ResolveSource;
    verified: boolean;
    warning?: string;
}
export interface ContextStoreResolver {
    resolve(chunkId: string, fingerprint: string): Promise<ResolveResult>;
}
/**
 * Inline resolver: text is already embedded in the manifest chunk.
 * Returns it directly without any external call.
 */
export declare class InlineResolver implements ContextStoreResolver {
    private readonly chunks;
    constructor(chunks: Array<{
        id: string;
        text: string;
        fingerprint: string;
    }>);
    resolve(chunkId: string, fingerprint: string): Promise<ResolveResult>;
}
/**
 * Preview resolver: only a text preview is available (first 200 chars).
 * Returns preview with source='preview' and verified=false.
 */
export declare class PreviewResolver implements ContextStoreResolver {
    private readonly previews;
    constructor(chunks: Array<{
        id: string;
        textPreview?: string;
    }>);
    resolve(chunkId: string, _fingerprint: string): Promise<ResolveResult>;
}
/**
 * Null resolver for external_store mode.
 * Returns stable chunk_id with no text. Caller must supply their own resolver.
 */
export declare class ExternalStoreResolver implements ContextStoreResolver {
    resolve(chunkId: string, _fingerprint: string): Promise<ResolveResult>;
}
