/**
 * ContextStoreResolver interface and built-in implementations.
 *
 * Resolves chunk text from a storage backend given a chunk_id and fingerprint.
 * The external_store mode stores no text in the manifest — callers must supply
 * a resolver that fetches text from their own store.
 */
/**
 * Inline resolver: text is already embedded in the manifest chunk.
 * Returns it directly without any external call.
 */
export class InlineResolver {
    chunks;
    constructor(chunks) {
        this.chunks = new Map(chunks.map((c) => [c.id, { text: c.text, fingerprint: c.fingerprint }]));
    }
    async resolve(chunkId, fingerprint) {
        const entry = this.chunks.get(chunkId);
        if (!entry) {
            return { text: '', source: 'missing', verified: false, warning: `chunk_id ${chunkId} not found in inline store` };
        }
        const verified = entry.fingerprint === fingerprint;
        return {
            text: entry.text,
            source: 'inline',
            verified,
            warning: verified ? undefined : `fingerprint mismatch for ${chunkId}`,
        };
    }
}
/**
 * Preview resolver: only a text preview is available (first 200 chars).
 * Returns preview with source='preview' and verified=false.
 */
export class PreviewResolver {
    previews;
    constructor(chunks) {
        this.previews = new Map(chunks.map((c) => [c.id, c.textPreview ?? '']));
    }
    async resolve(chunkId, _fingerprint) {
        const preview = this.previews.get(chunkId);
        if (preview === undefined) {
            return { text: '', source: 'missing', verified: false, warning: `chunk_id ${chunkId} not found` };
        }
        return { text: preview, source: 'preview', verified: false, warning: 'preview only — full text not available' };
    }
}
/**
 * Null resolver for external_store mode.
 * Returns stable chunk_id with no text. Caller must supply their own resolver.
 */
export class ExternalStoreResolver {
    async resolve(chunkId, _fingerprint) {
        return {
            text: '',
            source: 'external',
            verified: false,
            warning: `chunk_id ${chunkId} must be resolved from external store by caller`,
        };
    }
}
