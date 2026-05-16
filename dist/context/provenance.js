/**
 * Provenance record for a context pack build.
 * Captures the inputs and environment that produced a manifest so the build
 * can be audited and replayed deterministically.
 */
export function buildProvenance(params) {
    return {
        schemaVersion: 1,
        builtAt: new Date().toISOString(),
        vectorizerId: params.vectorizerId,
        vectorizerKind: params.vectorizerKind,
        dimensions: params.dimensions,
        chunkBytes: params.chunkBytes,
        bitsPerValue: params.bitsPerValue,
        storageMode: params.storageMode,
        rootFingerprint: params.rootFingerprint,
        totalInputBytes: params.totalInputBytes,
        chunkCount: params.chunks.length,
        chunks: params.chunks,
    };
}
