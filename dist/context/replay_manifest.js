/**
 * Replay manifest for a context pack build.
 * Contains everything needed to reproduce the exact same manifest from the
 * same source files, without re-running the full build pipeline.
 */
export function buildReplayManifest(params) {
    const cmd = [
        'turboquant_context_pack_build',
        `--dimensions ${params.provenance.dimensions}`,
        `--chunk-bytes ${params.provenance.chunkBytes}`,
        `--bits-per-value ${params.provenance.bitsPerValue}`,
        `--storage-mode ${params.provenance.storageMode}`,
        `--vectorizer ${params.provenance.vectorizerId}`,
    ].join(' ');
    return {
        schemaVersion: 1,
        manifestVersion: 1,
        provenance: params.provenance,
        compressedDatabaseB64: params.compressedDatabaseB64,
        replayCommand: cmd,
    };
}
export function verifyReplayManifest(manifest, expectedRootFingerprint) {
    if (manifest.schemaVersion !== 1) {
        return { ok: false, error: `unknown schemaVersion ${manifest.schemaVersion}` };
    }
    if (manifest.provenance.rootFingerprint !== expectedRootFingerprint) {
        return {
            ok: false,
            error: `rootFingerprint mismatch: expected ${expectedRootFingerprint}, got ${manifest.provenance.rootFingerprint}`,
        };
    }
    if (!manifest.compressedDatabaseB64 || manifest.compressedDatabaseB64.length === 0) {
        return { ok: false, error: 'compressedDatabaseB64 is empty' };
    }
    return { ok: true };
}
