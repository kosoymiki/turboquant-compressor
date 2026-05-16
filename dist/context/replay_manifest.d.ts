/**
 * Replay manifest for a context pack build.
 * Contains everything needed to reproduce the exact same manifest from the
 * same source files, without re-running the full build pipeline.
 */
import type { ContextPackProvenance } from './provenance.js';
export interface ReplayManifest {
    schemaVersion: 1;
    manifestVersion: number;
    provenance: ContextPackProvenance;
    compressedDatabaseB64: string;
    replayCommand: string;
}
export declare function buildReplayManifest(params: {
    provenance: ContextPackProvenance;
    compressedDatabaseB64: string;
}): ReplayManifest;
export declare function verifyReplayManifest(manifest: ReplayManifest, expectedRootFingerprint: string): {
    ok: boolean;
    error?: string;
};
