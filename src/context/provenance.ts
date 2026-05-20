/**
 * Provenance record for a context pack build.
 * Captures the inputs and environment that produced a manifest so the build
 * can be audited and replayed deterministically.
 */

export interface ChunkProvenance {
  chunkId: string;
  path: string;
  fingerprint: string;
  byteOffset: number;
  byteLength: number;
  approximateTokens: number;
}

export interface ContextPackProvenance {
  schemaVersion: 1;
  builtAt: string;
  vectorizerId: string;
  vectorizerKind: 'token_hash' | 'hashed_tfidf' | 'semantic_embedding';
  dimensions: number;
  chunkBytes: number;
  bitsPerValue: number;
  storageMode: 'inline_text' | 'preview_only' | 'external_store';
  rootFingerprint: string;
  totalInputBytes: number;
  chunkCount: number;
  chunks: ChunkProvenance[];
}

export function buildProvenance(params: {
  vectorizerId: string;
  vectorizerKind: 'token_hash' | 'hashed_tfidf' | 'semantic_embedding';
  dimensions: number;
  chunkBytes: number;
  bitsPerValue: number;
  storageMode: 'inline_text' | 'preview_only' | 'external_store';
  rootFingerprint: string;
  totalInputBytes: number;
  chunks: ChunkProvenance[];
}): ContextPackProvenance {
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
