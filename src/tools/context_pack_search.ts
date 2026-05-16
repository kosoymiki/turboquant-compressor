import { z } from 'zod';
import type { ContextPackSearchResult, ContextPackManifest } from '../cost/types.js';
import { searchVectors } from '../index.js';
import { createTokenHashVectorizer } from '../context/vectorizer.js';
import { validateTopK } from '../cost/limits.js';

const ExternalStoreEntrySchema = z.object({
  chunk_id: z.string(),
  fingerprint: z.string(),
  text: z.string().max(1000000),
});

const ExternalStoreInputSchema = z.object({
  kind: z.literal('inline_entries'),
  entries: z.array(ExternalStoreEntrySchema).max(1000),
});

const ContextPackSearchInputSchema = z.object({
  manifest: z.custom<ContextPackManifest>(),
  query: z.string(),
  top_k: z.number().min(1).max(100).default(5),
  externalStore: ExternalStoreInputSchema.optional(),
});

function resolveExternalStoreEntry(
  chunkId: string,
  fingerprint: string,
  externalStore: z.infer<typeof ExternalStoreInputSchema> | undefined
): { text: string; source: 'external' | 'missing'; text_available: boolean; verified: boolean; warning?: string } {
  if (!externalStore) {
    return {
      text: '',
      source: 'missing',
      text_available: false,
      verified: false,
      warning: 'No externalStore provided for external_store manifest.'
    };
  }
  const entry = externalStore.entries.find((e) => e.chunk_id === chunkId);
  if (!entry) {
    return {
      text: '',
      source: 'missing',
      text_available: false,
      verified: false,
      warning: `chunk_id ${chunkId} not found in externalStore entries.`
    };
  }
  if (entry.fingerprint !== fingerprint) {
    return {
      text: entry.text,
      source: 'external',
      text_available: true,
      verified: false,
      warning: `Fingerprint mismatch for chunk_id ${chunkId}: manifest=${fingerprint} store=${entry.fingerprint}`
    };
  }
  return {
    text: entry.text,
    source: 'external',
    text_available: true,
    verified: true
  };
}

export function turboquantContextPackSearch(
  rawInput: unknown
): ContextPackSearchResult {
  const input = ContextPackSearchInputSchema.parse(rawInput);

  validateTopK(input.top_k);

  const vectorizer = createTokenHashVectorizer(input.manifest.dimensions);
  const queryVector = vectorizer.embed(input.query);

  const searchResult = searchVectors({
    compressed_database_b64: input.manifest.compressedDatabaseB64,
    query_vector: queryVector,
    top_k: input.top_k,
    metric: 'cosine',
  });

  const results = searchResult.results.map((r) => {
    const chunk = input.manifest.chunks[r.index];
    if (!chunk) {
      return {
        path: 'unknown',
        chunk_id: `chunk_${r.index}`,
        score: r.score,
        text: '',
        text_available: false,
        source: 'missing' as const,
        verified: false,
        warning: `No chunk at index ${r.index} in manifest.`,
      };
    }

    const mode = input.manifest.storageMode;

    if (mode === 'inline_text' && chunk.text) {
      return {
        path: chunk.path,
        chunk_id: chunk.id,
        score: r.score,
        text: chunk.text,
        text_available: true,
        source: 'inline' as const,
        verified: true,
      };
    }

    if (mode === 'preview_only' || (mode === 'inline_text' && !chunk.text && chunk.textPreview)) {
      return {
        path: chunk.path,
        chunk_id: chunk.id,
        score: r.score,
        text: chunk.textPreview ?? '',
        text_available: false,
        source: 'preview' as const,
        verified: false,
        warning: 'Only text preview available; rebuild with storageMode=inline_text for full text.',
      };
    }

    if (mode === 'external_store') {
      const resolved = resolveExternalStoreEntry(chunk.id, chunk.fingerprint, input.externalStore);
      return {
        path: chunk.path,
        chunk_id: chunk.id,
        score: r.score,
        text: resolved.text,
        text_available: resolved.verified,
        source: 'external' as const,
        verified: resolved.verified,
        ...(resolved.warning ? { warning: resolved.warning } : {}),
      };
    }

    // Fallback: inline_text mode but no text or preview
    return {
      path: chunk.path,
      chunk_id: chunk.id,
      score: r.score,
      text: '',
      text_available: false,
      source: 'missing' as const,
      verified: false,
      warning: 'No text or preview available for this chunk.',
    };
  });

  const warnings: string[] = [
    'Local deterministic vectorizer is not a substitute for semantic embeddings.',
    'This tool does not reduce Anthropic API token billing directly.',
    'Use context_pack_build to create the index first.',
  ];

  if (input.manifest.storageMode === 'external_store') {
    if (!input.externalStore) {
      warnings.push(
        'Manifest storageMode=external_store but no externalStore provided; text will be empty. Pass externalStore={kind:"inline_entries",entries:[...]} to resolve.'
      );
    } else {
      warnings.push(
        'External store entries resolved inline. Fingerprint verification applied per chunk.'
      );
    }
  }

  if (input.manifest.storageMode === 'preview_only') {
    warnings.push(
      'Only text previews are available. For full text, rebuild with storageMode=inline_text.'
    );
  }

  return {
    algorithm_level: 'COST_AWARE_BRAIN_V1',
    results,
    storage_mode: input.manifest.storageMode,
    warnings,
  };
}
