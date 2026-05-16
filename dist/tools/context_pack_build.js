import { z } from 'zod';
import { compressVectors } from '../index.js';
import { sha256Text, approximateTokens } from '../cost/fingerprint.js';
import { createTokenHashVectorizer } from '../context/vectorizer.js';
import { buildProvenance } from '../context/provenance.js';
import { buildReplayManifest } from '../context/replay_manifest.js';
import { validateContextFiles, validateContextTotalBytes, validateContextFileBytes, validateContextPathLength, validateChunkBytes, validateInlineChunkBytes, } from '../cost/limits.js';
const ContextPackBuildInputSchema = z.object({
    files: z.array(z.object({
        path: z.string(),
        text: z.string(),
    })),
    dimensions: z.number().min(1).max(8192).default(384),
    chunkBytes: z.number().min(256).max(65536).default(4096),
    bitsPerValue: z.number().min(2).max(8).default(4),
    storageMode: z.enum(['inline_text', 'preview_only', 'external_store']).default('preview_only'),
    maxInlineChunkBytes: z.number().min(0).max(65536).default(4096),
});
function chunkTextForPack(text, maxBytes) {
    const chunks = [];
    let current = '';
    for (const line of text.split(/\r?\n/)) {
        const next = current + line + '\n';
        if (Buffer.byteLength(next, 'utf8') > maxBytes && current.length > 0) {
            chunks.push(current);
            current = '';
        }
        current += line + '\n';
    }
    if (current.trimEnd().length > 0)
        chunks.push(current.trimEnd());
    return chunks;
}
export function turboquantContextPackBuild(rawInput) {
    const input = ContextPackBuildInputSchema.parse(rawInput);
    // Validate limits
    validateContextFiles(input.files);
    validateChunkBytes(input.chunkBytes);
    validateInlineChunkBytes(input.maxInlineChunkBytes);
    const allChunks = [];
    let totalBytes = 0;
    for (const file of input.files) {
        validateContextPathLength(file.path);
        const fileBytes = Buffer.byteLength(file.text, 'utf8');
        validateContextFileBytes(fileBytes);
        totalBytes += fileBytes;
        const chunks = chunkTextForPack(file.text, input.chunkBytes);
        for (const chunk of chunks) {
            allChunks.push({
                path: file.path,
                text: chunk,
                fingerprint: sha256Text(chunk),
                approximateTokens: approximateTokens(chunk),
            });
        }
    }
    validateContextTotalBytes(totalBytes);
    if (allChunks.length === 0) {
        throw new Error('No chunks generated for context pack');
    }
    const vectorizer = createTokenHashVectorizer(input.dimensions);
    const vectors = allChunks.map((c) => vectorizer.embed(c.text));
    const compressed = compressVectors({
        vectors,
        dimensions: input.dimensions,
        bitsPerValue: input.bitsPerValue,
        seed: 42,
    });
    const rootFingerprint = sha256Text(allChunks.map((c) => c.fingerprint).join(''));
    const provenance = buildProvenance({
        vectorizerId: vectorizer.id,
        vectorizerKind: vectorizer.kind,
        dimensions: input.dimensions,
        chunkBytes: input.chunkBytes,
        bitsPerValue: input.bitsPerValue,
        storageMode: input.storageMode,
        rootFingerprint,
        totalInputBytes: totalBytes,
        chunks: allChunks.map((c, idx) => ({
            chunkId: `chunk_${idx}`,
            path: c.path,
            fingerprint: c.fingerprint,
            byteOffset: 0,
            byteLength: Buffer.byteLength(c.text, 'utf8'),
            approximateTokens: c.approximateTokens,
        })),
    });
    const replay = buildReplayManifest({
        provenance,
        compressedDatabaseB64: compressed.compressed_database_b64,
    });
    const manifest = {
        version: 1,
        createdAt: new Date().toISOString(),
        rootFingerprint,
        dimensions: input.dimensions,
        vectorizer: {
            id: vectorizer.id,
            kind: vectorizer.kind,
            dimensions: vectorizer.dimensions,
        },
        provenance,
        replay,
        chunkCount: allChunks.length,
        approximateTokens: allChunks.reduce((s, c) => s + c.approximateTokens, 0),
        compressedDatabaseB64: compressed.compressed_database_b64,
        storageMode: input.storageMode,
        maxInlineChunkBytes: input.maxInlineChunkBytes,
        chunks: allChunks.map((c, idx) => {
            const chunkInfo = {
                id: `chunk_${idx}`,
                path: c.path,
                fingerprint: c.fingerprint,
                approximateTokens: c.approximateTokens,
            };
            if (input.storageMode === 'inline_text' && c.text.length <= input.maxInlineChunkBytes) {
                chunkInfo.text = c.text;
                chunkInfo.textPreview = c.text.slice(0, 200);
            }
            else {
                chunkInfo.textPreview = c.text.slice(0, 200);
            }
            return chunkInfo;
        }),
    };
    const warnings = [
        'Token hashing is deterministic and local; this is not semantic embedding quality.',
        'LEVEL_0 implementation does not store QJL payload.',
        'This tool does not reduce Anthropic API token billing directly.',
    ];
    if (input.storageMode === 'external_store') {
        warnings.push('Context text is not embedded in manifest; caller must resolve chunk_id from external store.');
    }
    return {
        algorithm_level: 'COST_AWARE_BRAIN_V1',
        manifest,
        warnings,
    };
}
