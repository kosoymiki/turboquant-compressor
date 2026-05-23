/**
 * TurboQuant Core — Context Pack
 * Ported from TypeScript src/core/context_pack_v2.ts
 *
 * Context compression for prompts with chunk indexing and search.
 */

#ifndef TQ_CONTEXT_PACK_H
#define TQ_CONTEXT_PACK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Storage modes ────────────────────────────────────────────────────────────
#define TQ_CP_STORAGE_INLINE_TEXT   0
#define TQ_CP_STORAGE_PREVIEW_ONLY   1
#define TQ_CP_STORAGE_EXTERNAL_STORE 2

// ── Manifest header ─────────────────────────────────────────────────────────
typedef struct {
    uint32_t magic;            // 0x54514350 ('TQCP')
    uint32_t version;          // 2
    uint32_t chunk_count;
    uint32_t total_tokens;
    uint8_t  compression_bits;
    uint32_t embeddings_dim;
    uint8_t  storage_mode;
    uint32_t max_inline_chunk_bytes;
    uint64_t compressed_db_size;
    char     root_fingerprint[64];
} tq_context_pack_header;

// ── Chunk info ───────────────────────────────────────────────────────────────
typedef struct {
    char     chunk_id[32];
    char     path[256];
    uint32_t start_line;
    uint32_t end_line;
    char     fingerprint[64];
    uint32_t approximate_tokens;
    uint32_t text_preview_offset;  // offset in pack, 0 if inline_text
    uint32_t text_preview_size;
} tq_context_chunk;

// ── Build config ─────────────────────────────────────────────────────────────
typedef struct {
    uint32_t chunk_count;
    uint32_t embeddings_dim;
    uint8_t  compression_bits;
    uint8_t  storage_mode;
    uint32_t max_inline_chunk_bytes;
} tq_context_pack_config;

// ── Build result ─────────────────────────────────────────────────────────────
typedef struct {
    char*    base64_out;
    size_t   base64_len;
    uint32_t chunk_count;
    char     root_fingerprint[64];
} tq_context_pack_build_result;

// ── Search result ────────────────────────────────────────────────────────────
typedef struct {
    char     chunk_id[32];
    char     path[256];
    float    score;
    char*    text;
    uint32_t text_size;
    uint8_t  source;           // 0=inline, 1=preview, 2=external, 3=missing
} tq_context_search_result;

// ── Functions ─────────────────────────────────────────────────────────────────
int tq_context_pack_build(
    const float* embeddings,      // [chunk_count, embeddings_dim]
    const tq_context_chunk* chunks,
    const tq_context_pack_config* config,
    char** out_base64,
    size_t* out_len
);

int tq_context_pack_search(
    const char* pack_base64,
    size_t pack_base64_len,
    const float* query_embedding,
    uint32_t top_k,
    tq_context_search_result** results_out,
    uint32_t* result_count_out
);

int tq_context_pack_parse_header(
    const char* pack_base64,
    size_t pack_base64_len,
    tq_context_pack_header* header_out
);

void tq_context_pack_free_results(
    tq_context_search_result* results,
    uint32_t count
);

void tq_context_pack_free_build_result(
    tq_context_pack_build_result* result
);

#ifdef __cplusplus
}
#endif

#endif // TQ_CONTEXT_PACK_H