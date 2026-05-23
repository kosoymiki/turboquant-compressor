/**
 * TurboQuant Core — Context Pack Implementation
 */

#include "tq_context_pack.h"
#include "tq_core.h"
#include "tq_base64.h"
#include "tq_kernel_inline.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>

// Magic number for TQCP format
#define TQCP_MAGIC 0x54514350

// ── Build context pack ────────────────────────────────────────────────────────
int tq_context_pack_build(
    const float* embeddings,
    const tq_context_chunk* chunks,
    const tq_context_pack_config* config,
    char** out_base64,
    size_t* out_len
) {
    if (!embeddings || !chunks || !config || !out_base64 || !out_len) {
        return TQ_ERR_NULL;
    }

    uint32_t chunk_count = config->chunk_count;
    uint32_t dim = config->embeddings_dim;
    uint8_t bits = config->compression_bits ? config->compression_bits : 4;

    // Compress embeddings using TurboQuant
    size_t est = tq_estimate_compressed_size(chunk_count, dim, bits, 0);
    std::vector<char> compressed(est * 2);
    size_t comp_len = est * 2;

    int r = tq_compress(embeddings, chunk_count, dim, bits, 42, 0, 0,
                        compressed.data(), &comp_len, nullptr);
    if (r != TQ_OK) return r;

    // Generate root fingerprint (simple hash of first/last chunk IDs)
    char root_fp[64] = {0};
    if (chunk_count > 0) {
        uint32_t h = 0;
        const char* first_id = chunks[0].chunk_id;
        while (*first_id) { h = h * 31 + *first_id++; }
        const char* last_id = chunks[chunk_count - 1].chunk_id;
        while (*last_id) { h = h * 17 + *last_id++; }
        snprintf(root_fp, sizeof(root_fp), "%08x%08x", h, chunk_count);
    }

    // Build binary pack: header + chunks metadata + compressed db
    size_t header_size = sizeof(tq_context_pack_header);
    size_t chunks_meta_size = chunk_count * sizeof(tq_context_chunk);
    size_t total_size = header_size + chunks_meta_size + comp_len;

    std::vector<uint8_t> pack_bin(total_size);
    memset(pack_bin.data(), 0, total_size);

    // Pack header
    tq_context_pack_header* hdr = (tq_context_pack_header*)pack_bin.data();
    hdr->magic = TQCP_MAGIC;
    hdr->version = 2;
    hdr->chunk_count = chunk_count;
    hdr->total_tokens = 0;  // computed from chunks
    hdr->compression_bits = bits;
    hdr->embeddings_dim = dim;
    hdr->storage_mode = config->storage_mode;
    hdr->max_inline_chunk_bytes = config->max_inline_chunk_bytes;
    hdr->compressed_db_size = comp_len;
    strncpy(hdr->root_fingerprint, root_fp, 63);
    hdr->root_fingerprint[63] = '\0';

    // Compute total tokens
    uint32_t total_tokens = 0;
    for (uint32_t i = 0; i < chunk_count; i++) {
        total_tokens += chunks[i].approximate_tokens;
    }
    hdr->total_tokens = total_tokens;

    // Pack chunks metadata
    uint8_t* chunks_ptr = pack_bin.data() + header_size;
    memcpy(chunks_ptr, chunks, chunks_meta_size);

    // Pack compressed database
    uint8_t* db_ptr = pack_bin.data() + header_size + chunks_meta_size;
    memcpy(db_ptr, compressed.data(), comp_len);

    // Encode to base64
    std::string b64 = tq::core::base64::encode(
        (const uint8_t*)pack_bin.data(), pack_bin.size()
    );

    char* out = (char*)malloc(b64.size() + 1);
    if (!out) return TQ_ERR_ALLOC;
    memcpy(out, b64.c_str(), b64.size());
    out[b64.size()] = '\0';

    *out_base64 = out;
    *out_len = b64.size();
    return TQ_OK;
}

// ── Parse header ──────────────────────────────────────────────────────────────
int tq_context_pack_parse_header(
    const char* pack_base64,
    size_t pack_base64_len,
    tq_context_pack_header* header_out
) {
    if (!pack_base64 || !header_out) return TQ_ERR_NULL;

    std::string b64(pack_base64, pack_base64_len);
    std::vector<uint8_t> bin = tq::core::base64::decode(b64);

    if (bin.size() < sizeof(tq_context_pack_header)) {
        return TQ_ERR_INTERNAL;
    }

    memcpy(header_out, bin.data(), sizeof(tq_context_pack_header));

    if (header_out->magic != TQCP_MAGIC) {
        return TQ_ERR_INTERNAL;
    }

    return TQ_OK;
}

// ── Search context pack ──────────────────────────────────────────────────────
int tq_context_pack_search(
    const char* pack_base64,
    size_t pack_base64_len,
    const float* query_embedding,
    uint32_t top_k,
    tq_context_search_result** results_out,
    uint32_t* result_count_out
) {
    if (!pack_base64 || !query_embedding || !results_out || !result_count_out) {
        return TQ_ERR_NULL;
    }

    // Parse header
    tq_context_pack_header header;
    int r = tq_context_pack_parse_header(pack_base64, pack_base64_len, &header);
    if (r != TQ_OK) return r;

    uint32_t chunk_count = header.chunk_count;
    uint32_t dim = header.embeddings_dim;

    // Extract compressed DB
    size_t header_size = sizeof(tq_context_pack_header);
    size_t chunks_meta_size = chunk_count * sizeof(tq_context_chunk);

    std::string b64(pack_base64, pack_base64_len);
    std::vector<uint8_t> bin = tq::core::base64::decode(b64);

    if (bin.size() < header_size + chunks_meta_size + header.compressed_db_size) {
        return TQ_ERR_INTERNAL;
    }

    const char* db_base64 = (const char*)bin.data() + header_size + chunks_meta_size;
    size_t db_len = header.compressed_db_size;
    (void)db_len;  // reserved for future use

    // Search using TurboQuant
    uint32_t* indices = (uint32_t*)malloc(top_k * sizeof(uint32_t));
    float* scores = (float*)malloc(top_k * sizeof(float));
    if (!indices || !scores) {
        if (indices) free(indices);
        if (scores) free(scores);
        return TQ_ERR_ALLOC;
    }

    r = tq_search(db_base64, query_embedding, dim, top_k, 0, 0, indices, scores, nullptr);
    if (r != TQ_OK) {
        free(indices); free(scores);
        return r;
    }

    // Build results
    tq_context_search_result* results = (tq_context_search_result*)
        malloc(top_k * sizeof(tq_context_search_result));
    if (!results) {
        free(indices); free(scores);
        return TQ_ERR_ALLOC;
    }

    const tq_context_chunk* chunks = (const tq_context_chunk*)(bin.data() + header_size);

    for (uint32_t i = 0; i < top_k && i < chunk_count; i++) {
        uint32_t idx = indices[i];
        memset(&results[i], 0, sizeof(tq_context_search_result));

        if (idx < chunk_count) {
            strncpy(results[i].chunk_id, chunks[idx].chunk_id, 31);
            strncpy(results[i].path, chunks[idx].path, 255);
            results[i].score = scores[i];
            results[i].source = header.storage_mode;

            // Text from preview if available
            if (chunks[idx].text_preview_size > 0 && chunks[idx].text_preview_offset > 0) {
                size_t text_off = header_size + chunks_meta_size + header.compressed_db_size
                                  + chunks[idx].text_preview_offset;
                if (text_off + chunks[idx].text_preview_size <= bin.size()) {
                    results[i].text = (char*)malloc(chunks[idx].text_preview_size + 1);
                    if (results[i].text) {
                        memcpy(results[i].text, bin.data() + text_off, chunks[idx].text_preview_size);
                        results[i].text[chunks[idx].text_preview_size] = '\0';
                        results[i].text_size = chunks[idx].text_preview_size;
                    }
                }
            }
        }
    }

    free(indices);
    free(scores);

    *results_out = results;
    *result_count_out = top_k < chunk_count ? top_k : chunk_count;
    return TQ_OK;
}

// ── Cleanup ──────────────────────────────────────────────────────────────────
void tq_context_pack_free_results(tq_context_search_result* results, uint32_t count) {
    if (!results) return;
    for (uint32_t i = 0; i < count; i++) {
        if (results[i].text) free(results[i].text);
    }
    free(results);
}

void tq_context_pack_free_build_result(tq_context_pack_build_result* result) {
    if (!result) return;
    if (result->base64_out) free(result->base64_out);
}