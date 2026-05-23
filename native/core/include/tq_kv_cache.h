/**
 * TurboQuant Core — KV Cache Compression
 *
 * v4.6.1: P0 Hadamard QJL, P1 2-bit Beta, P2 ProductQuantizer
 *
 * Features:
 *   - Mixed KV types (different bits for K and V)
 *   - Block size 128 for cache-local scoring
 *   - Precomputed scaled centroids per block
 *   - Sparse V thresholding for low-contribution tokens
 *   - q8 query quantization path
 *   - Tiled token processing (TILE_SIZE=64)
 *   - RotateKV: outlier-aware channel reordering
 */

#ifndef TQ_KV_CACHE_H
#define TQ_KV_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Constants ────────────────────────────────────────────────────────────────
#define TQ_KV_BLOCK_SIZE          128
#define TQ_KV_TILE_SIZE           64
#define TQ_KV_SPARSE_DEFAULT      0.001f
#define TQ_KV_SINK_TOKENS         4
#define TQ_KV_MAX_HEAD_DIM        8192
#define TQ_KV_MAX_SEQ_LEN         128000

// ── Structures ──────────────────────────────────────────────────────────────
typedef struct {
    uint8_t*  indices;     // packed quantized key indices
    float*    norms;       // L2 norms of keys
    uint32_t  num_tokens;
    uint32_t  head_dim;
    uint8_t   bits;
} tq_key_quantized;

typedef struct {
    uint8_t*  data;        // packed quantized values
    float*    scales;     // per-group scales
    float*    zeros;      // per-group zeros
    uint32_t  num_tokens;
    uint32_t  head_dim;
    uint32_t  group_size;
    uint8_t   bits;
} tq_value_quantized;

typedef struct {
    uint32_t  head_dim;
    uint8_t   key_bits;
    uint8_t   value_bits;
    uint32_t  value_group_size;
    uint32_t  buffer_size;
    float     sparse_threshold;
    uint8_t   use_q8_query;
    uint8_t   outlier_aware_rotation;
    uint32_t  sink_tokens;
    uint32_t  rotation_seed;
} tq_kv_cache_config;

typedef struct {
    float   sparse_skip_ratio;
    uint64_t sparse_skipped;
    uint64_t total_scores;
    uint32_t seq_len;
} tq_kv_stats;

typedef struct {
    uint64_t quantized_keys_bytes;
    uint64_t quantized_values_bytes;
    uint64_t buffer_bytes;
    uint64_t total_bytes;
} tq_kv_memory;

// ── Cache lifecycle ─────────────────────────────────────────────────────────
int tq_kv_cache_create(tq_kv_cache_config* config, void** cache_out);
void tq_kv_cache_destroy(void* cache);

// ── Operations ─────────────────────────────────────────────────────────────
int tq_kv_cache_prefill(
    void* cache,
    const float* keys,      // [num_tokens, head_dim]
    const float* values,   // [num_tokens, head_dim]
    uint32_t num_tokens
);

int tq_kv_cache_append(
    void* cache,
    const float* key,      // [head_dim]
    const float* value     // [head_dim]
);

// ── Attention scoring ───────────────────────────────────────────────────────
int tq_kv_cache_attention(
    void* cache,
    const float* query,     // [head_dim]
    float* output          // [head_dim] output
);

// ── Decode ─────────────────────────────────────────────────────────────────
int tq_kv_cache_dequantize_keys(
    void* cache,
    float* keys_out        // [num_tokens, head_dim]
);

// ── Stats ───────────────────────────────────────────────────────────────────
void tq_kv_cache_get_stats(void* cache, tq_kv_stats* stats);
void tq_kv_cache_get_memory(void* cache, tq_kv_memory* mem);
uint32_t tq_kv_cache_get_seq_len(void* cache);

// ── Binary serialization ────────────────────────────────────────────────────
int tq_kv_cache_serialize(void* cache, char* out_base64, size_t* out_len);
int tq_kv_cache_deserialize(void** cache_out, const char* base64, size_t base64_len);

#ifdef __cplusplus
}
#endif

#endif // TQ_KV_CACHE_H