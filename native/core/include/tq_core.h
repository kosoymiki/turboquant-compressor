/**
 * TurboQuant Core C++ Library — High-level compression + search API (C ABI)
 *
 * End-to-end pipeline: rotation → quantization → QJL sign sketch → format encode
 * Integrates: libtq_kernel.a (CPU kernels) + libtq_gpu_pipeline.a (GPU)
 * v4.6.1: P0 Hadamard QJL (Zandieh ICML 2024), P1 2-bit Beta, P2 ProductQuantizer
 */

#ifndef TQ_CORE_H
#define TQ_CORE_H

#include <stddef.h>
#include <stdint.h>

#define TQ_CORE_VERSION "4.6.1"
#define TQ_CORE_MAGIC   0x544D4331

#define TQ_CODEBOOK_UNIFORM  0  // uniform symmetric (idx/(L-1))*2-1, fast, no precompute
#define TQ_CODEBOOK_BETA     1  // Beta(d/2, d/2) Lloyd-Max, precomputed codebook

#define TQ_OK            0
#define TQ_ERR_NULL     -1
#define TQ_ERR_DIM      -2
#define TQ_ERR_ALLOC     -3
#define TQ_ERR_FORMAT   -4
#define TQ_ERR_BASE64   -5
#define TQ_ERR_INTERNAL -6

typedef struct {
    int32_t   magic;
    int32_t   version;
    uint32_t  original_bytes;
    uint32_t  compressed_bytes;
    float     compression_ratio;
    uint32_t  vector_count;
    uint32_t  dimensions;
    uint32_t  padded_dimensions;
    uint8_t   bits_per_value;
    uint8_t   codebook_type;   // TQ_CODEBOOK_UNIFORM or TQ_CODEBOOK_BETA
    uint32_t  qjl_bytes;
    uint8_t   qjl_dims;
    uint8_t   has_qjl;
    uint32_t  compress_ns;
} tq_compress_stats;

typedef struct {
    uint32_t  vector_count;
    uint32_t  dimensions;
    uint32_t  top_k;
    uint32_t  searched_count;
    uint32_t  qjl_applied;
    float     first_score;
    uint32_t  search_ns;
} tq_search_stats;

typedef struct {
    uint32_t  magic;
    uint32_t  version;
    uint32_t  dimensions;
    uint32_t  padded_dimensions;
    uint32_t  vector_count;
    uint8_t   bits_per_value;
    uint32_t  rotation_seed;
    uint8_t   flags;
    uint8_t   codebook_type;   // TQ_CODEBOOK_UNIFORM or TQ_CODEBOOK_BETA
    uint8_t   qjl_dims;
    uint32_t  qjl_bytes;
    uint8_t   has_qjl;
} tq_db_header;

#ifdef __cplusplus
extern "C" {
#endif

int tq_compress(
    const float* vectors, uint32_t N, uint32_t dims, uint8_t bits,
    uint32_t seed, uint8_t qjl_dims, uint8_t codebook_type,
    char* out_base64, size_t* out_len,
    tq_compress_stats* stats
);

int tq_search(
    const char* db_base64,
    const float* query, uint32_t dims, uint32_t k,
    uint8_t metric, uint8_t use_qjl,
    uint32_t* out_indices, float* out_scores,
    tq_search_stats* stats
);

int tq_decode_header(const char* db_base64, tq_db_header* header);
const char* tq_core_version(void);
size_t tq_estimate_compressed_size(uint32_t N, uint32_t dims, uint8_t bits, uint8_t qjl_dims);

#ifdef __cplusplus
}
#endif

#endif
