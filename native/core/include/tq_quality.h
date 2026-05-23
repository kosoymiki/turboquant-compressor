/**
 * TurboQuant Core — Quality Benchmarks
 * v4.5.3
 *
 * Metrics:
 *   - MSE per coordinate (quantization distortion)
 *   - Recall@k (self-match, nearest-neighbor accuracy)
 *   - Compression ratio (original vs compressed bytes)
 *   - Search latency (ns per query)
 *   - QJL agreement (self-match agreement fraction)
 *
 * Ported from TypeScript src/core/quality.ts
 */

#ifndef TQ_QUALITY_H
#define TQ_QUALITY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Benchmark configuration ─────────────────────────────────────────────────
typedef struct {
    uint32_t  top_k;           // recall@top_k
    uint32_t  iterations;        // number of random queries to run
    uint8_t   use_qjl;           // include QJL in search
    uint8_t   codebook_type;      // TQ_CODEBOOK_UNIFORM or TQ_CODEBOOK_BETA
} tq_quality_config;

// ── Benchmark results ────────────────────────────────────────────────────────
typedef struct {
    float     compression_ratio;
    float     recall_at_1;
    float     recall_at_5;
    float     mse_per_coord;
    float     avg_search_ns;
    float     qjl_self_agreement;  // self-match QJL agreement (should be > 0.8)
    uint32_t  total_queries;
    uint32_t  total_db_vectors;
    uint32_t  dimensions;
    uint8_t   codebook_type;
} tq_quality_result;

// Run quality benchmark on pre-compressed database
// vectors: original float vectors (for ground truth)
// N, dims: database size
// db_base64: compressed database
// cfg: quality configuration
// Returns: result struct (caller owns nothing — all inline)
tq_quality_result tq_quality_benchmark(
    const float* vectors,
    uint32_t N,
    uint32_t dims,
    const char* db_base64,
    tq_quality_config cfg
);

// Lightweight MSE only (no search)
float tq_quality_mse(
    const float* original_vectors,
    uint32_t N,
    uint32_t dims,
    uint8_t codebook_type
);

#ifdef __cplusplus
}
#endif

#endif // TQ_QUALITY_H