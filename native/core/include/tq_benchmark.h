/**
 * TurboQuant Core — Benchmark Framework
 * Ported from TypeScript src/core/benchmark.ts
 */

#ifndef TQ_BENCHMARK_H
#define TQ_BENCHMARK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Configuration ────────────────────────────────────────────────────────────
typedef struct {
    uint32_t dimensions;
    uint32_t vector_count;
    uint8_t  bits_per_value;
    uint8_t  rotation_mode;     // 0=none, 1=fwht_sign
    uint8_t  codebook_type;     // 0=uniform, 1=lloydmax, 2=beta
    uint8_t  use_qjl;
} tq_benchmark_config;

typedef struct {
    float encode_time_ms;
    float decode_time_ms;
    float compression_ratio;
    float encode_throughput;     // vectors per second
    float decode_throughput;     // vectors per second
    float throughput_mb_per_sec;
    float mse;
    float max_error;
} tq_benchmark_metrics;

typedef struct {
    tq_benchmark_config config;
    tq_benchmark_metrics metrics;
} tq_benchmark_result;

// ── Summary ──────────────────────────────────────────────────────────────────
typedef struct {
    uint32_t total_tests;
    float    avg_compression_ratio;
    float    avg_encode_throughput;
    float    avg_decode_throughput;
    float    best_compression_ratio;
    float    best_encode_throughput;
    float    best_decode_throughput;
    float    worst_compression_ratio;
    float    worst_encode_throughput;
    float    worst_decode_throughput;
} tq_benchmark_summary;

// ── Functions ─────────────────────────────────────────────────────────────────
int tq_benchmark_run_single(
    const tq_benchmark_config* cfg,
    tq_benchmark_result* result
);

int tq_benchmark_run_matrix(
    const tq_benchmark_config* configs,
    uint32_t config_count,
    tq_benchmark_result** results_out,
    uint32_t* result_count_out
);

int tq_benchmark_summarize(
    const tq_benchmark_result* results,
    uint32_t result_count,
    tq_benchmark_summary* summary_out
);

void tq_benchmark_free_results(tq_benchmark_result* results, uint32_t count);

#ifdef __cplusplus
}
#endif

#endif // TQ_BENCHMARK_H