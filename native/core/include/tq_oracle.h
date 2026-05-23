/**
 * TurboQuant Core — Quality Oracle
 * Ported from TypeScript src/core/oracle.ts
 */

#ifndef TQ_ORACLE_H
#define TQ_ORACLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Data characteristics ──────────────────────────────────────────────────────
typedef struct {
    float min;
    float max;
    float mean;
    float variance;
    float std_dev;
    float skewness;
    float kurtosis;
    float entropy;
    float dynamic_range;
} tq_data_characteristics;

// ── Oracle recommendation ────────────────────────────────────────────────────
typedef struct {
    uint8_t  bits_per_value;
    float    compression_ratio;
    float    estimated_mse;
    float    confidence;
} tq_oracle_recommendation;

// ── Functions ─────────────────────────────────────────────────────────────────
void tq_oracle_analyze(const float* data, uint32_t count, tq_data_characteristics* chars);
float tq_oracle_estimate_mse(const tq_data_characteristics* chars, uint8_t bits);
float tq_oracle_estimate_compression_ratio(const tq_data_characteristics* chars, uint8_t bits, uint8_t original_bits);
float tq_oracle_calculate_confidence(const tq_data_characteristics* chars, uint8_t bits);

int tq_oracle_recommend(
    const float* data,
    uint32_t count,
    float target_mse,
    float target_compression,
    tq_oracle_recommendation* result
);

uint8_t tq_oracle_optimal_bits(const float* data, uint32_t count, float tolerance);
float tq_oracle_compression_ratio(const float* data, uint32_t count, uint8_t bits);
float tq_oracle_predict_error(const float* data, uint32_t count, uint8_t bits);

#ifdef __cplusplus
}
#endif

#endif // TQ_ORACLE_H