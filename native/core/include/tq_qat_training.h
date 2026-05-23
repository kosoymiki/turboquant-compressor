/**
 * TurboQuant Core — QAT Training Loop
 * Full Quantization-Aware Training Pipeline
 * Combines calibration, LSQ, and distillation
 */

#ifndef TQ_QAT_TRAINING_H
#define TQ_QAT_TRAINING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── QAT config ───────────────────────────────────────────────────────────
typedef struct {
    uint32_t batch_size;
    uint32_t vocab_size;
    uint8_t bits;
    float temperature;
    float alpha;    // KL weight
    float beta;      // anchor MSE weight
    float learning_rate;
    uint32_t epochs;
} tq_qat_config;

// ── QAT result ───────────────────────────────────────────────────────────
typedef struct {
    float initial_quant_loss;
    float post_lsq_loss;
    float initial_distillation_loss;
    float post_distillation_loss;
    float aggregate_initial_loss;
    float aggregate_final_loss;
    float aggregate_improvement;
    uint32_t epochs_completed;
    uint8_t calibration_ready;
    uint8_t lsq_ready;
    uint8_t distillation_ready;
    uint8_t loop_ready;
} tq_qat_result;

// ── Functions ─────────────────────────────────────────────────────────────
int tq_qat_run(
    const float* activation_data, uint32_t activation_count,
    const float* weight_data, uint32_t weight_count,
    const float* kv_data, uint32_t kv_count,
    const float* teacher_data, const float* student_data, uint32_t data_count,
    const tq_qat_config* config,
    tq_qat_result* result
);

#ifdef __cplusplus
}
#endif

#endif // TQ_QAT_TRAINING_H