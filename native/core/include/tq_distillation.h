/**
 * TurboQuant Core — Knowledge Distillation
 * Teacher-student quantization distillation
 * Ported from distillation_for_quantization.ts
 */

#ifndef TQ_DISTILLATION_H
#define TQ_DISTILLATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Distillation result ─────────────────────────────────────────────────────
typedef struct {
    float temperature;
    uint8_t quant_bits;
    float quant_step_size;
    float teacher_entropy_mean;
    float initial_kl_mean;
    float updated_kl_mean;
    float initial_anchor_mse;
    float updated_anchor_mse;
    float initial_composite_loss;
    float updated_composite_loss;
    float update_norm;
    float alpha;  // KL weight
    float beta;   // anchor MSE weight
    float learning_rate;
} tq_distillation_result;

// ── Functions ─────────────────────────────────────────────────────────────────
void tq_distillation_make_teacher_logits(
    float* out, uint32_t batch_size, uint32_t vocab_size,
    float phase, float amplitude
);

void tq_distillation_make_student_base(
    float* out, uint32_t batch_size, uint32_t vocab_size,
    float phase, float amplitude
);

void tq_distillation_softmax_row(
    const float* logits, uint32_t offset, uint32_t width,
    float temperature, float* out
);

float tq_distillation_kl_divergence(
    const float* p, const float* q, uint32_t size
);

float tq_distillation_mse(const float* a, const float* b, uint32_t size);

void tq_distillation_quantize_symmetric(
    const float* values, uint32_t count, uint8_t bits,
    float* out, float* step_size_out
);

int tq_distillation_run(
    uint32_t batch_size, uint32_t vocab_size,
    uint8_t bits, float temperature,
    float alpha, float beta, float learning_rate,
    tq_distillation_result* result
);

#ifdef __cplusplus
}
#endif

#endif // TQ_DISTILLATION_H